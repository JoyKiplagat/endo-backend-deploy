import os
import gc
import joblib
from pathlib import Path
import numpy as np
from PIL import Image
import cv2
import torch
import torch.nn as nn
from google import genai
from dotenv import load_dotenv

# Limit PyTorch CPU threads to prevent CPU & RAM spikes on Render
torch.set_num_threads(1)

# Locate project root dynamically
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env file explicitly
env_path = BASE_DIR / ".env"
load_dotenv(dotenv_path=env_path)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
gemini_client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None

# Model output paths using absolute BASE_DIR
MRI_CHECKPOINT_PATH = BASE_DIR / "outputs" / "mri_best_model.pt"
MRI_CALIBRATION_PATH = BASE_DIR / "outputs" / "mri_calibration.pkl"
LAPARO_CHECKPOINT_PATH = BASE_DIR / "outputs" / "mri_best_model" / "data.pt"

IMG_SIZE = 224

# Cache for single loaded model
_ACTIVE_MODALITY = None
_ACTIVE_MODEL = None
_ACTIVE_PLATT = None


def build_efficientnet_b0_head(dropout: float = 0.3) -> nn.Module:
    import timm
    model = timm.create_model('efficientnet_b0', pretrained=False)
    in_feat = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(in_feat, 1))
    return model.to(torch.device('cpu'))


def _get_model_for_modality(modality: str):
    """Loads ONLY the requested model to save RAM on Render (512MB limit)."""
    global _ACTIVE_MODALITY, _ACTIVE_MODEL, _ACTIVE_PLATT

    if _ACTIVE_MODALITY == modality and _ACTIVE_MODEL is not None:
        return _ACTIVE_MODEL, _ACTIVE_PLATT

    # Free previous model from RAM before loading a new one
    _ACTIVE_MODEL = None
    _ACTIVE_PLATT = None
    gc.collect()

    device = torch.device('cpu')

    if modality == 'mri':
        model = build_efficientnet_b0_head()
        if MRI_CHECKPOINT_PATH.exists():
            ckpt = torch.load(MRI_CHECKPOINT_PATH, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            del ckpt
        model.eval()
        platt = joblib.load(MRI_CALIBRATION_PATH)['platt'] if MRI_CALIBRATION_PATH.exists() else None
    else:
        model = build_efficientnet_b0_head()
        if LAPARO_CHECKPOINT_PATH.exists():
            ckpt = torch.load(LAPARO_CHECKPOINT_PATH, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            del ckpt
        model.eval()
        platt = None

    _ACTIVE_MODALITY = modality
    _ACTIVE_MODEL = model
    _ACTIVE_PLATT = platt
    gc.collect()

    return _ACTIVE_MODEL, _ACTIVE_PLATT


def detect_modality(image_path, channel_diff_threshold: float = 5.0) -> str:
    with Image.open(image_path) as raw_img:
        img = np.array(raw_img.convert('RGB'), dtype=np.float32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mean_channel_diff = (np.abs(r - g).mean() + np.abs(g - b).mean() + np.abs(r - b).mean()) / 3
    return 'laparoscopy' if mean_channel_diff > channel_diff_threshold else 'mri'


def preprocess_for_model(image_path, modality: str):
    with Image.open(image_path) as raw_img:
        if modality == 'mri':
            raw = np.array(raw_img.convert('L'), dtype=np.float32)
            sl = raw / 255.0 * 2.0 - 1.0
            if sl.shape[0] != IMG_SIZE or sl.shape[1] != IMG_SIZE:
                from scipy.ndimage import zoom as scipy_zoom
                sl = scipy_zoom(sl, (IMG_SIZE / sl.shape[0], IMG_SIZE / sl.shape[1]), order=1)
            return torch.tensor(np.stack([sl, sl, sl], 0), dtype=torch.float32)
        else:
            img = np.array(raw_img.convert('RGB').resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_norm = (img - mean) / std
            return torch.tensor(img_norm.transpose(2, 0, 1), dtype=torch.float32)


def make_grad_cam(model, target_layer):
    activations, gradients = {}, {}

    def fwd_hook(module, inp, out): activations['feat'] = out.detach()
    def bwd_hook(module, g_in, g_out): gradients['feat'] = g_out[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    def grad_cam(img_tensor):
        model.eval()
        img = img_tensor.unsqueeze(0).to(torch.device('cpu')).requires_grad_(True)
        out = model(img)
        model.zero_grad()
        out.backward()

        grads, acts = gradients['feat'].squeeze(0), activations['feat'].squeeze(0)
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = (weights * acts).sum(dim=0).cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam /= cam.max()

        del img, out, grads, acts, weights
        gc.collect()

        return np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)) / 255.0

    return grad_cam, (h1, h2)


def describe_attention_region(cam: np.ndarray, threshold: float = 0.6) -> str:
    h, w = cam.shape
    ys, xs = np.where(cam > threshold)
    if len(ys) == 0: return "no single strongly localised region"
    cy, cx = ys.mean() / h, xs.mean() / w
    vert = 'upper' if cy < 0.4 else ('lower' if cy > 0.6 else 'central')
    horiz = 'left' if cx < 0.4 else ('right' if cx > 0.6 else 'central')
    return 'central' if (vert == 'central' and horiz == 'central') else f'{vert}-{horiz}'


def route_and_explain(image_path: str) -> str:
    modality = detect_modality(image_path)
    model, platt = _get_model_for_modality(modality)
    device = torch.device('cpu')

    threshold = 0.36 if modality == 'mri' else 0.5
    finding_labels = {
        'mri': {
            1: 'Possible endometriosis indicators visible on this MRI slice',
            0: 'No endometriosis indicators visible on this MRI slice',
        },
        'laparoscopy': {
            1: 'Possible endometriosis tissue visible on this laparoscopy frame',
            0: 'No endometriosis tissue visible on this laparoscopy frame',
        }
    }

    tens = preprocess_for_model(image_path, modality)

    with torch.no_grad():
        raw_prob = torch.sigmoid(model(tens.unsqueeze(0).to(device))).item()

    confidence = float(platt.predict_proba([[raw_prob]])[:, 1][0]) if platt else raw_prob
    finding = int(confidence > threshold)

    target_layer = model.blocks[-1][-1] if modality == 'mri' else model.blocks[6]

    with torch.enable_grad():
        grad_cam_fn, hooks = make_grad_cam(model, target_layer)
        cam = grad_cam_fn(tens)
        hooks[0].remove()
        hooks[1].remove()

    with Image.open(image_path) as raw_img:
        raw = np.array(raw_img.convert('RGB'), dtype=np.float32)
    
    cam_resized = cv2.resize(cam, (raw.shape[1], raw.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    overlay = (0.55 * raw + 0.45 * cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)).astype(np.uint8)

    region_desc = describe_attention_region(cam)
    confidence_label = f"{confidence*100:.0f}%" if platt else "Moderate"
    finding_label = finding_labels[modality][finding]

    prompt = f"""
You are EndoScan AI, a compassionate and helpful health assistant explaining scan results to a patient.

Scan Details:
- Scan Type: {modality}
- Result: {finding_label}
- Confidence: {confidence_label}
- Highlighted Area: {region_desc}

INSTRUCTIONS:
1. Write in plain, everyday English that anyone can easily understand.
2. ABSOLUTELY NO MEDICAL JARGON. Use simple words like "pelvic area", "tissue", "dark spots", "inflammation", or "scarring".
3. Do not use Markdown header symbols (###) or dividers (---).
4. Keep line spacing tight and paragraph gaps minimal.

Follow this structure exactly:

AI Analysis Summary
• Finding: [Translate finding_label into a simple phrase]
• Confidence: {confidence_label}
• Focused Area: [Translate region_desc into a simple location]

What the AI Saw
[2-3 simple sentences explaining what the AI looked at in the highlighted area of the image and whether it saw signs of endometriosis.]

Important Things to Remember
1. One Picture Isn't the Whole Story: Explain simply that a single photo cannot check every hidden area in the pelvis.
2. Complete Doctor Review: Explain simply why surgeon evaluation during surgery and lab testing are needed for certainty.
3. Next Steps: Advise discussing these results with their doctor at their follow-up visit.
"""

    overlay_pil = Image.fromarray(overlay)

    try:
        if not gemini_client:
            raise ValueError("Gemini API key is missing.")

        response = gemini_client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[overlay_pil, prompt]
        )
        return response.text
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return (
            "AI Analysis Summary\n"
            f"• Finding: {finding_label}\n"
            f"• Confidence: {confidence_label}\n"
            f"• Focused Area: {region_desc}\n\n"
            "What the AI Saw\n"
            "The AI examined the highlighted section of your scan. In this image, it checked for unusual tissue patterns, dark spots, or scar tissue commonly linked with endometriosis.\n\n"
            "Important Things to Remember\n"
            "1. One Picture Isn't the Whole Story: A single clear image does not rule out endometriosis in other hidden areas of the pelvis.\n"
            "2. Complete Doctor Review: Your doctor will check the full area during surgery and send tissue samples to a lab for complete confirmation.\n"
            "3. Next Steps: Please discuss these images and your surgical report with your doctor during your follow-up visit."
        )