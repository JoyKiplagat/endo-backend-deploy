import os
import gc
from pathlib import Path

# Locate project root dynamically
BASE_DIR = Path(__file__).resolve().parent.parent

# Model output paths using absolute BASE_DIR
MRI_CHECKPOINT_PATH = BASE_DIR / "outputs" / "mri_best_model.pt"
MRI_CALIBRATION_PATH = BASE_DIR / "outputs" / "mri_calibration.pkl"
LAPARO_CHECKPOINT_PATH = BASE_DIR / "outputs" / "mri_best_model" / "data.pt"

IMG_SIZE = 224


def detect_modality(image_path, channel_diff_threshold: float = 5.0) -> str:
    from PIL import Image
    import numpy as np
    
    with Image.open(image_path) as raw_img:
        img = np.array(raw_img.convert('RGB'), dtype=np.float32)
    r, g, b = img[..., 0], img[..., 1], img[..., 2]
    mean_channel_diff = (np.abs(r - g).mean() + np.abs(g - b).mean() + np.abs(r - b).mean()) / 3
    return 'laparoscopy' if mean_channel_diff > channel_diff_threshold else 'mri'


def preprocess_for_model(image_path, modality: str):
    import torch
    import numpy as np
    from PIL import Image

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


def route_and_explain(image_path: str) -> str:
    # 1. LAZY IMPORTS (Prevents startup memory crash on Render)
    import torch
    import torch.nn as nn
    import timm
    import joblib
    import cv2
    import numpy as np
    from PIL import Image
    from google import genai
    from dotenv import load_dotenv

    # Limit PyTorch CPU threads to prevent memory spikes
    torch.set_num_threads(1)

    # Load environment variables
    load_dotenv(dotenv_path=BASE_DIR / ".env")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    modality = detect_modality(image_path)
    device = torch.device('cpu')

    # 2. Build EfficientNet Architecture dynamically
    model = timm.create_model('efficientnet_b0', pretrained=False)
    in_feat = model.classifier.in_features
    model.classifier = nn.Sequential(nn.Dropout(p=0.3), nn.Linear(in_feat, 1))
    model = model.to(device)

    platt = None

    # 3. Load Model Weights on Demand
    if modality == 'mri':
        if MRI_CHECKPOINT_PATH.exists():
            ckpt = torch.load(MRI_CHECKPOINT_PATH, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            del ckpt
        if MRI_CALIBRATION_PATH.exists():
            platt = joblib.load(MRI_CALIBRATION_PATH)['platt']
        threshold = 0.36
        target_layer = model.blocks[-1][-1]
    else:
        if LAPARO_CHECKPOINT_PATH.exists():
            ckpt = torch.load(LAPARO_CHECKPOINT_PATH, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state_dict'])
            del ckpt
        threshold = 0.5
        target_layer = model.blocks[6]

    model.eval()

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

    # 4. Run Model Forward Pass
    with torch.no_grad():
        raw_prob = torch.sigmoid(model(tens.unsqueeze(0).to(device))).item()

    confidence = float(platt.predict_proba([[raw_prob]])[:, 1][0]) if platt else raw_prob
    finding = int(confidence > threshold)

    # 5. Compute Grad-CAM Heatmap
    activations, gradients = {}, {}

    def fwd_hook(module, inp, out): activations['feat'] = out.detach()
    def bwd_hook(module, g_in, g_out): gradients['feat'] = g_out[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    with torch.enable_grad():
        img_tensor = tens.unsqueeze(0).to(device).requires_grad_(True)
        out = model(img_tensor)
        model.zero_grad()
        out.backward()

        grads = gradients['feat'].squeeze(0)
        acts = activations['feat'].squeeze(0)
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = (weights * acts).sum(dim=0).cpu().numpy()
        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam /= cam.max()

    h1.remove()
    h2.remove()
    
    cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)) / 255.0

    # 6. Build Heatmap Overlay
    with Image.open(image_path) as raw_img:
        raw = np.array(raw_img.convert('RGB'), dtype=np.float32)

    cam_full = cv2.resize(cam_resized, (raw.shape[1], raw.shape[0]))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_full), cv2.COLORMAP_JET)
    overlay = (0.55 * raw + 0.45 * cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)).astype(np.uint8)

    # Describe Attention Region
    h_cam, w_cam = cam_resized.shape
    ys, xs = np.where(cam_resized > 0.6)
    if len(ys) == 0:
        region_desc = "no single strongly localised region"
    else:
        cy, cx = ys.mean() / h_cam, xs.mean() / w_cam
        vert = 'upper' if cy < 0.4 else ('lower' if cy > 0.6 else 'central')
        horiz = 'left' if cx < 0.4 else ('right' if cx > 0.6 else 'central')
        region_desc = 'central' if (vert == 'central' and horiz == 'central') else f'{vert}-{horiz}'

    confidence_label = f"{confidence*100:.0f}%" if platt else "Moderate"
    finding_label = finding_labels[modality][finding]

    # Free memory immediately
    del model, tens, img_tensor, grads, acts
    gc.collect()

    # 7. Call Gemini for Clinical Summary
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
        if not GEMINI_API_KEY:
            raise ValueError("Gemini API key missing")

        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
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