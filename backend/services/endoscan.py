import os
import io
import base64
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import cv2
from django.conf import settings
import google.generativeai as genai
import os
from huggingface_hub import hf_hub_download, snapshot_download

# Configure Gemini API
genai.configure(api_key=getattr(settings, "GEMINI_API_KEY", os.environ.get("GEMINI_API_KEY")))

REPO_ID = "JoyKiplagat/endoscan-ner-model"
HF_TOKEN = os.getenv("HF_TOKEN")
# Weights directory setup
WEIGHTS_DIR = os.path.join(settings.BASE_DIR, 'backend', 'services', 'weights')

# Single .pth weight paths generated from fix_archive.py
CONSOLIDATED_MRI_FILE = os.path.join(WEIGHTS_DIR, 'mri_model.pth')
CONSOLIDATED_LAPAROSCOPY_FILE = os.path.join(WEIGHTS_DIR, 'laparoscopy_model.pth')


class EndoScanModel(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.target_layer.register_forward_hook(lambda m, i, o: setattr(self, 'activations', o))
        self.target_layer.register_full_backward_hook(lambda m, gi, go: setattr(self, 'gradients', go[0]))

    def generate_cam(self, input_tensor):
        self.model.eval()
        output = self.model(input_tensor)
        self.model.zero_grad()
        output.backward(torch.ones_like(output), retain_graph=True)

        gradients = self.gradients.data.cpu().numpy()[0]
        activations = self.activations.data.cpu().numpy()[0]
        weights = np.mean(gradients, axis=(1, 2))
        
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        if cam.max() > 0:
            cam /= cam.max()
        
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        
        with torch.no_grad():
            prob = torch.sigmoid(output).item()

        return cam, prob


def load_model_weights(modality, device):
    # Retrieve model weights dynamically from Hugging Face Hub
    filename = "laparoscopy_model.pth" if modality == "Laparoscopy" else "mri_model.pth"
    
    try:
        load_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            token=HF_TOKEN
        )
    except Exception as e:
        print(f" Error downloading {filename} from Hugging Face: {e}")
        load_path = None

    model = EndoScanModel().to(device)

    if load_path and os.path.isfile(load_path):
        try:
            try:
                checkpoint = torch.load(load_path, map_location=device, weights_only=True)
            except Exception:
                checkpoint = torch.load(load_path, map_location=device, weights_only=False)
            
            if isinstance(checkpoint, nn.Module):
                model = checkpoint.to(device)
            else:
                if isinstance(checkpoint, dict):
                    state_dict = checkpoint.get('state_dict', checkpoint.get('model', checkpoint))
                else:
                    state_dict = checkpoint

                cleaned_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
                model.load_state_dict(cleaned_state_dict, strict=False)

            print(f" Loaded weights for {modality} from Hugging Face cache ({load_path})")
        except Exception as e:
            print(f" Error loading weights from {load_path}: {e}")
    else:
        print(f" Weight file not found for {modality}")

    model.eval()
    return model


def detect_scan_modality(img_pil):
    img_np = np.array(img_pil)
    if len(img_np.shape) == 2 or img_np.shape[2] == 1:
        return "MRI"
    
    r, g, b = img_np[:, :, 0], img_np[:, :, 1], img_np[:, :, 2]
    rg_diff = np.mean(np.abs(r.astype(int) - g.astype(int)))
    rb_diff = np.mean(np.abs(r.astype(int) - b.astype(int)))

    if (rg_diff + rb_diff) > 12.0:
        return "Laparoscopy"
    return "MRI"


def get_heatmap_region(cam_mask):
    h, w = cam_mask.shape
    top_half = cam_mask[:h//2, :]
    bottom_half = cam_mask[h//2:, :]
    left_half = cam_mask[:, :w//2]
    right_half = cam_mask[:, w//2:]

    v_pos = "upper" if np.sum(top_half) > np.sum(bottom_half) else "lower"
    h_pos = "left" if np.sum(left_half) > np.sum(right_half) else "right"
    return f"{v_pos}-{h_pos}"


def generate_gemini_explanation(modality, probability, region_desc):
    pos_prob_pct = round(probability * 100, 1)
    neg_prob_pct = round((1 - probability) * 100, 1)
    is_detected = probability >= 0.5
    
    # Map spatial locations to clear everyday terms
    region_friendly = {
        "upper-left": "top-left",
        "upper-right": "top-right",
        "lower-left": "bottom-left",
        "lower-right": "bottom-right"
    }.get(region_desc, region_desc)

    # Friendly descriptions of scan types
    modality_description = (
        "keyhole camera image" if modality == "Laparoscopy" else "detailed magnetic imaging scan"
    )

    prompt = f"""
    You are a caring, friendly health assistant explaining an imaging check to a patient in everyday language. Write a complete, clear overview of their scan results using plain words and zero medical jargon.

    Scan Information:
    - Scan Category: {modality} ({modality_description})
    - Main Findings Chance: {pos_prob_pct}% likelihood of showing visual patterns related to endometriosis
    - Standard Tissue Chance: {neg_prob_pct}% likelihood of normal, healthy tissue
    - Highlighted Focus Area: {region_friendly} section of the scan
    - Overall Result: {"Potential tissue changes identified" if is_detected else "Mostly clear image with standard patterns"}

    Instructions for standard generation:
    1. Write 3-4 friendly, reassuring sentences explaining the whole scan.
    2. Explain what type of scan was analyzed ({modality}) and what the AI was checking for in plain terms.
    3. State the percentage breakdown ({pos_prob_pct}% chance of tissue changes vs {neg_prob_pct}% chance of clear tissue) naturally.
    4. Explain that the brightly colored overlay shows the exact spot ({region_friendly} area) where the system looked closest.
    5. Warmly remind the patient that this automated check is a screening tool to help start a conversation with their doctor.
    6. STRICT RULE: Do NOT use markdown headers, bullet points, or bold text (no asterisks). Use simple paragraph text only.
    """

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception:
        if is_detected:
            return (
                f"Your {modality} scan was reviewed by our automated screening system to look for subtle signs of tissue changes. "
                f"The analysis found a {pos_prob_pct}% chance of patterns that may suggest endometriosis tissue, along with a {neg_prob_pct}% likelihood of standard healthy tissue. "
                f"The bright highlighted colors on your image point out that the system focused its attention mostly on the {region_friendly} part of the scan. "
                f"Please keep in mind that this report is designed to assist your health check, so be sure to discuss these results with your doctor."
            )
        else:
            return (
                f"Our automated system analyzed your {modality} scan to check for any unusual visual patterns. "
                f"Your scan looks mostly clear, showing a {neg_prob_pct}% likelihood of standard healthy tissue and only a {pos_prob_pct}% chance of any tissue changes. "
                f"The colored highlights in the {region_friendly} section simply mark where the system performed its closest check for comparison. "
                f"This initial review is meant to support your care, so please feel free to go over this image with your healthcare provider."
            )


def run_endoscan_inference(image_bytes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Preprocess input image
    orig_img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = preprocess(orig_img).unsqueeze(0).to(device)

    # 2. Modality Detection & Model Loading
    detected_modality = detect_scan_modality(orig_img)
    model = load_model_weights(detected_modality, device)

    # 3. Grad-CAM Feature Map & Sigmoid Inference
    cam_generator = GradCAM(model, model.backbone.features[-1])
    cam_mask, prob = cam_generator.generate_cam(input_tensor)

    # 4. Generate Heatmap Base64 Overlay
    img_np = np.array(orig_img.resize((224, 224))) / 255.0
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) / 255.0
    overlay = np.clip(0.6 * img_np + 0.4 * heatmap, 0, 1)

    overlay_uint8 = (overlay * 255).astype(np.uint8)
    _, buffer = cv2.imencode('.jpg', cv2.cvtColor(overlay_uint8, cv2.COLOR_RGB2BGR))
    overlay_b64 = base64.b64encode(buffer).decode('utf-8')

    # 5. Region Localization & Dynamic Plain-Language Gemini Summary
    region_desc = get_heatmap_region(cam_mask)
    gemini_explainer = generate_gemini_explanation(detected_modality, prob, region_desc)

    pos_pct = round(prob * 100, 1)
    neg_pct = round((1 - prob) * 100, 1)

    return {
        "detected_modality": detected_modality,
        "probability": float(prob),
        "confidence_percentage": pos_pct,
        "neg_confidence_percentage": neg_pct,
        "detected": bool(prob >= 0.5),
        "summary": gemini_explainer,
        "heatmap_overlay": f"data:image/jpeg;base64,{overlay_b64}"
    }