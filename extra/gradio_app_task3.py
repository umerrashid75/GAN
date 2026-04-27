"""
Task 3: CycleGAN - Unpaired Image-to-Image Translation
Gradio app for Sketch ↔ Photo domain translation.

Usage:
  1. Place cyclegan_G_AB_final.pth and cyclegan_G_BA_final.pth in this directory
  2. Run: python gradio_app_task3.py
"""

import os
import torch
import torch.nn as nn
import gradio as gr
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

# ========================
# Configuration
# ========================
IMAGE_SIZE = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = os.path.dirname(os.path.abspath(__file__))

# ========================
# Model Architecture (must match training notebook exactly)
# ========================

class ResNetBlock(nn.Module):
    """Residual block with reflection padding and instance norm."""
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, 3),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)


class Generator(nn.Module):
    """
    ResNet-based Generator for CycleGAN.
    Architecture: c7s1-64, d128, d256, R256×6, u128, u64, c7s1-3
    """
    def __init__(self, in_ch=3, out_ch=3, ngf=64, n_blocks=6):
        super().__init__()
        # Encoder
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, ngf, 7), nn.InstanceNorm2d(ngf), nn.ReLU(True),
        ]
        in_f = ngf
        for _ in range(2):
            out_f = in_f * 2
            model += [nn.Conv2d(in_f, out_f, 3, stride=2, padding=1),
                      nn.InstanceNorm2d(out_f), nn.ReLU(True)]
            in_f = out_f
        # Transformer
        for _ in range(n_blocks):
            model += [ResNetBlock(in_f)]
        # Decoder
        for _ in range(2):
            out_f = in_f // 2
            model += [nn.ConvTranspose2d(in_f, out_f, 3, stride=2, padding=1, output_padding=1),
                      nn.InstanceNorm2d(out_f), nn.ReLU(True)]
            in_f = out_f
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, out_ch, 7), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


# ========================
# Load Models
# ========================
G_AB = Generator(in_ch=3, out_ch=3, ngf=64, n_blocks=6).to(DEVICE)  # Sketch → Photo
G_BA = Generator(in_ch=3, out_ch=3, ngf=64, n_blocks=6).to(DEVICE)  # Photo → Sketch


def load_models():
    """Load pre-trained generator weights."""
    g_ab_path = os.path.join(CHECKPOINT_DIR, 'cyclegan_G_AB_final.pth')
    g_ba_path = os.path.join(CHECKPOINT_DIR, 'cyclegan_G_BA_final.pth')

    if os.path.exists(g_ab_path):
        G_AB.load_state_dict(torch.load(g_ab_path, map_location=DEVICE, weights_only=True))
        print("✓ Sketch→Photo generator loaded")
    else:
        print(f"⚠ Not found: {g_ab_path}")

    if os.path.exists(g_ba_path):
        G_BA.load_state_dict(torch.load(g_ba_path, map_location=DEVICE, weights_only=True))
        print("✓ Photo→Sketch generator loaded")
    else:
        print(f"⚠ Not found: {g_ba_path}")

    G_AB.eval()
    G_BA.eval()


# ========================
# Image Processing
# ========================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])


def denorm(t):
    return ((t * 0.5 + 0.5).clamp(0, 1) * 255).to(torch.uint8)


def preprocess(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))
    return transform(image.convert('RGB')).unsqueeze(0).to(DEVICE)


def postprocess(tensor):
    return denorm(tensor.squeeze(0)).permute(1, 2, 0).cpu().numpy()


# ========================
# Translation Functions
# ========================
def sketch_to_photo(image):
    """Translate sketch → photo."""
    if image is None:
        return None
    with torch.no_grad():
        return postprocess(G_AB(preprocess(image)))


def photo_to_sketch(image):
    """Translate photo → sketch."""
    if image is None:
        return None
    with torch.no_grad():
        return postprocess(G_BA(preprocess(image)))


def cycle_demo(image, direction):
    """
    Demonstrate cycle consistency:
    Input → Translated → Reconstructed
    """
    if image is None:
        return None, None, None

    with torch.no_grad():
        inp = preprocess(image)
        if direction == "Sketch → Photo → Sketch":
            translated = G_AB(inp)
            reconstructed = G_BA(translated)
        else:
            translated = G_BA(inp)
            reconstructed = G_AB(translated)

    return (
        postprocess(inp),
        postprocess(translated),
        postprocess(reconstructed),
    )


# ========================
# Gradio Interface
# ========================
def build_app():
    with gr.Blocks(
        title="CycleGAN: Sketch ↔ Photo Translation",
        theme=gr.themes.Soft(),
    ) as app:
        gr.Markdown(
            """
            # 🎨 CycleGAN: Sketch ↔ Photo Translation
            **Model:** ResNet Generator (6 blocks) + PatchGAN Discriminator  
            **Trained on:** TU-Berlin + Sketchy + QuickDraw (sketches) and STL-10 (photos)
            """
        )

        with gr.Tabs():
            # --- Tab 1: Sketch → Photo ---
            with gr.Tab("Sketch → Photo"):
                with gr.Row():
                    s2p_in = gr.Image(type="numpy", label="Input Sketch")
                    s2p_out = gr.Image(type="numpy", label="Generated Photo")
                gr.Button("Translate").click(sketch_to_photo, s2p_in, s2p_out)

            # --- Tab 2: Photo → Sketch ---
            with gr.Tab("Photo → Sketch"):
                with gr.Row():
                    p2s_in = gr.Image(type="numpy", label="Input Photo")
                    p2s_out = gr.Image(type="numpy", label="Generated Sketch")
                gr.Button("Translate").click(photo_to_sketch, p2s_in, p2s_out)

            # --- Tab 3: Cycle Consistency ---
            with gr.Tab("Cycle Consistency"):
                gr.Markdown("Demonstrates structural preservation: Input → Translated → Reconstructed")
                cyc_dir = gr.Radio(
                    ["Sketch → Photo → Sketch", "Photo → Sketch → Photo"],
                    value="Sketch → Photo → Sketch",
                    label="Cycle Direction",
                )
                cyc_in = gr.Image(type="numpy", label="Input")
                with gr.Row():
                    cyc_orig = gr.Image(type="numpy", label="Original")
                    cyc_trans = gr.Image(type="numpy", label="Translated")
                    cyc_recon = gr.Image(type="numpy", label="Reconstructed")
                gr.Button("Run Cycle").click(
                    cycle_demo, [cyc_in, cyc_dir], [cyc_orig, cyc_trans, cyc_recon]
                )

        gr.Markdown(
            """
            ---
            **Architecture:** Generator uses 6 ResNet blocks with InstanceNorm and ReflectionPadding.  
            **Training:** LSGAN loss, cycle consistency (λ=10), identity loss (λ=5), linear LR decay.  
            **Evaluation:** SSIM and PSNR metrics on cycle reconstruction quality.
            """
        )

    return app


# ========================
# Main
# ========================
if __name__ == "__main__":
    print(f"Device: {DEVICE}")
    print(f"Loading models from: {CHECKPOINT_DIR}")
    load_models()

    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
