"""
Gradio App — DCGAN vs WGAN-GP: Tackling Mode Collapse in GANs
22F-3396 & 22F-3369 | GenAI Assignment 03
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils as vutils
import gradio as gr
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io

# ─── Hyperparameters (must match training) ───
NZ = 100
NGF = 64
NDF = 64
NC = 3

# ─── Model Architectures ───

class DCGANGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(NZ, NGF*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF*8, NGF*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF*4, NGF*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF*2, NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF, NC, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


class WGANGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(NZ, NGF*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(NGF*8),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF*8, NGF*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF*4, NGF*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF*2, NGF, 4, 2, 1, bias=False),
            nn.BatchNorm2d(NGF),
            nn.ReLU(True),
            nn.ConvTranspose2d(NGF, NC, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)


# ─── Load Models ───
device = torch.device('cpu')  # CPU for Gradio serving

dcgan_gen = DCGANGenerator().to(device)
wgan_gen = WGANGenerator().to(device)

DCGAN_PATH = 'dcgan_generator_final.pth'
WGAN_PATH = 'wgangp_generator_final.pth'
LOGS_PATH = 'training_logs.json'

models_loaded = {'dcgan': False, 'wgan': False}

if os.path.exists(DCGAN_PATH):
    dcgan_gen.load_state_dict(torch.load(DCGAN_PATH, map_location=device))
    dcgan_gen.eval()
    models_loaded['dcgan'] = True
    print("✅ DCGAN generator loaded")
else:
    print(f"⚠️ DCGAN weights not found at {DCGAN_PATH}")

if os.path.exists(WGAN_PATH):
    wgan_gen.load_state_dict(torch.load(WGAN_PATH, map_location=device))
    wgan_gen.eval()
    models_loaded['wgan'] = True
    print("✅ WGAN-GP generator loaded")
else:
    print(f"⚠️ WGAN-GP weights not found at {WGAN_PATH}")

# Load training logs
training_logs = None
if os.path.exists(LOGS_PATH):
    with open(LOGS_PATH, 'r') as f:
        training_logs = json.load(f)
    print("✅ Training logs loaded")


# ─── Helper Functions ───

def tensor_to_pil(tensor_img):
    """Convert a [-1,1] tensor to PIL Image."""
    img = tensor_img.cpu().detach()
    img = (img + 1) / 2.0  # to [0,1]
    img = img.clamp(0, 1)
    img = img.permute(1, 2, 0).numpy()
    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)


def generate_grid(generator, num_images, seed=None):
    """Generate a grid of images."""
    if seed is not None:
        torch.manual_seed(seed)
    with torch.no_grad():
        noise = torch.randn(num_images, NZ, 1, 1, device=device)
        fake = generator(noise)
    nrow = min(num_images, 5)
    grid = vutils.make_grid(fake, nrow=nrow, padding=2, normalize=True)
    return tensor_to_pil(grid)


# ─── Gradio Functions ───

def generate_dcgan(num_images, seed):
    if not models_loaded['dcgan']:
        return None, "❌ DCGAN model not loaded. Train the model first."
    seed = int(seed) if seed else None
    img = generate_grid(dcgan_gen, int(num_images), seed)
    return img, f"✅ Generated {int(num_images)} images with DCGAN (seed={seed})"


def generate_wgan(num_images, seed):
    if not models_loaded['wgan']:
        return None, "❌ WGAN-GP model not loaded. Train the model first."
    seed = int(seed) if seed else None
    img = generate_grid(wgan_gen, int(num_images), seed)
    return img, f"✅ Generated {int(num_images)} images with WGAN-GP (seed={seed})"


def generate_comparison(num_images, seed):
    seed_val = int(seed) if seed else 42
    results = []
    for name, gen, loaded in [("DCGAN", dcgan_gen, models_loaded['dcgan']),
                               ("WGAN-GP", wgan_gen, models_loaded['wgan'])]:
        if loaded:
            torch.manual_seed(seed_val)
            results.append(generate_grid(gen, int(num_images), seed_val))
        else:
            placeholder = Image.new('RGB', (320, 320), (50, 50, 50))
            results.append(placeholder)

    return results[0], results[1], f"✅ Side-by-side comparison (seed={seed_val})"


def show_training_plots():
    if training_logs is None:
        return None, "❌ Training logs not found."

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if 'dcgan' in training_logs:
        d = training_logs['dcgan']
        axes[0].plot(d['G_losses'], label='Generator', color='#2196F3', linewidth=2)
        axes[0].plot(d['D_losses'], label='Discriminator', color='#FF5722', linewidth=2)
        axes[0].set_title('DCGAN Losses', fontsize=13, fontweight='bold')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

    if 'wgangp' in training_logs:
        d = training_logs['wgangp']
        axes[1].plot(d['G_losses'], label='Generator', color='#4CAF50', linewidth=2)
        axes[1].plot(d['C_losses'], label='Critic', color='#9C27B0', linewidth=2)
        axes[1].set_title('WGAN-GP Losses', fontsize=13, fontweight='bold')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf)
    return img, "✅ Training loss plots rendered."


def compute_diversity(num_samples):
    results = []
    for name, gen, loaded in [("DCGAN", dcgan_gen, models_loaded['dcgan']),
                               ("WGAN-GP", wgan_gen, models_loaded['wgan'])]:
        if not loaded:
            results.append(f"  {name}: ❌ Not loaded")
            continue
        with torch.no_grad():
            noise = torch.randn(int(num_samples), NZ, 1, 1, device=device)
            imgs = gen(noise).view(int(num_samples), -1)
        n_pairs = min(500, int(num_samples) * (int(num_samples)-1) // 2)
        idx = torch.randint(0, int(num_samples), (n_pairs, 2))
        dists = torch.norm(imgs[idx[:,0]] - imgs[idx[:,1]], dim=1)
        results.append(f"  {name}: {dists.mean():.4f} ± {dists.std():.4f}")

    return "📊 Diversity (Avg Pairwise L2 Distance):\n" + "\n".join(results)


# ─── Gradio UI ───

THEME = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="cyan",
    font=gr.themes.GoogleFont("Inter"),
)

with gr.Blocks(theme=THEME, title="GAN Mode Collapse Demo") as demo:
    gr.Markdown("""
    # 🎨 Tackling Mode Collapse in GANs
    ### DCGAN vs WGAN-GP — Interactive Demo
    **22F-3396 & 22F-3369 | GenAI Assignment 03**

    ---
    """)

    with gr.Tab("🖼️ DCGAN Generator"):
        gr.Markdown("Generate images using the **baseline DCGAN** model.")
        with gr.Row():
            dc_num = gr.Slider(1, 20, value=10, step=1, label="Number of Images")
            dc_seed = gr.Number(value=42, label="Random Seed", precision=0)
        dc_btn = gr.Button("🎲 Generate with DCGAN", variant="primary")
        dc_out = gr.Image(label="DCGAN Output", type="pil")
        dc_status = gr.Textbox(label="Status", interactive=False)
        dc_btn.click(generate_dcgan, [dc_num, dc_seed], [dc_out, dc_status])

    with gr.Tab("⚡ WGAN-GP Generator"):
        gr.Markdown("Generate images using the **advanced WGAN-GP** model.")
        with gr.Row():
            wg_num = gr.Slider(1, 20, value=10, step=1, label="Number of Images")
            wg_seed = gr.Number(value=42, label="Random Seed", precision=0)
        wg_btn = gr.Button("🎲 Generate with WGAN-GP", variant="primary")
        wg_out = gr.Image(label="WGAN-GP Output", type="pil")
        wg_status = gr.Textbox(label="Status", interactive=False)
        wg_btn.click(generate_wgan, [wg_num, wg_seed], [wg_out, wg_status])

    with gr.Tab("🔀 Side-by-Side Comparison"):
        gr.Markdown("Compare **DCGAN vs WGAN-GP** using the same noise vector.")
        with gr.Row():
            cmp_num = gr.Slider(1, 16, value=8, step=1, label="Number of Images")
            cmp_seed = gr.Number(value=42, label="Random Seed", precision=0)
        cmp_btn = gr.Button("🔍 Compare Models", variant="primary")
        with gr.Row():
            cmp_dc = gr.Image(label="DCGAN", type="pil")
            cmp_wg = gr.Image(label="WGAN-GP", type="pil")
        cmp_status = gr.Textbox(label="Status", interactive=False)
        cmp_btn.click(generate_comparison, [cmp_num, cmp_seed], [cmp_dc, cmp_wg, cmp_status])

    with gr.Tab("📈 Training Logs"):
        gr.Markdown("View the **Generator & Discriminator/Critic loss** over training epochs.")
        log_btn = gr.Button("📊 Show Training Plots", variant="primary")
        log_img = gr.Image(label="Loss Curves", type="pil")
        log_status = gr.Textbox(label="Status", interactive=False)
        log_btn.click(show_training_plots, [], [log_img, log_status])

    with gr.Tab("📏 Diversity Analysis"):
        gr.Markdown("Measure **image diversity** via average pairwise L2 distance.")
        div_num = gr.Slider(10, 200, value=100, step=10, label="Samples to Generate")
        div_btn = gr.Button("🔬 Analyze Diversity", variant="primary")
        div_out = gr.Textbox(label="Results", lines=5, interactive=False)
        div_btn.click(compute_diversity, [div_num], [div_out])

    gr.Markdown("""
    ---
    **Architecture Notes:**
    - **DCGAN**: BCE Loss + Sigmoid → prone to mode collapse & vanishing gradients
    - **WGAN-GP**: Wasserstein Loss + Gradient Penalty (λ=10) → stable training & diverse outputs
    - **Critic updates**: 5 per generator update for better gradient signal
    """)


if __name__ == "__main__":
    demo.launch(share=False)
