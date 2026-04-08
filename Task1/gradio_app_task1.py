"""
Task 1: CycleGAN - DCGAN vs WGAN-GP Comparison
Gradio app for comparing mode collapse mitigation techniques
"""
import os
import torch
import torch.nn as nn
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import make_grid

# Configuration
IMAGE_SIZE = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = os.path.dirname(__file__)

# Generator Architecture (shared)
class Generator(nn.Module):
    def __init__(self, nz=100, ngf=64):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)

# Initialize models
G_DCGAN = Generator(nz=100, ngf=64).to(DEVICE)
G_WGANGP = Generator(nz=100, ngf=64).to(DEVICE)

def load_models():
    """Load pre-trained models."""
    dcgan_path = os.path.join(CHECKPOINT_DIR, 'dcgan_G_final.pth')
    wgangp_path = os.path.join(CHECKPOINT_DIR, 'wgangp_G_final.pth')

    if os.path.exists(dcgan_path):
        G_DCGAN.load_state_dict(torch.load(dcgan_path, map_location=DEVICE))
        print("✓ DCGAN model loaded")
    else:
        print(f"⚠ DCGAN model not found at {dcgan_path}")

    if os.path.exists(wgangp_path):
        G_WGANGP.load_state_dict(torch.load(wgangp_path, map_location=DEVICE))
        print("✓ WGAN-GP model loaded")
    else:
        print(f"⚠ WGAN-GP model not found at {wgangp_path}")

def generate_images(num_images, model_type):
    """Generate images using selected model."""
    G_DCGAN.eval()
    G_WGANGP.eval()

    with torch.no_grad():
        z = torch.randn(num_images, 100, 1, 1, device=DEVICE)

        if model_type == "DCGAN":
            fake_images = G_DCGAN(z)
        else:  # WGAN-GP
            fake_images = G_WGANGP(z)

        grid = make_grid(fake_images, nrow=5, normalize=True, value_range=(-1, 1))
        img = grid.permute(1, 2, 0).cpu().numpy()
        img = (img + 1) / 2  # Denormalize
        img = np.clip(img, 0, 1)

    return (img * 255).astype(np.uint8)

def compare_models(num_samples):
    """Generate comparison between DCGAN and WGAN-GP."""
    dcgan_img = generate_images(num_samples, "DCGAN")
    wgangp_img = generate_images(num_samples, "WGAN-GP")

    # Create comparison figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    axes[0].imshow(dcgan_img)
    axes[0].set_title("DCGAN Generated Images (Prone to Mode Collapse)", fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(wgangp_img)
    axes[1].set_title("WGAN-GP Generated Images (Mitigates Mode Collapse)", fontsize=12, fontweight='bold')
    axes[1].axis('off')

    plt.tight_layout()
    return fig

def generate_dcgan(num_samples):
    """Generate DCGAN samples."""
    return generate_images(num_samples, "DCGAN")

def generate_wgangp(num_samples):
    """Generate WGAN-GP samples."""
    return generate_images(num_samples, "WGAN-GP")

def info_panel():
    """Display model information."""
    info = """
    ## Task 1: Tackling Mode Collapse in GANs

    ### DCGAN (Deep Convolutional GAN)
    - **Loss Function**: Binary Cross Entropy (BCE)
    - **Discriminator Output**: Sigmoid activation
    - **Updates**: 1 discriminator update per generator update
    - **Issue**: Prone to mode collapse (generator produces limited diversity)

    ### WGAN-GP (Wasserstein GAN with Gradient Penalty)
    - **Loss Function**: Wasserstein loss + Gradient Penalty (λ=10)
    - **Critic Output**: No activation (raw scores)
    - **Updates**: 5 critic updates per generator update
    - **Benefit**: Prevents mode collapse through better loss function and training stability

    ### Key Differences
    | Feature | DCGAN | WGAN-GP |
    |---------|-------|---------|
    | Loss | BCE | Wasserstein + GP |
    | Normalization | Batch Norm | Instance Norm |
    | Training Stability | Lower | Higher |
    | Diversity | Limited | Better |

    **Dataset**: Pokemon Sprites or Anime Faces (64×64)
    **Model Size**: 64×64 RGB images
    **Architecture**: 4-layer transposed convolutions
    """
    return info

# Load models on startup
load_models()

# Create Gradio interface
with gr.Blocks(title="Task 1: Mode Collapse Mitigation", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Task 1: Tackling Mode Collapse in Generative Adversarial Networks")

    with gr.Tabs():
        # Tab 1: Comparison
        with gr.Tab("📊 Model Comparison"):
            gr.Markdown("### Side-by-side Comparison of DCGAN vs WGAN-GP")
            num_samples = gr.Slider(minimum=5, maximum=25, value=10, step=5, label="Number of Samples")
            compare_btn = gr.Button("Generate Comparison", variant="primary")
            comparison_output = gr.Plot(label="Model Comparison")

            compare_btn.click(fn=compare_models, inputs=num_samples, outputs=comparison_output)

        # Tab 2: DCGAN Generator
        with gr.Tab("🎨 DCGAN Generator"):
            gr.Markdown("### DCGAN: Baseline Deep Convolutional GAN")
            gr.Markdown("**Note**: DCGAN is prone to mode collapse - you may notice limited diversity in generated samples.")

            num_dcgan = gr.Slider(minimum=5, maximum=25, value=10, step=5, label="Number of Samples")
            dcgan_btn = gr.Button("Generate DCGAN Images", variant="primary")
            dcgan_output = gr.Image(label="DCGAN Generated Images")

            dcgan_btn.click(fn=generate_dcgan, inputs=num_dcgan, outputs=dcgan_output)

        # Tab 3: WGAN-GP Generator
        with gr.Tab("✨ WGAN-GP Generator"):
            gr.Markdown("### WGAN-GP: Advanced Wasserstein GAN with Gradient Penalty")
            gr.Markdown("**Note**: WGAN-GP generates more diverse samples with better training stability.")

            num_wgangp = gr.Slider(minimum=5, maximum=25, value=10, step=5, label="Number of Samples")
            wgangp_btn = gr.Button("Generate WGAN-GP Images", variant="primary")
            wgangp_output = gr.Image(label="WGAN-GP Generated Images")

            wgangp_btn.click(fn=generate_wgangp, inputs=num_wgangp, outputs=wgangp_output)

        # Tab 4: Information
        with gr.Tab("ℹ️ Information"):
            gr.Markdown(info_panel())

if __name__ == "__main__":
    demo.launch(share=True)
