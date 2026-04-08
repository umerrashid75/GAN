"""
Task 3: CycleGAN - Unpaired Image-to-Image Translation
Gradio app for sketch-photo domain adaptation
"""
import os
import torch
import torch.nn as nn
import gradio as gr
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import make_grid

# Configuration
IMAGE_SIZE = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = os.path.dirname(__file__)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(out_channels)
        )

    def forward(self, x):
        return x + self.conv(x)

# ResNet Generator
class ResNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, ngf=64, num_blocks=6):
        super(ResNetGenerator, self).__init__()

        self.initial = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(ngf, ngf*2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf*2, ngf*4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ngf*4),
            nn.ReLU(inplace=True)
        )

        residual_blocks = [ResidualBlock(ngf*4, ngf*4) for _ in range(num_blocks)]
        self.residual = nn.Sequential(*residual_blocks)

        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(ngf*4, ngf*2, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf*2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf*2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True)
        )

        self.final = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7, padding=0),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.downsample(x)
        x = self.residual(x)
        x = self.upsample(x)
        x = self.final(x)
        return x

# Initialize generators
G_AB = ResNetGenerator(in_channels=3, out_channels=3, ngf=64, num_blocks=6).to(DEVICE)  # Sketch → Photo
G_BA = ResNetGenerator(in_channels=3, out_channels=3, ngf=64, num_blocks=6).to(DEVICE)  # Photo → Sketch

def load_models():
    """Load pre-trained models."""
    g_ab_path = os.path.join(CHECKPOINT_DIR, 'cyclegan_G_AB_final.pth')
    g_ba_path = os.path.join(CHECKPOINT_DIR, 'cyclegan_G_BA_final.pth')

    if os.path.exists(g_ab_path):
        G_AB.load_state_dict(torch.load(g_ab_path, map_location=DEVICE))
        print("✓ Sketch→Photo generator loaded")
    else:
        print(f"⚠ Sketch→Photo generator not found at {g_ab_path}")

    if os.path.exists(g_ba_path):
        G_BA.load_state_dict(torch.load(g_ba_path, map_location=DEVICE))
        print("✓ Photo→Sketch generator loaded")
    else:
        print(f"⚠ Photo→Sketch generator not found at {g_ba_path}")

def preprocess_image(image):
    """Preprocess image for model."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype('uint8'))

    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    return transform(image).unsqueeze(0).to(DEVICE)

def postprocess_image(tensor):
    """Convert tensor to numpy array."""
    with torch.no_grad():
        tensor = tensor.squeeze(0).permute(1, 2, 0).cpu()
        tensor = (tensor + 1) / 2  # Denormalize
        tensor = torch.clamp(tensor, 0, 1)
        img = (tensor.numpy() * 255).astype(np.uint8)
    return img

def sketch_to_photo(sketch_image):
    """Translate sketch to realistic photo."""
    if sketch_image is None:
        return None

    G_AB.eval()
    with torch.no_grad():
        input_tensor = preprocess_image(sketch_image)
        fake_photo = G_AB(input_tensor)
        output_img = postprocess_image(fake_photo)

    return output_img

def photo_to_sketch(photo_image):
    """Translate photo to sketch."""
    if photo_image is None:
        return None

    G_BA.eval()
    with torch.no_grad():
        input_tensor = preprocess_image(photo_image)
        fake_sketch = G_BA(input_tensor)
        output_img = postprocess_image(fake_sketch)

    return output_img

def cycle_consistency(input_image, forward_model="sketch_to_photo"):
    """Demonstrate cycle consistency."""
    if input_image is None:
        return None, None, None

    G_AB.eval()
    G_BA.eval()

    with torch.no_grad():
        input_tensor = preprocess_image(input_image)

        if forward_model == "sketch_to_photo":
            # Sketch → Photo → Sketch
            fake_photo = G_AB(input_tensor)
            reconstructed = G_BA(fake_photo)
            intermediate = postprocess_image(fake_photo)
        else:
            # Photo → Sketch → Photo
            fake_sketch = G_BA(input_tensor)
            reconstructed = G_AB(fake_sketch)
            intermediate = postprocess_image(fake_sketch)

        reconstructed_img = postprocess_image(reconstructed)

    return intermediate, reconstructed_img

def generate_random_samples(num_samples):
    """Generate samples from random noise."""
    G_AB.eval()
    G_BA.eval()

    with torch.no_grad():
        z = torch.randn(num_samples, 3, IMAGE_SIZE, IMAGE_SIZE).to(DEVICE)

        # Generate photos from noise
        photos = G_AB(z)
        photo_grid = make_grid(photos, nrow=5, normalize=True, value_range=(-1, 1))
        photo_img = photo_grid.permute(1, 2, 0).cpu().numpy()
        photo_img = (photo_img + 1) / 2

        # Generate sketches from noise
        sketches = G_BA(z)
        sketch_grid = make_grid(sketches, nrow=5, normalize=True, value_range=(-1, 1))
        sketch_img = sketch_grid.permute(1, 2, 0).cpu().numpy()
        sketch_img = (sketch_img + 1) / 2

    return (photo_img * 255).astype(np.uint8), (sketch_img * 255).astype(np.uint8)

# Load models
load_models()

# Create Gradio interface
with gr.Blocks(title="Task 3: CycleGAN Domain Adaptation", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Task 3: Domain Adaptation and Unpaired Image-to-Image Translation using CycleGAN")
    gr.Markdown("Unpaired image-to-image translation for Sketch ↔ Photo conversion")

    with gr.Tabs():
        # Tab 1: Sketch to Photo
        with gr.Tab("🎨 Sketch → Photo"):
            gr.Markdown("### Convert sketches to realistic photos (Unpaired)")

            with gr.Row():
                with gr.Column():
                    sketch_input = gr.Image(
                        label="Input Sketch Image",
                        type="numpy",
                        sources=["upload", "webcam", "clipboard"]
                    )
                    sketch_btn = gr.Button("Generate Photo", variant="primary", size="lg")

                with gr.Column():
                    sketch_output = gr.Image(label="Generated Realistic Photo")

            sketch_btn.click(fn=sketch_to_photo, inputs=sketch_input, outputs=sketch_output)

        # Tab 2: Photo to Sketch
        with gr.Tab("🎨 Photo → Sketch"):
            gr.Markdown("### Convert photos to sketches (Unpaired)")

            with gr.Row():
                with gr.Column():
                    photo_input = gr.Image(
                        label="Input Photo Image",
                        type="numpy",
                        sources=["upload", "webcam", "clipboard"]
                    )
                    photo_btn = gr.Button("Generate Sketch", variant="primary", size="lg")

                with gr.Column():
                    photo_output = gr.Image(label="Generated Sketch")

            photo_btn.click(fn=photo_to_sketch, inputs=photo_input, outputs=photo_output)

        # Tab 3: Cycle Consistency
        with gr.Tab("🔄 Cycle Consistency"):
            gr.Markdown("### Demonstrate cycle reconstruction")
            gr.Markdown("**Cycle Consistency**: Input → Intermediate → Reconstructed ≈ Input")

            direction = gr.Radio(
                choices=["sketch_to_photo", "photo_to_sketch"],
                value="sketch_to_photo",
                label="Translation Direction"
            )

            with gr.Row():
                with gr.Column():
                    cycle_input = gr.Image(
                        label="Input Image",
                        type="numpy",
                        sources=["upload", "webcam", "clipboard"]
                    )
                    cycle_btn = gr.Button("Generate Cycle", variant="primary", size="lg")

                with gr.Column():
                    cycle_intermediate = gr.Image(label="Intermediate (Translated)")

                with gr.Column():
                    cycle_reconstructed = gr.Image(label="Reconstructed (Back to Original)")

            cycle_btn.click(
                fn=cycle_consistency,
                inputs=[cycle_input, direction],
                outputs=[cycle_intermediate, cycle_reconstructed]
            )

        # Tab 4: Random Generation
        with gr.Tab("🎲 Random Generation"):
            gr.Markdown("### Generate images from random noise")

            num_samples = gr.Slider(minimum=5, maximum=25, value=10, step=5, label="Number of Samples")
            gen_btn = gr.Button("Generate Samples", variant="primary", size="lg")

            with gr.Row():
                gen_photo = gr.Image(label="Generated Photos")
                gen_sketch = gr.Image(label="Generated Sketches")

            gen_btn.click(
                fn=generate_random_samples,
                inputs=num_samples,
                outputs=[gen_photo, gen_sketch]
            )

        # Tab 5: Architecture
        with gr.Tab("ℹ️ Architecture"):
            gr.Markdown("""
            ## CycleGAN Architecture

            ### Key Innovation: Unpaired Training
            - **No paired data needed**: A and B domains are independent
            - **Cycle Consistency Loss**: Ensures reversibility
            - **Identity Loss**: Improves training stability

            ### Generators (ResNet-based)
            - **Architecture**: ResNet with 6 residual blocks
            - **Components**:
                1. Initial convolution (7×7, 64 filters)
                2. 2 downsampling layers (stride=2)
                3. 6 residual blocks
                4. 2 upsampling layers (transpose convolution)
                5. Final convolution + Tanh

            - **Normalization**: Instance Normalization (not Batch Norm)
            - **Why Instance Norm**: Normalizes per-sample, enables stable gradient penalty

            ### Discriminators (PatchGAN)
            - **Purpose**: Classify image patches as real or fake
            - **Input**: 128×128 images
            - **Output**: Matrix of patch scores
            - **Benefit**: Focuses on local realism

            ### Loss Functions

            #### 1. Adversarial Loss (GAN Loss)
            ```
            L_GAN = E[log D_A(x)] + E[log(1 - D_A(G_BA(y)))]
            ```
            - Ensures realism

            #### 2. Cycle Consistency Loss (λ=10)
            ```
            L_cycle = ||x - G_BA(G_AB(x))||₁ + ||y - G_AB(G_BA(y))||₁
            ```
            - Ensures reversibility: A → B → A should return to A

            #### 3. Identity Loss (λ=5)
            ```
            L_identity = ||x - G_BA(x)||₁ + ||y - G_AB(y)||₁
            ```
            - Forces generators to preserve color when translating to same domain

            #### Total Loss
            ```
            L_total = L_GAN + λ_cycle * L_cycle + λ_identity * L_identity
            ```

            ### Training Strategy
            - **Optimizer**: Adam (lr=0.0002, β₁=0.5, β₂=0.999)
            - **Batch Size**: 4 (optimized for Kaggle T4×2)
            - **Image Size**: 128×128
            - **Epochs**: 50+
            - **Key**: Update both generators and discriminators alternately

            ### Key Advantages
            ✓ **No paired data needed** - faster data collection
            ✓ **Cycle consistency** - ensures reversibility
            ✓ **Identity loss** - improves training stability
            ✓ **Unpaired learning** - more general approach
            """)

        # Tab 6: Datasets
        with gr.Tab("📊 Datasets"):
            gr.Markdown("""
            ## Supported Datasets

            ### TU-Berlin Sketch Database
            - **Source**: https://huggingface.co/datasets/sdiaeyu6n/tu-berlin
            - **Purpose**: Sketch to photo translation
            - **Domain A**: Hand-drawn sketches
            - **Domain B**: Real photographs
            - **Classes**: Multiple object categories

            ### Sketchy Dataset
            - **Source**: https://www.kaggle.com/datasets/sharanyasundar/sketchy-dataset
            - **Purpose**: Sketch-photo matching
            - **Read**: Explore README.md for structure
            - **Domain A**: Sketches
            - **Domain B**: Photos

            ### Google QuickDraw
            - **Source**: https://www.kaggle.com/c/quickdraw-doodle-recognition/data
            - **Purpose**: Doodle to object translation
            - **Domain A**: User doodles
            - **Domain B**: Real objects

            ## How to Use Datasets

            1. **Download** from Kaggle or HuggingFace
            2. **Extract** to Kaggle notebook environment
            3. **Update** `DATA_ROOT` path in Task3 notebook
            4. **Train** using CycleGAN notebook
            5. **Export** trained models:
               - `cyclegan_G_AB_final.pth` (Sketch→Photo)
               - `cyclegan_G_BA_final.pth` (Photo→Sketch)
            6. **Copy** .pth files to Task3 folder
            7. **Run** this Gradio app

            ## Important Notes

            ✓ **Unpaired**: Domains don't need corresponding images
            ✓ **Scalable**: Works with large datasets
            ✓ **Flexible**: Can adapt to any two image domains
            """)

if __name__ == "__main__":
    demo.launch(share=True)
