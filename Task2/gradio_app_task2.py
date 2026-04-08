"""
Task 2: Pix2Pix - Doodle-to-Real and Colorization
Gradio app for paired image-to-image translation
"""
import os
import torch
import torch.nn as nn
import gradio as gr
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import io

# Configuration
IMAGE_SIZE = 256
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT_DIR = os.path.dirname(__file__)

# U-Net Generator
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, ngf=64):
        super(UNetGenerator, self).__init__()

        # Encoder
        self.enc1 = self._conv_block(in_channels, ngf)
        self.enc2 = self._conv_block(ngf, ngf * 2)
        self.enc3 = self._conv_block(ngf * 2, ngf * 4)
        self.enc4 = self._conv_block(ngf * 4, ngf * 8)

        # Bottleneck
        self.bottleneck = self._conv_block(ngf * 8, ngf * 8)

        # Decoder
        self.dec4 = self._deconv_block(ngf * 16, ngf * 4)
        self.dec3 = self._deconv_block(ngf * 8, ngf * 2)
        self.dec2 = self._deconv_block(ngf * 4, ngf)
        self.dec1 = self._deconv_block(ngf * 2, out_channels)

        self.final = nn.Tanh()

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def _deconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 4, stride=2, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)

        b = self.bottleneck(e4)

        d4 = self.dec4(torch.cat([b, e4], 1))
        d3 = self.dec3(torch.cat([d4, e3], 1))
        d2 = self.dec2(torch.cat([d3, e2], 1))
        d1 = self.dec1(torch.cat([d2, e1], 1))

        return self.final(d1)

# Initialize model
generator = UNetGenerator(in_channels=3, out_channels=3, ngf=64).to(DEVICE)

def load_model():
    """Load pre-trained model."""
    model_path = os.path.join(CHECKPOINT_DIR, 'pix2pix_G_final.pth')

    if os.path.exists(model_path):
        generator.load_state_dict(torch.load(model_path, map_location=DEVICE))
        print("✓ Pix2Pix model loaded")
    else:
        print(f"⚠ Model not found at {model_path}")

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
    """Convert tensor to PIL Image."""
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

    generator.eval()
    with torch.no_grad():
        input_tensor = preprocess_image(sketch_image)
        output_tensor = generator(input_tensor)
        output_img = postprocess_image(output_tensor)

    return output_img

def grayscale_to_color(gray_image):
    """Colorize grayscale image."""
    if gray_image is None:
        return None

    # Convert to RGB if needed
    if len(gray_image.shape) == 2:
        gray_image = np.stack([gray_image] * 3, axis=-1)

    generator.eval()
    with torch.no_grad():
        input_tensor = preprocess_image(gray_image)
        output_tensor = generator(input_tensor)
        output_img = postprocess_image(output_tensor)

    return output_img

def create_comparison(input_img, output_img):
    """Create side-by-side comparison."""
    if input_img is None or output_img is None:
        return None

    # Resize to match
    h, w = input_img.shape[:2]
    output_resized = Image.fromarray(output_img).resize((w, h))

    # Create side-by-side
    comparison = Image.new('RGB', (w * 2, h))
    comparison.paste(Image.fromarray(input_img), (0, 0))
    comparison.paste(output_resized, (w, 0))

    return comparison

# Load model
load_model()

# Create Gradio interface
with gr.Blocks(title="Task 2: Pix2Pix Image-to-Image Translation", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Task 2: Doodle-to-Real Image Translation and Colorization using Pix2Pix")
    gr.Markdown("Paired image-to-image translation using conditional GANs (cGAN)")

    with gr.Tabs():
        # Tab 1: Sketch to Photo
        with gr.Tab("🎨 Sketch → Photo"):
            gr.Markdown("### Convert sketches and edge maps to realistic images")

            with gr.Row():
                with gr.Column():
                    sketch_input = gr.Image(
                        label="Input Sketch/Edge Image",
                        type="numpy",
                        sources=["upload", "webcam", "clipboard"]
                    )
                    sketch_btn = gr.Button("Generate Photo", variant="primary", size="lg")

                with gr.Column():
                    sketch_output = gr.Image(label="Generated Realistic Image")

            sketch_btn.click(fn=sketch_to_photo, inputs=sketch_input, outputs=sketch_output)

            with gr.Accordion("💡 Example Usage"):
                gr.Markdown("""
                1. Upload a hand-drawn sketch or edge map
                2. Click "Generate Photo" to translate it to a realistic image
                3. The model preserves structural information while adding realistic details

                **Best Results**: Clear sketches with distinct outlines
                """)

        # Tab 2: Grayscale to Color
        with gr.Tab("🎨 Grayscale → Color"):
            gr.Markdown("### Colorize grayscale images")

            with gr.Row():
                with gr.Column():
                    gray_input = gr.Image(
                        label="Input Grayscale Image",
                        type="numpy",
                        sources=["upload", "webcam", "clipboard"]
                    )
                    gray_btn = gr.Button("Colorize", variant="primary", size="lg")

                with gr.Column():
                    gray_output = gr.Image(label="Colorized Image")

            gray_btn.click(fn=grayscale_to_color, inputs=gray_input, outputs=gray_output)

            with gr.Accordion("💡 Example Usage"):
                gr.Markdown("""
                1. Upload a grayscale or black & white image
                2. Click "Colorize" to add realistic colors
                3. The model learns color distributions from training data

                **Best Results**: Clear, well-defined objects
                """)

        # Tab 3: Architecture Info
        with gr.Tab("ℹ️ Architecture"):
            gr.Markdown("""
            ## Pix2Pix Architecture

            ### Generator: U-Net
            - **Encoder**: 4 downsampling layers (stride=2)
            - **Bottleneck**: Feature extraction
            - **Decoder**: 4 upsampling layers (transpose convolution)
            - **Skip Connections**: Between encoder and decoder (U-Net style)
            - **Output**: 256×256 RGB image with Tanh activation [-1, 1]

            ### Discriminator: PatchGAN
            - **Patch Size**: 16×16
            - **Objective**: Classify if patches are real or fake
            - **Benefit**: Focuses on local image realism

            ### Loss Functions
            1. **Adversarial Loss (cGAN Loss)**
                - Ensures realistic generation
                - Formula: `L_cGAN = E[log D(x,y)] + E[log(1 - D(x,G(x)))]`

            2. **L1 Reconstruction Loss**
                - Preserves structural information
                - Formula: `L_L1 = ||y - G(x)||_1`

            3. **Total Loss**
                - Combined: `L = L_cGAN + λ * L_L1` (λ typically 100)

            ### Training Strategy
            - **Optimizer**: Adam (lr=0.0002, β₁=0.5, β₂=0.999)
            - **Batch Size**: 16-32
            - **Image Size**: 256×256
            - **Epochs**: 50+

            ### Key Benefits
            - ✓ Preserves structural information via skip connections
            - ✓ L1 loss ensures faithful reconstruction
            - ✓ PatchGAN improves texture quality
            - ✓ Conditional on input (supervised learning)
            """)

        # Tab 4: Datasets
        with gr.Tab("📊 Datasets"):
            gr.Markdown("""
            ## Supported Datasets

            ### CUHK Face Sketch Database (CUFS)
            - **Purpose**: Sketch to face photo translation
            - **Samples**: Paired sketch-photo pairs
            - **Resolution**: 256×256
            - **Domain**: Human faces

            ### Anime Sketch Colorization Dataset
            - **Purpose**: Anime sketch to colored image
            - **Samples**: Paired sketch-color pairs
            - **Resolution**: 256×256
            - **Domain**: Anime/manga characters

            ### How to Use
            1. Download dataset from Kaggle
            2. Place in project folder
            3. Update `DATA_ROOT` in notebook
            4. Train model with Jupyter notebook
            5. Copy trained `.pth` file to Task2 folder
            """)

if __name__ == "__main__":
    demo.launch(share=True)
