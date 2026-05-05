import gradio as gr
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# --- Model Architecture ---
class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, ngf=64, n_blocks=6):
        super().__init__()
        assert n_blocks >= 0

        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_channels, ngf, kernel_size=7),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
        ]

        in_features = ngf
        out_features = ngf * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features * 2

        for _ in range(n_blocks):
            model += [ResNetBlock(in_features)]

        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, kernel_size=3,
                                   stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features // 2

        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, out_channels, kernel_size=7),
            nn.Tanh(),
        ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# --- App Logic ---
IMG_SIZE = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load weights
print("Loading model weights...")
checkpoint_path = 'cyclegan_final.pth'
if not os.path.exists(checkpoint_path):
    raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found!")

checkpoint = torch.load(checkpoint_path, map_location=device)

# Initialize Generators
G_AB = Generator(n_blocks=6).to(device) # Sketch -> Photo
G_BA = Generator(n_blocks=6).to(device) # Photo -> Sketch

# Load state dicts
G_AB.load_state_dict(checkpoint['G_AB'])
G_BA.load_state_dict(checkpoint['G_BA'])

G_AB.eval()
G_BA.eval()

# Transformations
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def denormalize(tensor):
    return (tensor.data + 1) / 2.0

def predict(input_img, direction):
    if input_img is None:
        return None
    
    # Preprocess
    img_tensor = transform(input_img).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        if direction == "Sketch → Photo":
            output_tensor = G_AB(img_tensor)
        else:
            output_tensor = G_BA(img_tensor)
    
    # Postprocess
    output_img = denormalize(output_tensor.squeeze(0).cpu())
    output_img = transforms.ToPILImage()(output_img)
    
    return output_img

# --- Gradio Interface ---
css = """
footer {visibility: hidden}
.gradio-container {
    background: #0b0b0e !important;
}
#container {
    max-width: 1100px;
    margin: auto;
    padding: 30px;
    border-radius: 24px;
    background: #121217;
    border: 1px solid #23232b;
    box-shadow: 0 20px 50px rgba(0,0,0,0.6);
}
#title-header {
    text-align: center;
    padding-bottom: 20px;
}
#title-header h1 {
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff6b6b, #ff8e8e, #ff6b6b);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
#title-header p {
    color: #a1a1aa;
    font-size: 1.1rem;
}
.gr-button-primary {
    background: linear-gradient(135deg, #f43f5e, #e11d48) !important;
    border: none !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
}
.gr-button-primary:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(225, 29, 72, 0.3) !important;
}
.gr-input, .gr-output {
    border-radius: 16px !important;
    background: #1c1c24 !important;
    border: 1px solid #2d2d39 !important;
}
"""

with gr.Blocks() as demo:
    with gr.Column(elem_id="container"):
        with gr.Column(elem_id="title-header"):
            gr.Markdown("# 🎨 Domain Fusion: CycleGAN")
            gr.Markdown("Transform sketches into reality and photos into minimalist art.")
        
        with gr.Row():
            with gr.Column(scale=1):
                input_img = gr.Image(type="pil", label="Source Image")
                direction = gr.Dropdown(
                    choices=["Sketch → Photo", "Photo → Sketch"],
                    value="Sketch → Photo",
                    label="Translation Mode"
                )
                submit_btn = gr.Button("Generate Magic ✨", variant="primary")
            
            with gr.Column(scale=1):
                output_img = gr.Image(type="pil", label="Result")
        
        gr.Markdown("### 💡 Try these examples:")
        gr.Examples(
            examples=[
                ["examples/sketch_cat.png", "Sketch → Photo"],
                ["examples/photo_cat.png", "Photo → Sketch"],
                ["examples/sketch_flower.png", "Sketch → Photo"],
                ["examples/photo_flower.png", "Photo → Sketch"],
            ],
            inputs=[input_img, direction],
            outputs=output_img,
            fn=predict,
            cache_examples=False
        )

    submit_btn.click(fn=predict, inputs=[input_img, direction], outputs=output_img)

if __name__ == "__main__":
    print("Starting Gradio application...")
    demo.launch(
        server_name="0.0.0.0", 
        server_port=7860,
        css=css,
        theme=gr.themes.Default(primary_hue="rose", font=["Inter", "sans-serif"])
    )
