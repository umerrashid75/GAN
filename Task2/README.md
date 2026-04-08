# Task 2: Doodle-to-Real Image Translation and Colorization using Pix2Pix

## Overview
This folder contains:
- **GANt2.ipynb** - Complete PyTorch implementation of Pix2Pix (conditional GAN)
- **gradio_app_task2.py** - Interactive Gradio application for image-to-image translation

## Project Structure
```
Task2/
├── GANt2.ipynb                  # Training notebook
├── gradio_app_task2.py          # Gradio web interface
├── pix2pix_G_final.pth          # Trained generator (place here after training)
└── README.md
```

## Getting Started

### Step 1: Train Model (Kaggle)
1. Upload `GANt2.ipynb` to Kaggle
2. Add one of these datasets:
   - CUHK Face Sketch Database
   - Anime Sketch Colorization Dataset
3. Run all cells sequentially
4. Generator will be saved to `/kaggle/working/` as:
   - `pix2pix_G_final.pth`

### Step 2: Prepare Files
1. Download the trained `.pth` file from Kaggle
2. Copy to this folder (Task2/):
   - `pix2pix_G_final.pth`

### Step 3: Run Gradio App
```bash
cd Task2
pip install gradio torch torchvision pillow matplotlib
python gradio_app_task2.py
```

The app will launch at `http://localhost:7860`

## Model Architecture

### Generator: U-Net
- **Encoder**: 4 downsampling layers (conv stride=2)
- **Bottleneck**: Feature extraction at lowest resolution
- **Decoder**: 4 upsampling layers (transpose conv)
- **Skip Connections**: Between encoder and decoder layers
- **Input**: 256×256 RGB image
- **Output**: 256×256 RGB image with Tanh activation

### Discriminator: PatchGAN
- **Patch Size**: 16×16
- **Objective**: Classify if patches are real or fake
- **Layers**: 4 conv layers with decreasing resolution
- **Output**: Matrix of patch-wise probabilities

## Loss Functions

### Adversarial Loss (cGAN Loss)
Ensures generated images are realistic
```
L_cGAN = E[log D(x,y)] + E[log(1 - D(x,G(x)))]
```

### L1 Reconstruction Loss
Preserves structural information and details
```
L_L1 = ||y - G(x)||_1
```

### Total Loss
Combines both losses with weighting
```
L_total = L_cGAN + λ * L_L1  (λ = 100)
```

## Training Configuration
- **Optimizer**: Adam
- **Learning Rate**: 0.0002
- **Betas**: (0.5, 0.999)
- **Batch Size**: 16-32
- **Image Size**: 256×256
- **Epochs**: 50+
- **Mixed Precision**: Yes (torch.cuda.amp)

## Supported Use Cases

### 1. Sketch → Photo Translation
- Convert hand-drawn sketches to realistic images
- Preserves structural information
- Adds realistic textures and colors

### 2. Grayscale → Colorization
- Automatically colorize black & white images
- Learns realistic color distributions
- Works with various image types

### 3. Edge Maps → Photos
- Convert edge maps to realistic images
- Useful for image inpainting
- Preserves object structure

## Features in Gradio App

### Sketch to Photo Tab
- Upload sketch or edge map
- Generate realistic photo
- Instant translation

### Grayscale to Color Tab
- Upload B&W image
- Colorize automatically
- Preview results

### Architecture Info Tab
- Detailed model architecture
- Loss function equations
- Training strategy

### Datasets Tab
- Information about supported datasets
- How to prepare data
- Model export instructions

## Datasets

### CUHK Face Sketch Database (CUFS)
- **Domain**: Human faces
- **Pairs**: Sketch ↔ Photo (paired)
- **Resolution**: 256×256
- **Samples**: ~1000 pairs

### Anime Sketch Colorization Dataset
- **Domain**: Anime/manga characters
- **Pairs**: Sketch ↔ Colored (paired)
- **Resolution**: 256×256
- **Samples**: Variable

## Requirements

### For Training (Kaggle)
```
torch>=1.9.0
torchvision>=0.10.0
numpy
pillow
matplotlib
tqdm
opencv-python
```

### For Running Gradio App
```
gradio>=3.0
torch
torchvision
pillow
numpy
matplotlib
```

## Usage Examples

### Basic Usage
```python
# In Gradio interface:
1. Select "Sketch → Photo" tab
2. Upload sketch image
3. Click "Generate Photo"
4. Download result
```

### Command Line
```bash
python gradio_app_task2.py --share
```

## Model Weights

### File: `pix2pix_G_final.pth`
- **Size**: ~50-100 MB
- **Format**: PyTorch state dict
- **Architecture**: U-Net with skip connections
- **Input**: 256×256 RGB
- **Output**: 256×256 RGB

## Performance Tips

### For Better Results
1. **Input Quality**: Provide clear, well-defined sketches
2. **Preprocessing**: Ensure consistent image sizes
3. **Domain Match**: Use sketches similar to training data
4. **Batch Processing**: Works best with single images

### Memory Optimization
- Image size can be reduced to 128×128 if needed
- Batch size 1 recommended for inference
- GPU not required for inference (but faster)

## Troubleshooting

### Model not loading
- Verify file name: `pix2pix_G_final.pth`
- Check file is in Task2 folder
- Ensure PyTorch version compatibility

### Poor quality output
- Check input image quality
- Ensure input matches training domain
- Try different input styles

### CUDA out of memory
- Reduce image size if needed
- Use CPU for inference
- Reduce batch size

## Advanced Usage

### Change Input Size
Edit in `gradio_app_task2.py`:
```python
IMAGE_SIZE = 128  # Default: 256
```

### Load Custom Weights
```python
custom_path = "path/to/custom_weights.pth"
generator.load_state_dict(torch.load(custom_path))
```

## References
- **Paper**: Image-to-Image Translation with Conditional Adversarial Networks
  - arXiv:1611.05957 (Pix2Pix)
- **Architecture**: U-Net (arXiv:1505.04597)
- **PatchGAN**: Local discriminator approach

## Notes
- ✓ Uses supervised learning (paired data required)
- ✓ Best for deterministic translations
- ✓ Preserves structure via L1 loss
- ✓ Skip connections preserve fine details
- ⚠ Requires paired training data
- ⚠ Mode collapse less likely than traditional GANs

## License & Attribution
For academic use, please cite the Pix2Pix paper and this implementation.

## Contact & Support
Refer to main project documentation for support and issues.
