# Task 3: Domain Adaptation and Unpaired Image-to-Image Translation using CycleGAN

## Overview
This folder contains:
- **GANt3.ipynb** - Complete PyTorch implementation of CycleGAN
- **gradio_app_task3.py** - Interactive Gradio application for unpaired domain translation

## Project Structure
```
Task3/
├── GANt3.ipynb                      # Training notebook
├── gradio_app_task3.py              # Gradio web interface
├── cyclegan_G_AB_final.pth          # Sketch → Photo generator (place here after training)
├── cyclegan_G_BA_final.pth          # Photo → Sketch generator (place here after training)
└── README.md
```

## Getting Started

### Step 1: Train Model (Kaggle)
1. Upload `GANt3.ipynb` to Kaggle
2. Add one of these datasets:
   - TU-Berlin Sketch Database
   - Sketchy Dataset
   - Google QuickDraw Dataset
3. Run all cells sequentially
4. Generators will be saved to `/kaggle/working/` as:
   - `cyclegan_G_AB_final.pth` (Sketch → Photo)
   - `cyclegan_G_BA_final.pth` (Photo → Sketch)

### Step 2: Prepare Files
1. Download both trained `.pth` files from Kaggle
2. Copy to this folder (Task3/):
   - `cyclegan_G_AB_final.pth`
   - `cyclegan_G_BA_final.pth`

### Step 3: Run Gradio App
```bash
cd Task3
pip install gradio torch torchvision pillow matplotlib
python gradio_app_task3.py
```

The app will launch at `http://localhost:7860`

## Model Architecture

### Generators: ResNet-based
**Configuration**:
- Architecture: ResNet with 6 residual blocks
- Normalization: Instance Normalization (not Batch Norm)
- Input/Output: 128×128 RGB images

**Components**:
1. Initial convolution (7×7 kernel, 64 filters)
2. 2 Downsampling layers (stride=2)
3. 6 Residual blocks with skip connections
4. 2 Upsampling layers (transpose convolution)
5. Final convolution + Tanh activation

### Discriminators: PatchGAN
- Patch-based classification (16×16 patches)
- 4 convolution layers
- Instance Normalization
- LeakyReLU activations

## Loss Functions

### 1. Adversarial Loss (GAN Loss)
Ensures generated images are realistic and domain-appropriate
```
L_GAN = E[log D_A(x)] + E[log(1 - D_A(G_BA(y)))]
       + E[log D_B(y)] + E[log(1 - D_B(G_AB(x)))]
```

### 2. Cycle Consistency Loss (λ=10)
**Key Innovation**: Ensures reversibility without paired data
```
L_cycle = ||x - G_BA(G_AB(x))||_1 + ||y - G_AB(G_BA(y))||_1
```
- **Purpose**: A → B → A should return to A (and vice versa)
- **Benefit**: Prevents mode collapse and preserves structure

### 3. Identity Loss (λ=5)
Preserves color and appearance when translating within same domain
```
L_identity = ||x - G_BA(x)||_1 + ||y - G_AB(y)||_1
```

### Total Loss
```
L_total = L_GAN + λ_cycle * L_cycle + λ_identity * L_identity
```

## Training Configuration
- **Optimizer**: Adam
- **Learning Rate**: 0.0002
- **Betas**: (0.5, 0.999)
- **Batch Size**: 4 (optimized for Kaggle T4 x2)
- **Image Size**: 128×128
- **Epochs**: 50+
- **Mixed Precision**: Yes (torch.cuda.amp)
- **ResNet Blocks**: 6 (optimized for memory)

## Key Innovation: Unpaired Learning

### Why CycleGAN?
- ✓ **No paired data needed** - domains can be independent
- ✓ **Scalable** - easier data collection
- ✓ **Flexible** - works with any two image domains
- ✓ **Stable** - cycle consistency provides strong supervision

### How Cycle Consistency Works
```
Domain A (Sketch)  ──G_AB──→  Domain B (Photo)
                                    ↓
                               G_BA (Photo→Sketch)
                                    ↓
                             Reconstructed A

If reconstruction ≈ original → Cycle consistency achieved
```

## Features in Gradio App

### 1. Sketch → Photo Tab
- Upload sketch image
- Generate realistic photo
- Instant translation

### 2. Photo → Sketch Tab
- Upload photo
- Generate sketch
- Edge extraction

### 3. Cycle Consistency Tab
- Demonstrate A → B → A cycle
- Show intermediate translation
- Show reconstruction quality
- Toggle direction

### 4. Random Generation Tab
- Generate images from random noise
- Create photo samples
- Create sketch samples

### 5. Architecture Tab
- Detailed model architecture
- Loss function equations
- Training strategy explained

### 6. Datasets Tab
- Information about all datasets
- How to prepare data
- Export and deployment guide

## Datasets

### TU-Berlin Sketch Database
- **URL**: https://huggingface.co/datasets/sdiaeyu6n/tu-berlin
- **Domain A**: Hand-drawn sketches
- **Domain B**: Real objects
- **Classes**: 250+ object categories
- **Unpaired**: Domains are independent

### Sketchy Dataset
- **URL**: https://www.kaggle.com/datasets/sharanyasundar/sketchy-dataset
- **Domain A**: Sketches
- **Domain B**: Photos
- **Structure**: See README.md
- **Scalable**: Large dataset

### Google QuickDraw
- **URL**: https://www.kaggle.com/c/quickdraw-doodle-recognition/data
- **Domain A**: User doodles
- **Domain B**: Real objects
- **Classes**: 340+ categories
- **Large Scale**: Millions of doodles

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

## Model Weights

### Files
- **cyclegan_G_AB_final.pth**: Sketch → Photo generator (~100-150 MB)
- **cyclegan_G_BA_final.pth**: Photo → Sketch generator (~100-150 MB)

### Format
- PyTorch state dict
- Compatible with ResNetGenerator class
- Saved with torch.save()

## Usage Examples

### Basic Translation
```python
# In Gradio interface:
1. Go to "Sketch → Photo" tab
2. Upload sketch image
3. Click "Generate Photo"
4. Download result
```

### Cycle Consistency Demo
```python
# In Gradio interface:
1. Go to "Cycle Consistency" tab
2. Select direction: "sketch_to_photo"
3. Upload image
4. See A → B → A transformation
5. Compare input vs reconstruction
```

### Random Generation
```python
# In Gradio interface:
1. Go to "Random Generation" tab
2. Set number of samples (5-25)
3. Click "Generate Samples"
4. View photo and sketch grids
```

## Performance Characteristics

### Computational Cost
- **Memory**: 4-8 GB GPU (Kaggle T4)
- **Speed**: ~0.1-0.2s per image on GPU
- **Speed**: ~0.5-1s per image on CPU

### Output Quality
- **Resolution**: 128×128 (trainable)
- **Diversity**: High (unpaired training)
- **Realism**: Good with proper training
- **Consistency**: Excellent (cycle loss)

## Troubleshooting

### Models not loading
```bash
# Check file names:
# - cyclegan_G_AB_final.pth (Sketch→Photo)
# - cyclegan_G_BA_final.pth (Photo→Sketch)

# Verify path: ls -la Task3/*.pth
```

### Poor quality output
- Ensure models are fully trained (50+ epochs)
- Check input domain matches training data
- Verify .pth files are correct models

### CUDA out of memory
- Use CPU: `DEVICE = 'cpu'`
- Reduce image size: `IMAGE_SIZE = 64`
- Use smaller ResNet blocks

### Cycle consistency not working
- Check G_BA model is loaded
- Verify cycle direction selection
- Ensure input size matches (128×128)

## Advanced Usage

### Change Model Behavior
Edit in `gradio_app_task3.py`:
```python
IMAGE_SIZE = 64  # Reduce memory usage
RESNET_BLOCKS = 4  # Fewer blocks = faster inference
```

### Use Custom Models
```python
custom_g_ab_path = "path/to/custom_g_ab.pth"
custom_g_ba_path = "path/to/custom_g_ba.pth"
G_AB.load_state_dict(torch.load(custom_g_ab_path))
G_BA.load_state_dict(torch.load(custom_g_ba_path))
```

### Export for Web Deployment
```python
# Convert to ONNX for web deployment
torch.onnx.export(G_AB, dummy_input, "g_ab.onnx")
torch.onnx.export(G_BA, dummy_input, "g_ba.onnx")
```

## References

### Papers
- **CycleGAN**: Unpaired Image-to-Image Translation
  - arXiv:1703.10593 (Zhu et al.)
- **ResNet**: Deep Residual Learning
  - arXiv:1512.03385 (He et al.)
- **Instance Normalization**: Instance Norm vs Batch Norm
  - arXiv:1607.08022

### Key Concepts
- Unpaired learning eliminates data pairing bottleneck
- Cycle consistency prevents mode collapse
- Identity loss improves training stability
- PatchGAN focuses on local realism

## Notes

✓ **Strengths**:
- No paired data needed
- Excellent for domain adaptation
- Cycle consistency guarantees structure preservation
- Works with diverse domains

⚠ **Limitations**:
- Slower convergence than paired methods
- May require more careful hyperparameter tuning
- Training time ~8-12 hours on Kaggle T4 x2

## License & Attribution
For academic use, please cite CycleGAN paper and this implementation.

## Contact & Support
Refer to main project documentation for support and additional resources.

---

**Last Updated**: 2026-04-08
**Framework**: PyTorch
**Python Version**: 3.10+
