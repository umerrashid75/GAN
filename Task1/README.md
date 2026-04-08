# Task 1: Tackling Mode Collapse in GANs

## Overview
This folder contains:
- **GANt1.ipynb** - Complete PyTorch implementation of DCGAN and WGAN-GP
- **gradio_app_task1.py** - Interactive Gradio application for model comparison

## Project Structure
```
Task1/
├── GANt1.ipynb                  # Training notebook
├── gradio_app_task1.py          # Gradio web interface
├── dcgan_G_final.pth            # DCGAN generator (place here after training)
├── wgangp_G_final.pth           # WGAN-GP generator (place here after training)
└── README.md
```

## Getting Started

### Step 1: Train Models (Kaggle)
1. Upload `GANt1.ipynb` to Kaggle
2. Add either "anime-faces-64x64" or "pokemon-sprites" dataset
3. Run all cells sequentially
4. Models will be saved to `/kaggle/working/` as:
   - `dcgan_G_final.pth`
   - `wgangp_G_final.pth`

### Step 2: Prepare Files
1. Download the trained `.pth` files from Kaggle
2. Copy them to this folder (Task1/):
   - `dcgan_G_final.pth`
   - `wgangp_G_final.pth`

### Step 3: Run Gradio App
```bash
cd Task1
pip install gradio torch torchvision pillow matplotlib
python gradio_app_task1.py
```

The app will launch at `http://localhost:7860`

## Models

### DCGAN (Baseline)
- **Loss Function**: Binary Cross Entropy (BCE)
- **Updates**: 1 discriminator per generator update
- **Issue**: Prone to mode collapse
- **Output**: 64×64 RGB images

### WGAN-GP (Advanced)
- **Loss Function**: Wasserstein Loss + Gradient Penalty (λ=10)
- **Updates**: 5 critic updates per generator update
- **Benefit**: Mitigates mode collapse through stable training
- **Output**: 64×64 RGB images

## Key Differences

| Aspect | DCGAN | WGAN-GP |
|--------|-------|---------|
| Loss Function | BCE | Wasserstein |
| Output Activation | Sigmoid | None |
| Normalization | Batch Norm | Instance Norm |
| Training Stability | Lower | Higher |
| Mode Collapse | Yes | Minimal |

## Hyperparameters
- **Image Size**: 64×64
- **Batch Size**: 64
- **Learning Rate**: 0.0002
- **Epochs**: 50
- **Latent Vector**: 100-dimensional

## Features in Gradio App
1. **Model Comparison** - Side-by-side generation comparison
2. **DCGAN Generator** - Generate samples from baseline model
3. **WGAN-GP Generator** - Generate samples from advanced model
4. **Information Panel** - Architecture and training details

## Troubleshooting

### Models not loading
- Ensure `.pth` files are in the Task1 folder
- Check file names match exactly: `dcgan_G_final.pth`, `wgangp_G_final.pth`

### Out of memory error
- Reduce batch size in notebook if training locally
- Use Kaggle GPU with increased memory

### Poor quality samples
- Ensure models are fully trained (50+ epochs)
- Adjust learning rate if needed
- Check dataset quality

## References
- **Paper**: Wasserstein GAN with Gradient Penalty (arXiv:1704.00028)
- **Architecture**: Deep Convolutional GANs (arXiv:1511.06434)
- **Dataset**: Pokemon Sprites or Anime Faces (64×64)

## Notes
- WGAN-GP generates more diverse and realistic samples
- DCGAN may collapse to single mode if not trained carefully
- Both models use transposed convolutions for generation
- Training on GPU recommended (Kaggle T4 or better)

## Contact & Support
For issues or improvements, refer to the main project documentation.
