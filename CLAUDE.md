# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Kaggle Jupyter notebook that implements and compares two Generative Adversarial Network (GAN) architectures for image generation:
- **DCGAN** (Deep Convolutional GAN): Uses BCEWithLogitsLoss, Batch Normalization, single discriminator update per generator update
- **WGAN-GP** (Wasserstein GAN with Gradient Penalty): Uses Wasserstein loss, Instance Normalization, gradient penalty, 5 critic updates per generator update

The goal is to demonstrate mode collapse mitigation and training stability improvements of WGAN-GP over DCGAN.

## How to Run

**Environment:**
- Notebook is designed for Kaggle with GPU acceleration (T4 GPUs recommended)
- Works with both single and multiple GPU setups via `nn.DataParallel`
- Reproducibility is ensured via SEED=42 set in cell 1

**To run the notebook:**
1. Upload to Kaggle and add either the "anime-faces-64x64" or "pokemon-sprites" dataset
2. Execute cells sequentially from top to bottom
3. Training outputs (checkpoints, visualizations, loss logs) are saved to `/kaggle/working/`

**To adapt for local development:**
- Update `DATA_ROOT` path in cell 3 to point to your local dataset directory
- Change output paths from `/kaggle/working/` to your local output directory
- For CPU-only development, set `device = 'cpu'` in cell 2

## Key Hyperparameters

All hyperparameters are defined in cell 4 and can be easily modified:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `IMAGE_SIZE` | 64 | Input/output image size |
| `NZ` | 100 | Latent vector dimensionality |
| `NGF` / `NDF` | 64 | Generator/Discriminator feature map base multiplier |
| `BATCH_SIZE` | 64 | Batch size for training |
| `LR` | 0.0002 | Learning rate (Adam) |
| `DCGAN_EPOCHS` / `WGANGP_EPOCHS` | 50 | Number of training epochs |
| `LAMBDA_GP` | 10 | Gradient penalty weight (WGAN-GP only) |
| `CRITIC_ITERS` | 5 | Critic updates per generator update (WGAN-GP only) |

## Architecture Highlights

**Shared Generator (cells 9):**
- 4 transposed convolution layers with stride 2, upsampling from latent vector to 64×64 RGB image
- Batch Normalization + ReLU activations (except final Tanh)
- Output range: [-1, 1]

**DCGAN Discriminator (cell 9):**
- 4 convolution layers with stride 2, downsampling to binary classification
- Batch Normalization + LeakyReLU(0.2)
- Uses BCEWithLogitsLoss (sigmoid built-in)
- Note: Sigmoid removed from output layer to enable mixed precision training

**WGAN-GP Critic (cell 9):**
- Same layer structure as DCGAN but with Instance Normalization instead of Batch Norm
- No output activation (raw scores)
- Gradient penalty computed from interpolated real/fake samples (cell 14)
- DataParallel disabled to support autograd.grad in gradient penalty computation

## Training Loop Structure

**DCGAN Training (cells 11–12):**
- Each epoch: For each batch → Update D once, Update G once
- Uses `GradScaler` for mixed precision (AMP)
- Loss logged as `dcgan_G_losses` and `dcgan_D_losses`

**WGAN-GP Training (cells 14–15):**
- Each epoch: For each batch → Update C 5 times, Update G once
- Gradient penalty: `gp = ((grad_norm - 1)^2).mean()` where grad_norm is Jacobian norm
- Loss logged as `wgan_G_losses` and `wgan_C_losses`
- DataParallel disabled due to `torch.autograd.grad` requirements in penalty computation

## Data Pipeline

**Cell 3 — Dataset Detection:**
- Auto-detects Kaggle dataset paths for anime-faces or pokemon-sprites
- Falls back to scanning `/kaggle/input/` for any folder with >100 images
- Returns flattened list of image paths

**Cell 7 — ImageDataset Class:**
- Custom PyTorch Dataset that loads images from file paths
- Handles corrupt images gracefully (returns blank if file fails)
- Applies transforms: resize → center crop → normalize to [-1, 1]

## Checkpointing & Outputs

**Checkpoints (saved to `/kaggle/working/checkpoints/`):**
- Every 10 epochs (configurable via `CKPT_INTERVAL`)
- Format: `{dcgan,wgangp}_{G,D/C}_epoch{N}.pth` and `*_final.pth`
- Load via `model.load_state_dict(torch.load(path))`

**Visualizations (saved to `/kaggle/working/outputs/`):**
- `sample_training_images.png` — Dataset preview (16 samples)
- `dcgan_epoch{N}.png` / `wgangp_epoch{N}.png` — Generated grids at each checkpoint
- `dcgan_final_samples.png` / `wgangp_final_samples.png` — 2×5 grid of final samples
- `comparison.png` — Side-by-side DCGAN vs WGAN-GP on same noise
- `loss_curves.png` — Training loss plots for both models
- `loss_log.csv` — Per-epoch loss values

## Inference & Deployment

**Cell 25 — `generate_inference_images()` Function:**
- Loads a saved generator checkpoint from disk
- Generates N images from pure noise
- Returns tiled grid as numpy array ready for web display
- This function is designed to be copied into Gradio/Streamlit backends

Example usage:
```python
dcgan_img = generate_inference_images('dcgan', num_images=16, device=device)
wgan_img = generate_inference_images('wgan-gp', num_images=16, device=device)
```

## Important Technical Notes

1. **Mixed Precision (AMP):** Both models use `GradScaler` and `autocast()` for memory efficiency and speed on Tensor Cores
2. **Reproducibility:** SEED=42 is set globally; results are deterministic if `cudnn.deterministic = True` remains set
3. **GPU Memory:** Call `log_gpu_memory()` to monitor allocation during training
4. **Gradient Penalty Computation:** Must run on main device (not DataParallel) due to `autograd.grad` inside the computation
5. **Loss Function Design:** DCGAN uses BCEWithLogitsLoss (numerically stable). WGAN-GP uses unbounded Wasserstein loss (minimizes critic–generator gap, not probability)
6. **Normalization Impact:** Batch Norm in DCGAN couples batch statistics to training; Instance Norm in WGAN-GP normalizes per-sample, enabling stable gradient penalty

## Common Modifications

**To change dataset:**
- Modify `DATA_ROOT` or `ANIME_PATH`/`POKEMON_PATH` in cell 3

**To adjust training length:**
- Change `DCGAN_EPOCHS` or `WGANGP_EPOCHS` in cell 4

**To tune mode collapse trade-off in WGAN-GP:**
- Increase `LAMBDA_GP` (stronger penalty, smoother gradients, slower convergence)
- Increase `CRITIC_ITERS` (more critic updates, slower training but more stable)

**To experiment with architecture:**
- Modify `NGF` / `NDF` (more filters → more capacity, more memory)
- Modify `NZ` (larger latent space → more diversity but harder to train)
- Add/remove conv layers in Generator or Critic (remember stride must sum to 4 for 64×64)
