# Image Generation using Stable Diffusion

This is a deep learning-based project that transforms text prompts into high quality images. Leveraging the power of Stable Diffusion, this model enables users to generate visually stunning artwork, making it a valuable tool for artists and designers. By translating complex descriptions into detailed visuals, it highlights the immense potential of AI in creative expression.

---

## Table of Contents
## Table of Contents
- [Features](#features)
- [Technology Stack](#technology-stack)
- [Installation Guide](#installation-guide)
- [Usage Guide](#usage-guide)
- [Sample Input/Output](#sample-inputoutput)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

- **Text-to-Image Generation**: Create detailed images from text descriptions
- **Customizable Outputs**: Control resolution, steps, guidance scale and seed
- **Image-to-Image**: Modify existing images with text prompts
- **Optimized Performance**: Works on both GPU and CPU
- **User-Friendly Interface**: Simple command line interface

---

## Technology Stack

| Component | Purpose | Why I Chose It |
|-----------|---------|----------------|
| Python 3.9+ | Main language | Most support for ML libraries |
| PyTorch | Neural network framework | Best for research and experimentation |
| Hugging Face Diffusers | Pretrained models | Easy-to-use implementation |
| Transformers | Text processing | Handles prompt encoding efficiently |
| CUDA | GPU acceleration | Speeds up generation 5-10x |
| Pillow | Image handling | Simple image saving/loading |
---

## How It Actually Works

1. **From Words to Numbers**  
   Your text prompt gets converted into numerical embeddings using CLIP - this helps the AI "understand" the request.

2. **The Artistic Process**  
   The model starts with random noise (like TV static) and gradually refines it over multiple steps (typically 30-50), guided by your text.

3. **Efficient Generation**  
   Instead of working directly at high resolution, the AI:
   - Operates in a compressed "latent space" (64×64)
   - Uses a VQ-GAN decoder to upscale to 512×512
   - This makes it feasible to run on consumer hardware




## Installation Guide

## 📋 Prerequisites

- **Hardware**:
  - NVIDIA GPU (recommended) with at least 8GB VRAM
  - 10GB+ free disk space for models
  - 16GB+ RAM for optimal performance

- **Software**:
  - Python 3.7+
  - CUDA 11.7+ (if using NVIDIA GPU)
  - cuDNN (for GPU acceleration)

## 🛠 Installation Guide

### Method 1: Conda (Recommended for Isolation)

```bash
# Create and activate conda environment
conda create -n sd_env python=3.8 -y
conda activate sd_env

# Install PyTorch with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# Clone repository
git clone https://github.com/pavit15/Image-Generation-using-Stable-Diffusion.git
cd Image-Generation-using-Stable-Diffusion

# Install dependencies
pip install -r requirements.txt

---

### Method 2: Virtual Environment

```bash
# Create virtual environment
python -m venv sd_venv
source sd_venv/bin/activate  # Linux/Mac
# or sd_venv\Scripts\activate  # Windows
```

### Post-Installation Verification
```bash
python -c "import torch; print(torch.cuda.is_available())"
```
Should output 'True' if GPU is properly configured

## Usage Guide

Running the Jupyter Notebook
```bash
jupyter notebook StableDiffusion.ipynb
```

### Basic Image Generation
1. Open the notebook in Jupyter
2. Run all cells sequentially
3. Modify these key parameters in the generation cell:
   prompt = "A beautiful landscape of mountains at sunset"
   negative_prompt = "blurry, low quality, distortion"
   num_inference_steps = 50  # More steps = better quality
   guidance_scale = 7.5      # Higher = more prompt adherence
   height, width = 512, 512  # Image dimensions

### Application Workflow

1. **Enter Prompt**
   - Provide a textual description of the image you want to generate.

2. **Adjust Parameters**
   - Set parameters like resolution, number of inference steps, and guidance scale for customized results.

3. **Generate Image**
   - Click the "Generate" button to create the image.

4. **Save or Modify Image**
   - Save the generated image or tweak parameters to refine the output.

5. **Batch Generation (Optional)**
   - Generate multiple images by providing a list of prompts.

---

## Sample Input/Output

**Input Prompt:**
   _"a tower made of chocolate."_

## Generated Output  

![ComfyUI_00003_](https://github.com/user-attachments/assets/ed0eb4b1-ab15-4da4-8dc1-33df2bbcaae5)  

![image](https://github.com/user-attachments/assets/ed0eec2c-c8ec-4b4e-9932-154ec54b2281)  

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce image size (`--width 512 --height 512`) |
| Blurry images | Increase steps (`--steps 50`) |
| Strange artifacts | Adjust guidance scale (`--guidance_scale 7.5`) |
| Slow generation | Use `--half` precision |
---

## License

This project is licensed under the **MIT License**. Feel free to use, modify, and distribute it under the license terms.
---
