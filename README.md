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

### Prerequisites
- Install Python (version 3.8 or above) from the official [Python website](https://www.python.org/).
- Install VS Code or Jupyter Notebook for code execution.

### Setup Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/pavit15/Image-Generation-using-Stable-Diffusion.git
   cd Image-Generation-using-Stable-Diffusion
   ```

2. **Set Up Virtual Environment**
   Open a terminal in VS Code and run:
   ```bash
   python -m venv venv
   venv\Scripts\activate  # On Windows
   # Use `source venv/bin/activate` for Mac/Linux
   ```

3. **Install Dependencies**
   Upgrade `pip` and install the required libraries:
   ```bash
   python -m pip install --upgrade pip
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   pip install diffusers transformers gradio pillow

   OR run:
   pip install -r requirements.txt
   ```

4. **Configure Python Interpreter**
   Press `Ctrl + Shift + P` in VS Code, search for "Python: Select Interpreter", and choose the one from the `venv` folder.

---

## Usage Guide

### For Web Interface (Flask)
```bash
# Navigate to src directory
cd src/app

# Install requirements
```bash
pip install -r requirements.txt

# Run the Flask application
```bash
python main.py

Then open http://localhost:5000 in your browser.

For Command Line Usage
```bash
# From project root
python src/app/main.py --cli --prompt "your prompt here" --output image.png

Available Arguments (CLI Mode)
Argument	Description	Default
--cli	Enable command line mode	False
--prompt	Text description for generation	(required)
--output	Output file path	"output.png"
--steps	Number of diffusion steps	30
--width	Image width	512
--height	Image height	512

Web Interface Features:
- Interactive prompt input
- Real-time preview
- Parameter sliders
- Image gallery
- Download capability

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

Parameters Used:
Steps: 50
Guidance Scale: 7.5
Seed: 1234
Size: 512×512
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



