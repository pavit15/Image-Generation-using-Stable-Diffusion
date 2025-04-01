# Image Generation using Stable Diffusion

This is a deep learning-based project that transforms text prompts into high quality images. Leveraging the power of Stable Diffusion, this model enables users to generate visually stunning artwork, making it a valuable tool for artists and designers. By translating complex descriptions into detailed visuals, it highlights the immense potential of AI in creative expression.

---

## Table of Contents
- Features
- Technologies Used
- [How It Actually Works](#how-it-actually-works)
- [Tech Stack](#tech-stack)
- Installation Guide
- Usage Guide
- Sample Input/Output
- License

---

## Features
- **Text-to-Image Generation**: Create detailed images from textual prompts.
- **Customizable Outputs**: Adjust parameters like resolution, sampling steps, and guidance scale for better results.
- **GUI Interface**: User-friendly graphical interface for seamless interaction.
- **Batch Processing**: Generate multiple images at once.
- **Model Fine-tuning Support**: Option to use fine-tuned models for specialized outputs.

---

## Technologies Used
- **Python**: Core language for implementation.
- **Stable Diffusion**: Deep learning model for text-to-image generation.
- **Hugging Face Diffusers**: Library for running diffusion models.
- **PyTorch**: Deep learning framework.
- **Gradio**: For creating a simple and interactive user interface.
- **Transformers**: For handling model processing.
- **Pillow**: For image processing.

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


## Tech Stack


| Component | Purpose | Why I Chose It |
|-----------|---------|----------------|
| Python 3.9+ | Main language | Most support for ML libraries |
| PyTorch | Neural network framework | Best for research and experimentation |
| Hugging Face Diffusers | Pretrained models | Easy-to-use implementation |
| Transformers | Text processing | Handles prompt encoding efficiently |
| CUDA | GPU acceleration | Speeds up generation 5-10x |
| Pillow | Image handling | Simple image saving/loading |
## Installation Guide

### Prerequisites
- Install Python (version 3.8 or above) from the official [Python website](https://www.python.org/).
- Install VS Code or Jupyter Notebook for code execution.

### Setup Steps

1. **Clone the Repository**
   ```bash
   git clone <repository_url>
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
   ```

4. **Configure Python Interpreter**
   Press `Ctrl + Shift + P` in VS Code, search for "Python: Select Interpreter", and choose the one from the `venv` folder.

---

## Usage Guide

### Launch the Application
To start the image generation interface, run:
```bash
python app.py
```

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

## License
This project is licensed under the **MIT License**. Feel free to use, modify, and distribute it under the license terms.

---



