### 📝 Complete Walkthrough: Automated Video Model Benchmark Setup

---

## 🚀 Phase 0: The Universal Setup (ComfyUI via Docker)

This setup is the foundation of your benchmarking, whether you are running on your Mac (for slow validation) or a cloud GPU (for speed).

### 1\. **Initial Setup (Local & Cloud)**

| Step                 | Action                                                       | Command/Location                                                                           | Notes                                                                                                                      |
| :------------------- | :----------------------------------------------------------- | :----------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------- |
| **Install Docker**   | Install Docker Desktop (Mac) or Docker Engine (Linux/Cloud). | `https://www.docker.com/products/docker-desktop/`                                          | Docker ensures consistency between your Mac and the colocation server.                                                     |
| **Clone ComfyUI**    | Get the ComfyUI source code.                                 | `git clone https://github.com/comfyanonymous/ComfyUI.git`                                  | This creates the base structure, including the `/models` directory.                                                        |
| **Download Manager** | Install the **ComfyUI Manager** custom node.                 | `cd ComfyUI/custom_nodes` <br> `git clone https://github.com/ltdrdata/ComfyUI-Manager.git` | The Manager simplifies installing custom nodes and updating the software, which is critical for these cutting-edge models. |
| **Run ComfyUI**      | Start the server.                                            | `python main.py`                                                                           | You will eventually use a dedicated Docker command to start this on your cloud GPU.                                        |

### 2\. **Model File Placement**

When you download a model file (`.safetensors` or `.ckpt`), it must go into the correct subdirectory inside your `ComfyUI/models/` folder.

| Model Type                                   | Destination Folder                |
| :------------------------------------------- | :-------------------------------- |
| **UNET/Diffusion** (The main model)          | `ComfyUI/models/diffusion_models` |
| **VAE** (The decoder)                        | `ComfyUI/models/vae`              |
| **Text Encoder** (The prompt interpreter)    | `ComfyUI/models/text_encoders`    |
| **CLIP Vision** (The image-to-video encoder) | `ComfyUI/models/clip_vision`      |

---

## 🥇 Model 1: Wan-AI (Wan2.2 I2V 14B)

Wan-AI is highly optimized, particularly its Mixture-of-Experts (MoE) architecture, making it a strong contender for high throughput.

### 1\. **Download & Implementation**

| Component        | File Name (Example)                        | Destination         | Download Method (Hugging Face)                                                    |
| :--------------- | :----------------------------------------- | :------------------ | :-------------------------------------------------------------------------------- |
| **UNET Model**   | `wan2.2_i2v_a14b.safetensors`              | `/diffusion_models` | Use `huggingface-cli` or `modelscope-cli` download (recommended for large files). |
| **VAE**          | `wan2.2_vae.safetensors`                   | `/vae`              | Download separately.                                                              |
| **Text Encoder** | (A large CLIP model, often used by others) | `/text_encoders`    | Check the specific GitHub instructions for the required CLIP model.               |
| **Custom Nodes** | `ComfyUI-WanVideoWrapper`                  | `/custom_nodes`     | Use ComfyUI Manager to search for and install.                                    |

### 2\. **How to Run/Implement in ComfyUI**

- **Workflow:** You need a specific JSON workflow file (often available on the Wan-AI Hugging Face or GitHub pages).
- **Nodes Used:** Look for **`UNETLoader`**, **`VAELoader`**, **`CLIPLoader`**, and the specialized **`Wan22ImageToVideoLatent`** node.
- **I2V Specifics:** This workflow uses the **`Wan22ImageToVideoLatent`** node to take your input image and encode it _before_ it hits the KSampler. This is how it conditions the video on your **Key Image**.

---

## 🥈 Model 2: HunyuanVideo (13B)

Hunyuan is known for its high visual quality and strong temporal coherence.

### 1\. **Download & Implementation**

| Component        | File Name (Example)                                  | Destination         | Key Note                                                                                    |
| :--------------- | :--------------------------------------------------- | :------------------ | :------------------------------------------------------------------------------------------ |
| **UNET Model**   | `hunyuan_video_image_to_video_720p_bf16.safetensors` | `/diffusion_models` | There are often V1 ("concat") and V2 ("replace") variants. **Test both** in your shootout\! |
| **VAE**          | `hunyuan_video_vae_bf16.safetensors`                 | `/vae`              | Download separately.                                                                        |
| **CLIP Vision**  | `llava_llama3_vision.safetensors`                    | `/clip_vision`      | This is the crucial encoder that interprets your input image for the model.                 |
| **Custom Nodes** | `ComfyUI-HunyuanVideoWrapper`                        | `/custom_nodes`     | Use ComfyUI Manager to search for and install this wrapper.                                 |

### 2\. **How to Run/Implement in ComfyUI**

- **Workflow:** Look for the **Hunyuan Image-to-Video JSON** workflow.
- **Nodes Used:** The workflow will include a **`LoadImage`** node wired directly to the custom Hunyuan nodes, which use the **CLIP Vision** model to embed the image data along with the text prompt.
- **I2V Specifics:** The Hunyuan model incorporates the image information via latent concatenation, which makes it very strong at adhering to the style and composition of the initial image.

---

## 🥉 Model 3: Mochi 1 (10B AsymmDiT)

Mochi 1 is highly permissive (Apache 2.0 license) and designed for consumer-grade GPU cards like the 4090, making it fast and easy to scale.

### 1\. **Download & Implementation**

| Component        | File Name (Example)             | Destination         | Key Note                                                                                    |
| :--------------- | :------------------------------ | :------------------ | :------------------------------------------------------------------------------------------ |
| **UNET Model**   | `mochi1_fp8_single.safetensors` | `/diffusion_models` | **FP8 (8-bit precision)** is highly recommended for speed and lower VRAM usage on the 4090. |
| **VAE**          | `mochi1_vae.safetensors`        | `/vae`              | Download separately.                                                                        |
| **Text Encoder** | `t5xxl_fp16.safetensors`        | `/text_encoders`    | A large text encoder is often required for the AsymmDiT architecture.                       |
| **Custom Nodes** | `ComfyUI-MochiWrapper`          | `/custom_nodes`     | Use ComfyUI Manager to search for and install.                                              |

### 2\. **How to Run/Implement in ComfyUI**

- **Workflow:** Mochi 1 uses a dedicated **`Mochi 1 Sampler`** node where you control the video's frames, resolution, and steps.
- **I2V Specifics:** Mochi 1 is fundamentally an Image-to-Video model. Its strong adherence to the initial frame and its high speed make it ideal for volume.

---

---

## 🎯 Step-by-Step: Cloud GPU Setup for Shootout

### Option A: RunPod (Recommended for Beginners)

1. **Create Account:** https://www.runpod.io/
2. **Select Template:** Search for "ComfyUI" in community templates
3. **Choose GPU:** Select RTX 4090 (24GB VRAM minimum)
4. **Launch Instance:** Click "Deploy" - takes 2-3 minutes
5. **Access ComfyUI:** Click "Connect" → "HTTP Service" → Opens ComfyUI in browser

### Option B: Vast.ai (Cheapest Option)

1. **Create Account:** https://vast.ai/
2. **Search Offers:** Filter by GPU (RTX 4090), VRAM (>24GB), and "Docker" support
3. **Select Instance:** Sort by $/hr (typically $0.80-1.50/hr for 4090)
4. **Template:** Use `pytorch/pytorch:latest` or search for ComfyUI templates
5. **SSH Access:** Connect via SSH, install ComfyUI manually:
   ```bash
   git clone https://github.com/comfyanonymous/ComfyUI.git
   cd ComfyUI
   pip install -r requirements.txt
   python main.py --listen 0.0.0.0
   ```

### Option C: Lambda Labs (Fastest Setup)

1. **Create Account:** https://lambdalabs.com/
2. **Launch Instance:** Select A10G or RTX 6000 Ada (24GB+)
3. **Pre-installed CUDA:** No driver setup needed
4. **Install ComfyUI:** Run setup script (same as Vast.ai)

---

## 📥 Exact Download Commands

### Wan-AI 2.2 I2V 14B

```bash
# Install Hugging Face CLI
pip install huggingface-hub

# Download model files (run from ComfyUI directory)
huggingface-cli download WanAI/Wan2.2-I2V-14B-fp16 \
  --local-dir models/diffusion_models/wan22 \
  --include "*.safetensors"

huggingface-cli download WanAI/Wan2.2-I2V-14B-fp16 \
  --local-dir models/vae \
  --include "vae/*.safetensors"

# Install custom nodes
cd custom_nodes
git clone https://github.com/kijai/ComfyUI-WanVideoWrapper.git
cd ..
```

### HunyuanVideo 13B

```bash
# Download model files
huggingface-cli download hunyuanvideo/HunyuanVideo \
  --local-dir models/diffusion_models/hunyuan \
  --include "hunyuan_video_*.safetensors"

huggingface-cli download hunyuanvideo/HunyuanVideo \
  --local-dir models/clip_vision \
  --include "clip_vision/*.safetensors"

# Install custom nodes
cd custom_nodes
git clone https://github.com/kijai/ComfyUI-HunyuanVideoWrapper.git
cd ..
```

### Mochi 1 10B

```bash
# Download FP8 model (optimized for 4090)
huggingface-cli download genmo/mochi-1-preview \
  --local-dir models/diffusion_models/mochi \
  --include "mochi_preview_fp8_single.safetensors"

huggingface-cli download genmo/mochi-1-preview \
  --local-dir models/vae \
  --include "vae/*.safetensors"

# Install custom nodes
cd custom_nodes
git clone https://github.com/kijai/ComfyUI-MochiWrapper.git
cd ..
```

---

## 🐳 Docker Compose Setup (Advanced)

For consistent environments between local Mac and cloud GPU:

```yaml
# docker-compose.yml
version: "3.8"

services:
  comfyui:
    image: pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime
    container_name: comfyui-video-gen
    volumes:
      - ./ComfyUI:/workspace/ComfyUI
      - ./models:/workspace/ComfyUI/models
      - ./input:/workspace/ComfyUI/input
      - ./output:/workspace/ComfyUI/output
    ports:
      - "8188:8188"
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    command: >
      bash -c "cd /workspace/ComfyUI &&
               pip install -r requirements.txt &&
               python main.py --listen 0.0.0.0 --port 8188"
```

**Run with:**

```bash
docker-compose up -d
```

---

## 🎯 Next Step: Running the Benchmark

Once ComfyUI is running and models are installed:

1. **Load workflow:** Download the model-specific workflow JSON from each model's Hugging Face page
2. **Upload test keyframe:** Place your test image in `ComfyUI/input/`
3. **Run each model:** Execute workflow, record times
4. **Compare outputs:** Save videos to `output/` for side-by-side review

**Automated benchmarking scripts** (Python API automation) can be added later, but manual testing is sufficient for the initial shootout.

---

## 📚 Reference Links

- **ComfyUI Documentation:** https://github.com/comfyanonymous/ComfyUI
- **Wan-AI Model Page:** https://huggingface.co/WanAI/Wan2.2-I2V-14B-fp16
- **HunyuanVideo Model Page:** https://huggingface.co/hunyuanvideo/HunyuanVideo
- **Mochi 1 Model Page:** https://huggingface.co/genmo/mochi-1-preview
- **ComfyUI Manager:** https://github.com/ltdrdata/ComfyUI-Manager
