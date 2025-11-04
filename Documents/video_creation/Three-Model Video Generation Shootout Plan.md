# 🏁 AI Video Model Shootout Plan: Cloud Benchmarking

This plan outlines the steps to perform a cost-effective A/B/C test using temporary cloud GPU resources to select the best offline video generation model for a high-throughput commercial workflow.

**GOAL:** Select the model that offers the best balance of **Temporal Coherence (Quality)** and **Inference Speed (Throughput)** with a consistent "Key Image" input.

---

## Step 1: Pre-Test Setup (Your M4 Pro)

This ensures all tests start from an identical, high-quality, pre-selected input image.

1.  **Define a Consistent Key Image:** Generate a single, highly detailed, non-controversial image that will be used for all three tests (e.g., "A futuristic metallic sports car parked on a misty street at night.").
2.  **Define Consistent Test Parameters (Metadata):**
    - **Video Length:** 4 seconds.
    - **Resolution:** 512x720 (Portrait aspect ratio).
    - **Frame Rate:** 24 FPS (Frames Per Second).
    - **Prompt:** Use the _exact same_ descriptive prompt for all three model runs (e.g., "A slow, cinematic camera dollies smoothly forward over the hood of the car, dynamic lighting, ultra-realistic, octane render.").

---

## Step 2: Cloud GPU Setup (The "Lab")

This minimizes upfront hardware costs and provides the necessary high-end GPU power.

1.  **Choose a Cloud Platform:** Select an hourly GPU rental provider (e.g., Runpod, Vast.ai, Lambda Labs) known for fast setup.
2.  **Recommended GPU:** Look for an instance with **at least 24GB VRAM** (e.g., NVIDIA RTX 4090, A10G, or A100/H100) to ensure the models can run efficiently.
3.  **Launch ComfyUI:** Start a cloud instance that runs the **ComfyUI environment**. ComfyUI is the standard platform for running these models offline.
4.  **Install Models:** Download and install the necessary weights and ComfyUI custom nodes for the three contenders:
    - **Wan-AI 2.2 I2V 14B**
    - **HunyuanVideo 13B**
    - **Mochi 1 10B**

---

## Step 3: The Benchmark Runs (The Test)

Run the same test three times. Use automated metrics where possible to reduce subjectivity.

| Model            | Setup in ComfyUI                                                                        | Focus for Audit                                                                                            | Inference Time (seconds) | Quality Metrics |
| :--------------- | :-------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------- | :----------------------- | :-------------- |
| **Wan-AI**       | Load the `I2V-14B` checkpoint. Input the Key Image as the control image.                | **SPEED and Image Adherence.** How fast is the generation? Does the output strongly match the input image? | _Record actual time_     | _See below_     |
| **HunyuanVideo** | Load the `HunyuanVideo 13B` checkpoint. Use the Key Image for first-frame conditioning. | **CINEMATIC QUALITY.** How smooth is the camera movement? Is the lighting and texture highly realistic?    | _Record actual time_     | _See below_     |
| **Mochi 1**      | Load the `Mochi 1 10B` checkpoint.                                                      | **MOTION SMOOTHNESS.** How fluid is the motion? Is the video stable without flickering?                    | _Record actual time_     | _See below_     |

### Quality Metrics to Capture:

**Objective Metrics (Automated):**

- **Frame-to-Frame Consistency (SSIM):** Measures similarity between consecutive frames (higher = more stable, target >0.85)
- **Flash Detection:** Count frames with >50% brightness change (target = 0 flash frames)
- **Temporal Variance:** Standard deviation of pixel changes across frames (lower = smoother)
- **VRAM Usage:** Peak memory consumption during generation

**Subjective Metrics (Manual Review):**

- **First Frame Fidelity:** Does frame 1 match the input keyframe? (1-5 scale)
- **Motion Smoothness:** Is camera movement fluid without jitter? (1-5 scale)
- **Artifact Count:** Warping, ghosting, or distortion visible? (count instances)
- **Overall Commercial Viability:** Would you deliver this to a client? (Yes/No)

---

## Step 4: Decision Criteria

The final decision should be based on the model's commercial viability for your high-throughput business model.

### Weighted Scoring System:

| Criterion                | Weight | Measurement                              | Threshold                     |
| ------------------------ | ------ | ---------------------------------------- | ----------------------------- |
| **Speed**                | 40%    | Inference time (seconds per 4-sec video) | Must be <30 sec on RTX 4090   |
| **Temporal Quality**     | 35%    | SSIM + Flash Count + Manual Review       | SSIM >0.85, Zero flash frames |
| **First Frame Fidelity** | 15%    | Visual match to input keyframe           | Manual rating >4/5            |
| **VRAM Efficiency**      | 10%    | Peak memory usage                        | Must fit in 24GB (4090)       |

### Selection Logic:

1. **Eliminate any model that fails threshold requirements** (too slow, too much VRAM, flash frames detected)
2. **Calculate weighted scores** for remaining models
3. **Commercial Winner:** Highest weighted score
4. **Tiebreaker:** Faster inference time wins

### Hardware Planning:

- **Winner model dictates** the final colocation configuration
- **Example:** If Mochi 1 wins at 15 sec/video → 2x 4090s = ~8 videos/min = 1,000+ videos/day capacity
- **Budget:** Winner's VRAM + speed determines whether you need 2x, 3x, or 4x GPUs

### Cost Tracking:

Record cloud GPU costs during shootout:

- **Hourly rate:** $**\_**/hr
- **Total hours:** **\_** hours
- **Total shootout cost:** $**\_**
- **Cost per model tested:** $**\_**

This ensures your $50-150 testing investment is optimized and hardware decisions are data-driven.
