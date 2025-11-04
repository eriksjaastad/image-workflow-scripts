# 🎬 Image-to-Video Generation: System Audit & Blueprint

## Executive Summary

Your current image-workflow system is **exceptionally well-positioned** for video generation. The key insight: your selection + cropping pipeline maps almost perfectly to **keyframe selection + camera motion definition** for video.

Note: Several scripts referenced in this blueprint are planned (TBD) and not yet in the repo. See `Documents/video_creation/README.md` → “Planned Artifacts” for the list; treat those references as placeholders until implemented.

---

## 📊 PART 1: SYSTEM AUDIT TABLE

| Current Tool                                                | Video Generation Role/Goal                                                                                                                                                                                                       | Required Modification or New System                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| ----------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Tool 1: Selection AI Model (Aesthetic/Commercial Score)** | **Role:** Score keyframe candidates for "video-ability" — static aesthetic quality PLUS temporal stability indicators (depth cues, subject coherence, lack of artifacts that would cause flickering/warping in video).           | **Modifications:**<br>• Add **temporal stability features** to CLIP embeddings: detect flat/uniform backgrounds (good for motion), high-frequency noise patterns (bad for video), depth perception indicators<br>• New model head: Multi-task learning that predicts both _static quality_ (current) + _video viability_ score (new)<br>• Training data: Augment your existing training CSV with a `video_viable` column (binary or 0-1 score). Manually label ~500 images: "would this animate well?" based on composition, subject isolation, background simplicity<br>• Keep existing 512→256→64→1 MLP architecture, add parallel branch: 512→128→1 for video score<br>• **Output:** Two scores per image: `aesthetic_score` (existing) + `video_viability_score` (new)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| **Tool 2: Cropping System**                                 | **Role:** Transform from static crop coordinates to **camera movement path definition**. Instead of defining a fixed rectangle, define START and END rectangles to create zoom/pan motion. Crop coords become motion parameters. | **Modifications:**<br>• **Data structure change:** Existing crop coords `(x1, y1, x2, y2)` normalized [0-1] → Expand to motion descriptor:<br> ```python<br> {<br> "keyframe": (x1, y1, x2, y2), # Your existing crop<br> "motion_type": "zoom_in                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   | zoom_out | pan_left                                                                                                                                                                                                                                                                                                                               | pan_right | static",<br> "motion_intensity": 0.0-1.0, # How much movement<br> "start_frame": (x1_s, y1_s, x2_s, y2_s), # Initial view<br> "end_frame": (x1_e, y1_e, x2_e, y2_e), # Final view<br> "duration_sec": 4.0-8.0<br> }<br> ```<br>• **AI Crop Proposer v3:** Retrain model to output 8 coordinates (start + end rect) instead of 4<br>• **Desktop UI adaptation:** Your interactive crop tool shows TWO rectangles: green=start, blue=end. User drags both to define motion<br>• **Rule-based fallbacks:** If no motion desired, start_frame == end_frame (static shot)<br>• **New validation:** Ensure motion stays within image bounds throughout interpolation |
| **Tool 3: Compliance/Logging**                              | **Role:** Video-specific compliance checks. Static images have one compliance moment; videos have temporal compliance (motion speed limits, flash-frame detection, epilepsy-safe checks).                                        | **New Requirements:**<br>• **Pre-generation validation:**<br> - Motion speed check: No movement faster than 20% frame width per second (prevents nausea)<br> - Brightness variance: No >30% luminance change between frames (photosensitivity)<br> - Aspect ratio stability: Camera path maintains valid aspect ratios<br>• **Post-generation validation:**<br> - Frame-by-frame scan: Detect flash frames (>50% brightness swing in 1 frame)<br> - Temporal artifact detection: Warping, ghosting, flickering patterns<br> - Audio safety: If audio generation added later, check for sudden volume spikes<br>• **Logging additions to FileTracker:**<br> ```python<br> tracker.track_video_operation(<br> operation='generate                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     | validate | transcode',<br> keyframe_path='image.png',<br> video_output='video.mp4',<br> motion_params={...},<br> validation_results={...},<br> duration_sec=6.0,<br> frame_count=180<br> )<br> ```<br>• **Companion files expansion:** Existing .yaml/.caption + new .video_meta.json with motion params, validation results, generation settings |
| **Tool 4: Overall Workflow**                                | **Role:** Your M4 Pro selection/crop system outputs **structured JSON prompt files** that the colocation GPU machine consumes to generate videos. Decouple selection (Mac) from generation (datacenter).                         | **Architecture:**<br>• **Phase 1 (M4 Pro - Your current system):**<br> 1. Run AI Reviewer: Select best keyframe from group → outputs to `__video_keyframes/`<br> 2. Run Desktop Motion Tool: Define camera movement → outputs .motion.json sidecar<br> 3. Validation: Check motion safety, compliance<br> 4. Package: Create video generation manifest JSON<br>• **Phase 2 (Transfer):**<br> - Sync keyframes + manifests to colocation server via rsync/sftp<br>• **Phase 3 (Colocation GPUs):**<br> - Batch process: Load manifest → generate videos → validate output<br> - Store videos + metadata in structured output directory<br>• **Phase 4 (Transfer back):**<br> - Sync completed videos back to Mac for review<br> - Update companion .video_meta.json with generation results<br><br>**JSON Manifest Format:**<br>`json<br>{<br>  "project_id": "video_gen_001",<br>  "batch_id": "20250704_143022",<br>  "keyframes": [<br>    {<br>      "image_path": "__video_keyframes/img_001.png",<br>      "motion": {<br>        "type": "zoom_in",<br>        "start_rect": [0.1, 0.1, 0.9, 0.9],<br>        "end_rect": [0.3, 0.3, 0.7, 0.7],<br>        "duration_sec": 6.0,<br>        "easing": "ease_in_out"<br>      },<br>      "generation_params": {<br>        "model": "stable_video_diffusion_v1.1",<br>        "motion_bucket_id": 127,<br>        "fps": 30,<br>        "noise_aug_strength": 0.02,<br>        "seed": 42<br>      }<br>    }<br>  ]<br>}<br>` |

---

## 🏗️ PART 2: VIDEO GENERATION BLUEPRINT

### 1. **Video Generation Model Strategy**

**Recommended Approach:** **Model Shootout (Wan-AI / HunyuanVideo / Mochi 1)**

**Top Contenders for Benchmarking:**

1. **Wan-AI 2.2 I2V 14B** - Mixture-of-Experts architecture optimized for throughput
2. **HunyuanVideo 13B** - Known for cinematic quality and temporal coherence
3. **Mochi 1 10B** - Apache 2.0 license, optimized for 4090 GPUs, fast inference

**Why These Models:**

- ✅ **Single keyframe input:** Perfect fit for your workflow (1 selected image → 1 video)
- ✅ **Temporal consistency:** All three specifically trained to avoid flickering/warping
- ✅ **Open weights:** Can deploy on your colocation GPUs without API costs
- ✅ **Production-ready:** Used by commercial services (not experimental)
- ✅ **4090-compatible:** All can run on RTX 4090 (24GB VRAM)

**Selection Process:**

Before committing to hardware, run the **Three-Model Shootout** (see separate document) to determine which model offers the best balance of quality and speed for your specific use case.

**Deprecated Options:**

- ~~Stable Video Diffusion (SVD)~~ - Older generation, superseded by above models
- **AnimateDiff:** More flexible but requires prompt engineering (overkill for your use case)
- **Zeroscope:** Faster but lower quality
- **Gen-2/Pika API:** Expensive at scale ($0.05-0.10 per 4sec video)

**Hardware Requirements (Post-Shootout):**

- **Minimum:** 1x RTX 4090 (24GB VRAM) → Estimated 10-20 sec per 4sec video @ 576x1024
- **Optimal:** 2x RTX 4090 → parallel generation, target 8-12 videos/minute
- **Scaling:** Each additional GPU increases throughput linearly
- **Note:** Final specifications depend on shootout winner's actual performance

---

### 2. **Prompt Engineering Pivot**

Your current image prompts are visual descriptors. Video generation requires **temporal + motion parameters**.

#### **New Critical Parameters:**

| Parameter                                | Purpose                                 | Your Implementation                                              |
| ---------------------------------------- | --------------------------------------- | ---------------------------------------------------------------- |
| `motion_bucket_id` (or model equivalent) | Controls motion intensity               | Map from `motion_intensity` to the model's control parameter     |
| `fps`                                    | Frame rate (24/30/60)                   | Default 30fps for smooth playback                                |
| `num_frames`                             | Base frames generated (model-dependent) | 4sec @ 30fps = 120 frames; generate base frames then interpolate |
| `noise_aug_strength`                     | Prevents flickering (0.0-0.1 range)     | Start at 0.02, tune per content type                             |
| `conditioning_frames`                    | Number of keyframes (usually 1)         | Always 1 (your selected image)                                   |
| `seed`                                   | Reproducibility                         | Store in manifest for regeneration                               |

#### **Camera Movement Translation:**

Your crop motion types → Video generation motion parameters (model-specific):

```python

def motion_to_model_params(motion_descriptor):

    motion_map = {

        "static": {"motion_bucket_id": 10, "camera_motion": "static"},

        "zoom_in": {"motion_bucket_id": 127, "camera_motion": "zoom_in"},

        "zoom_out": {"motion_bucket_id": 127, "camera_motion": "zoom_out"},

        "pan_left": {"motion_bucket_id": 100, "camera_motion": "pan_left"},

        "pan_right": {"motion_bucket_id": 100, "camera_motion": "pan_right"},

    }



    base_params = motion_map[motion_descriptor["motion_type"]]



    # Adjust motion intensity

    intensity_scale = motion_descriptor["motion_intensity"]  # 0.0-1.0

    base_params["motion_bucket_id"] = int(base_params["motion_bucket_id"] * intensity_scale)



    # Add your crop coordinates as spatial conditioning

    base_params["crop_coords"] = {

        "start": motion_descriptor["start_frame"],

        "end": motion_descriptor["end_frame"]

    }



    return base_params

```

#### **Temporal Continuity Settings:**

**Critical:** These prevent the "seizure effect":

```python

generation_config = {

    "decode_chunk_size": 8,  # Process 8 frames at a time (VRAM vs quality)

    "noise_aug_strength": 0.02,  # Low noise = more stability

    "num_inference_steps": 25,  # More steps = smoother (diminishing returns after 25)

    "min_guidance_scale": 1.0,  # Prevent over-stylization

    "max_guidance_scale": 3.0,  # Balance creativity vs stability

}

```

---

### 3. **Post-Processing Necessity**

The **single most critical** post-processing step:

#### **🎯 Frame Interpolation (REQUIRED)**

**Why:** Base I2V models typically output tens of frames per pass. Interpolate to reach 24–30fps for 4–6s clips.

**Solution:** **FILM (Frame Interpolation for Large Motion)** or **RIFE**

```python

# Generate base frames

base_frames = generate_video(keyframe, motion_params)

# Interpolate to target FPS

interpolated = interpolate_frames(

    base_frames,

    target_fps=30,

    target_duration_sec=4.0,

    method="FILM"  # or "RIFE"

)  # Output: 120 frames @ 30fps

```

**Performance:**

- FILM: Slower but smoother (2-3 sec per frame pair on RTX 4090) - CPU-based
- RIFE: Faster with GPU acceleration (0.1-0.3 sec per frame pair on RTX 4090)
- **Recommendation:** GPU-accelerated RIFE for commercial throughput (run on GPU 2 while GPU 1 generates)

#### **Secondary Post-Processing (Optional but Recommended):**

1. **Temporal Stabilization:** Remove jitter/shake
   - Tool: `ffmpeg -i input.mp4 -vf deshake output.mp4`
2. **Color Grading:** Ensure consistent look across batch
   - Tool: Apply LUT (Look-Up Table) via ffmpeg
3. **Compression:** Reduce file size without quality loss
   - Codec: H.265 (HEVC) for 40% smaller files vs H.264
   - Command: `ffmpeg -i raw.mp4 -c:v libx265 -crf 23 -preset medium final.mp4`

---

### 4. **Hardware Utilization Strategy**

#### **Option A: Parallel Keyframe Selection (Mac) + Serial Video Generation (Colocation)**

**Best for:** High-volume batches (1000+ videos)

```

┌─────────────────────┐

│ M4 Pro Mac          │

│ - AI Selection      │ ← Fast (100 groups/min)

│ - Motion Definition │

│ - Manifest Creation │

└──────────┬──────────┘

           │ rsync keyframes + manifests

           ↓

┌─────────────────────┐

│ Colocation Server   │

│ 2x RTX 4090         │

│ - GPU 1: Generate   │ ← 4-6 videos/min per GPU

│ - GPU 2: RIFE Interp│ ← GPU-accelerated frame interpolation

│ - CPU: Compression  │ ← H.265 encoding in parallel (or NVENC where available)

└─────────────────────┘

```

**Throughput:**

- Selection: 100 keyframes/min (Mac)
- Generation: 8-12 videos/min (2x GPU colocation)
- **Bottleneck:** Video generation (expected)

#### **Option B: Unified Pipeline on Colocation**

**Best for:** Lower volume, fully automated

Move entire workflow to colocation:

- Selection AI runs on GPU 1
- Video generation on GPU 2
- No transfer overhead

**Tradeoff:** Lose local review capability, but faster E2E

#### **Recommendation: Start with Option A**

Why:

- Leverage your existing Mac setup
- Keep selection/QA loop local (fast iteration)
- Only send finalized keyframes to expensive GPU time
- Easy to scale GPU count later (3x, 4x RTX 4090s)

---

## 🔧 PART 3: IMPLEMENTATION ROADMAP

### **Phase 0: Model Selection (Week 1)**

1. **Run Three-Model Shootout:**
   - Set up ComfyUI on cloud GPU (see setup-local-ai.md)
   - Benchmark Wan-AI, HunyuanVideo, and Mochi 1
   - Select winning model based on quality + speed
   - Document winning model's specific parameters

### **Phase 1: Prototype (Week 2-3)**

1. **Extend AI Reviewer:**
   - Add "Video Mode" flag
   - Output to `__video_keyframes/` instead of `__selected/`
   - Create `.motion.json` sidecar with static motion (all motion_type: "static")
2. **Test Winning Model Locally:**
   - Install winning model on Mac (will be slow, just for validation)
   - Generate 10 test videos from your best images
   - Verify temporal consistency meets commercial standards
3. **Build Manifest Generator:**
   - Script: `scripts/tools/create_video_manifest.py` (TBD)
   - Input: `__video_keyframes/` directory
   - Output: `video_generation_manifest.json`
   - Include winning model's specific parameters

### **Phase 2: Motion System (Week 3-4)**

1. **Desktop Motion Tool:**
   - Fork `scripts/02_ai_desktop_multi_crop.py` → `scripts/02_ai_desktop_motion_tool.py` (TBD)
   - Display TWO crop rectangles (start=green, end=blue)
   - User drags both to define camera path
   - Output: `.motion.json` with start/end coordinates
2. **Motion Validation:**
   - Script: `scripts/utils/validate_motion.py` (TBD)
   - Check speed limits, aspect ratio stability
   - Flag dangerous movements
3. **AI Motion Proposer (Optional):**
   - Extend Crop Proposer v2 to suggest motion
   - Training data: Label 200 images with ideal camera movements
   - Model predicts motion_type + intensity

### **Phase 3: Colocation Setup (Week 5-6)**

1. **Server Provisioning:**

   - Install Ubuntu 22.04 LTS
   - NVIDIA drivers + CUDA 12.1
   - Docker for isolation

2. **Video Generation Service:**

   - Docker container with winning model + GPU-accelerated RIFE
   - REST API: POST manifest → returns video URLs (TBD service)
   - Background worker: Processes queue
   - GPU 1: Video generation, GPU 2: Frame interpolation

3. **Transfer Pipeline:**

   - Script: `scripts/tools/sync_to_colocation.py` (TBD)
   - rsync keyframes + manifests
   - Monitor generation progress
   - Download completed videos

### **Phase 4: Post-Processing (Week 7)**

1. **Frame Interpolation:**
   - Integrate GPU-accelerated RIFE
   - Run on dedicated GPU (GPU 2) in parallel with generation
2. **Quality Validation:**
   - Flash frame detection
   - Temporal artifact scanning
   - Automatic flagging for human review
3. **Compression Pipeline:**
   - H.265 encoding
   - Consistent bitrate across batch

### **Phase 5: Production (Week 8+)**

1. **Dashboard Integration:**
   - Add video generation metrics to your existing dashboard
   - Track: keyframes selected, videos generated, GB processed
2. **Batch Automation:**
   - Cron job: Auto-sync new keyframes every hour
   - Monitor colocation GPU utilization
   - Alert on failures
3. **Client Delivery:**
   - Extend `07_finish_project.py` to include videos
   - ZIP structure: `videos/`, `keyframes/`, `metadata/`

---

## 📋 PART 4: COMPANION FILE SYSTEM FOR VIDEO

Your existing companion file system maps perfectly:

### **Existing (Images):**

```

image_001.png

image_001.yaml       ← Metadata

image_001.caption    ← Caption

```

### **New (Videos):**

```

image_001.png                    ← Keyframe (original)

image_001.yaml                   ← Original metadata

image_001.caption                ← Original caption

image_001.motion.json            ← Motion parameters (Phase 2)

image_001.video_manifest.json    ← Generation settings (Phase 3)

image_001.mp4                    ← Generated video (Phase 3)

image_001.video_meta.json        ← Post-generation validation (Phase 4)

```

**Extend your companion file utilities:**

```python

# scripts/utils/companion_file_utils.py

VIDEO_COMPANION_EXTENSIONS = {

    ".motion.json",      # Motion definition

    ".video_manifest.json",  # Generation params

    ".mp4", ".mov",      # Video files

    ".video_meta.json"   # Post-gen metadata

}

def find_all_companions_including_video(image_path: Path) -> list[Path]:

    companions = find_all_companion_files(image_path)  # Your existing function



    # Add video companions

    stem = image_path.stem

    for ext in VIDEO_COMPANION_EXTENSIONS:

        companion = image_path.parent / f"{stem}{ext}"

        if companion.exists():

            companions.append(companion)



    return companions

```

---

### 🔒 File Safety Alignment

- New artifacts (`.motion.json`, `.video_manifest.json`, `.video_meta.json`, `.mp4`) must be CREATED ONLY in safe zones such as `sandbox/`, `data/`, or temporary worker output directories. Do not write into production image folders (`mojo*/`, `__crop/`, `__cropped/`, `__selected/`).
- Moves must always include companions. Use existing companion utilities and extend them to include video companions.
- All file operations must be logged via `FileTracker` per `Documents/safety/FILE_SAFETY_SYSTEM.md`.

## 🎯 SUCCESS METRICS

Track these to measure video system performance:

| Metric                       | Target                                       | Measurement                           |
| ---------------------------- | -------------------------------------------- | ------------------------------------- |
| **Keyframe Selection Speed** | 100+ groups/min                              | Already hitting this on Mac           |
| **Video Generation Speed**   | 6-8 videos/min per GPU                       | Benchmark winning model on colocation |
| **Temporal Consistency**     | <5% videos flagged for artifacts             | Automated validation                  |
| **Client Acceptance Rate**   | >90% videos approved first pass              | Track revisions                       |
| **Cost per Video**           | <$0.02 (assuming $1.50/hr GPU, 8 videos/min) | Monitor GPU utilization               |
| **Throughput**               | 1000 videos/day sustainable                  | Run 24hr batch test                   |

---

## ⚠️ CRITICAL RISKS & MITIGATIONS

| Risk                             | Impact            | Mitigation                                                                    |
| -------------------------------- | ----------------- | ----------------------------------------------------------------------------- |
| **Models generate artifacts**    | Unusable videos   | Pre-filter keyframes with video_viability score; reject low scorers           |
| **Colocation network lag**       | Slow transfers    | Upload keyframes in batches; use compression; consider local GPU rental first |
| **Frame interpolation too slow** | Bottleneck        | Use GPU-accelerated RIFE on dedicated GPU; fall back to FILM on CPU if needed |
| **Motion makes people nauseous** | Client complaints | Strict motion speed validation; conservative defaults; user testing           |
| **File management chaos**        | Lost work         | Extend FileTracker to log video operations; rigorous companion file handling  |

---

## 💰 ESTIMATED COSTS

### **Colocation GPU Rental:**

- **Provider:** vast.ai, runpod.io, or lambda labs
- **Config:** 2x RTX 4090 (48GB total VRAM)
- **Cost:** $1.20-1.80/hr
- **Monthly:** ~$900-1300 (24/7 operation)
- **Per video:** ~$0.015-0.025 (at 8 videos/min)

### **Bandwidth:**

- **Upload:** 2MB per keyframe × 1000 = 2GB/day
- **Download:** 50MB per video × 1000 = 50GB/day
- **Total:** ~1.5TB/month
- **Cost:** $0-30/month (depends on colocation plan)

### **Storage:**

- **Videos:** 50MB × 30K videos = 1.5TB
- **Cost:** $15-30/month (S3 or Backblaze B2)

**Total Monthly:** ~$950-1400 for high-volume operation (1000 videos/day)

---

## 🚀 QUICK START COMMAND

When you're ready to test:

```bash

# Phase 0: Model Shootout (Run on cloud GPU first)

# See: Documents/video_creation/Three-Model Video Generation Shootout Plan.md
# See: Documents/video_creation/setup-local-ai.md

# Phase 1 Prototype Test (After selecting winning model)

cd /Users/eriksjaastad/projects/image-workflow

source .venv311/bin/activate

# 1. Select keyframes (your existing workflow, video mode flag added)

python scripts/01_ai_assisted_reviewer.py sandbox/test_video_batch \

  --video-mode \

  --batch-size 10

# 2. Generate video manifest (TBD script – placeholder)

python scripts/tools/create_video_manifest.py \

  --keyframes __video_keyframes \

  --output video_manifest.json \

  --model <WINNING_MODEL>

# 3. Test local generation with winning model (slow on Mac, just validation) (TBD script – placeholder)

python scripts/tools/generate_videos_local.py \

  --manifest video_manifest.json \

  --output sandbox/test_videos

# 4. Review output

open sandbox/test_videos/

```

---

## ❓ DECISION GATES

Before each phase:

### **Before Phase 0 (Model Shootout):**

1. **Cloud GPU budget approved?** $50-150 for 8-12 hours of testing
2. **Test keyframe selected?** High-quality image representative of your typical content

### **Before Phase 1 (Prototype):**

1. **Winning model selected?** Based on shootout results
2. **Performance targets met?** Quality and speed acceptable for commercial use

### **Before Phase 3 (Colocation Setup):**

1. **Target video volume confirmed?** 100/day? 1000/day? (Determines GPU count)
2. **Colocation access secured?** Provider selected, AUP reviewed
3. **Docker/Linux comfort level assessed?** Determines automation complexity
4. **Motion definition approach decided?** Manual, AI-suggested, or fully automated
5. **Audio requirements clarified?** Music, voiceover, or no audio

### **Before Phase 5 (Production):**

1. **Pilot batch completed successfully?** 100-500 videos delivered and accepted
2. **Cost per video validated?** Actual costs match projections
3. **Client acceptance rate measured?** >90% approval on first delivery
