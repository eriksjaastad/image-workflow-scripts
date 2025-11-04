# 🎬 Video Generation System Documentation

**Status:** Ready for Phase 0 (Model Shootout)  
**Last Updated:** November 4, 2025

---

## 📚 Document Overview

This directory contains the complete planning documentation for pivoting your image-workflow system to AI video generation.

| Document | Purpose | Read When |
|----------|---------|-----------|
| **[Image-to-Video Generation - System Audit and Blueprint.md](./Image-to-Video%20Generation%20-%20System%20Audit%20and%20Blueprint.md)** | Master technical plan mapping your existing image workflow to video generation | Planning overall strategy and understanding full system architecture |
| **[Three-Model Video Generation Shootout Plan.md](./Three-Model%20Video%20Generation%20Shootout%20Plan.md)** | Step-by-step guide for benchmarking Wan-AI, HunyuanVideo, and Mochi 1 models | Before committing to hardware - run this shootout first |
| **[setup-local-ai.md](./setup-local-ai.md)** | Hands-on ComfyUI setup guide with exact download commands and cloud GPU instructions | When actually running the shootout - practical implementation |
| **[transition-to-video.md](./transition-to-video.md)** | Quality review of the blueprint identifying risks and optimizations | Reference for understanding strategic decisions and tradeoffs |

---

## 🚀 Quick Start Guide

### Step 1: Understand the Strategy (30 minutes)
1. Read **Blueprint** (focus on Executive Summary and Part 1: System Audit Table)
2. Skim **Transition Review** to understand the key decisions

### Step 2: Run the Shootout (8-12 hours)
1. Follow **Shootout Plan** for methodology
2. Use **Setup Local AI** for technical implementation
3. Rent cloud GPU (RunPod/Vast.ai: $50-150 budget)
4. Test all 3 models, record metrics

### Step 3: Select Winner & Update Plans (1-2 hours)
1. Use weighted scoring system from Shootout Plan
2. Update Blueprint Phase 1-5 with winning model's parameters
3. Calculate final hardware needs based on winner's performance

### Step 4: Implement Phase 0-5 (8+ weeks)
1. Follow updated Blueprint implementation roadmap
2. Gate each phase with decision criteria
3. Build infrastructure incrementally

---

## 🎯 Current Status

✅ **Documentation Complete** - All 4 documents updated and internally consistent  
⏳ **Awaiting:** Decision to proceed with Phase 0 (Model Shootout)  
📊 **Budget:** $50-150 for cloud GPU testing before hardware commitment  
🏆 **Goal:** Select winning model, then implement full production system

---

## 🧩 Planned Artifacts (TBD in repo)

The following scripts/services are referenced in the docs as part of the architecture plan but are not yet present in the repository. They will be added during implementation phases after the model shootout:

- `scripts/tools/create_video_manifest.py` — generates the per-batch JSON manifest consumed by GPU workers
- `scripts/utils/validate_motion.py` — validates camera path safety and constraints
- `scripts/tools/generate_videos_local.py` — local validation runner for the winning model (slow on Mac)
- `scripts/tools/sync_to_colocation.py` — rsync/sftp sync of keyframes/manifests and results
- `scripts/02_ai_desktop_motion_tool.py` — fork of `scripts/02_ai_desktop_multi_crop.py` for start/end motion rectangles

Until these exist, treat any command references to them as placeholders.

---

## 🔑 Key Strategic Decisions Made

### 1. Model Selection Approach
- **Decision:** Run competitive shootout (Wan-AI / HunyuanVideo / Mochi 1) BEFORE buying hardware
- **Rationale:** $50-150 testing cost validates $14K+ hardware investment (0.5% risk mitigation)
- **Deprecated:** Stable Video Diffusion (SVD) - superseded by newer models

### 2. Architecture Pattern
- **Decision:** Decouple selection (Mac M4 Pro) from generation (Colocation GPUs)
- **Rationale:** Leverage existing Mac expertise, only pay GPU costs for actual generation
- **Implementation:** JSON manifest as contract between systems

### 3. Frame Interpolation Strategy
- **Decision:** GPU-accelerated RIFE on dedicated GPU (not CPU-based FILM)
- **Rationale:** CPU interpolation would bottleneck throughput at scale
- **Hardware Impact:** Requires 2x GPUs minimum (1 for generation, 1 for interpolation)

### 4. Companion File System
- **Decision:** Extend existing `.yaml`/`.caption` system with video-specific companions
- **Rationale:** Maintains data integrity philosophy, leverages existing infrastructure
- **New Files:** `.motion.json`, `.video_manifest.json`, `.video_meta.json`, `.mp4`

---

## 📊 Success Metrics (Target)

| Metric | Target | Current Status |
|--------|--------|----------------|
| **Keyframe Selection Speed** | 100+ groups/min | ✅ Already achieved on Mac |
| **Video Generation Speed** | 8-12 videos/min (2x GPU) | ⏳ TBD after shootout |
| **Temporal Consistency** | <5% flagged for artifacts | ⏳ TBD after shootout |
| **Cost per Video** | <$0.02 | ⏳ TBD after shootout |
| **Client Acceptance Rate** | >90% first pass | ⏳ TBD after pilot batch |
| **Throughput** | 1000 videos/day | ⏳ Goal for production |

---

## 💰 Cost Breakdown

### Phase 0: Model Shootout
- Cloud GPU rental: $50-150 (8-12 hours @ $1-2/hr)
- Zero hardware commitment
- **ROI:** Validates entire strategy before $14K+ investment

### Production Operation (1000 videos/day)
- GPU rental: $900-1300/month (2x RTX 4090 @ $1.20-1.80/hr, 24/7)
- Bandwidth: $0-30/month (~1.5TB)
- Storage: $15-30/month (1.5TB videos)
- **Total:** ~$950-1400/month
- **Per video:** $0.015-0.025

---

## ⚠️ Critical Risks & Mitigations

| Risk | Mitigation Strategy | Status |
|------|---------------------|--------|
| **Models generate artifacts** | Pre-filter with `video_viability_score` on Mac before GPU generation | ✅ Designed into Blueprint |
| **Frame interpolation bottleneck** | GPU-accelerated RIFE on dedicated hardware | ✅ Designed into Blueprint |
| **Wrong hardware investment** | Shootout validates model BEFORE buying hardware | ✅ Phase 0 gates hardware decision |
| **Colocation complexity** | Start with cloud GPU rental, move to colocation after validation | ✅ Staged approach |
| **File management chaos** | Extend proven companion file system | ✅ Designed into Blueprint |

---

## 🎓 Technical Foundation Leveraged

Your existing image-workflow provides exceptional foundation for video:

| Existing System | Video Translation | Confidence |
|-----------------|-------------------|------------|
| **Selection AI (Ranker v3)** | Keyframe quality scoring | 🟢 High - proven system |
| **Crop Proposer v2** | Camera motion path definition | 🟡 Medium - needs retraining for motion |
| **Companion file management** | Video metadata handling | 🟢 High - extend existing patterns |
| **FileTracker logging** | Video operation audit trail | 🟢 High - add video-specific operations |
| **Desktop UI tools** | Motion definition interface | 🟢 High - fork existing crop tool |
| **Batch processing workflows** | Video generation pipeline | 🟢 High - similar patterns |

---

## 📞 Decision Gates

### Before Phase 0 (Model Shootout):
- [ ] Cloud GPU budget approved ($50-150)
- [ ] Test keyframe selected (high-quality, representative image)
- [ ] 8-12 hours allocated for testing

### Before Phase 1 (Prototype):
- [ ] Winning model selected
- [ ] Performance targets met (quality + speed acceptable)
- [ ] Model-specific parameters documented

### Before Phase 3 (Colocation Setup):
- [ ] Target video volume confirmed (100/day? 1000/day?)
- [ ] Colocation provider selected OR cloud GPU strategy finalized
- [ ] Motion definition approach decided (manual/AI-suggested/automated)
- [ ] Audio requirements clarified

### Before Phase 5 (Production):
- [ ] Pilot batch successful (100-500 videos delivered + accepted)
- [ ] Cost per video validated against projections
- [ ] Client acceptance rate >90%

---

## 🔗 External Dependencies

| Dependency | Purpose | Alternatives |
|------------|---------|--------------|
| **ComfyUI** | Video model inference platform | Direct model integration (more complex) |
| **Hugging Face** | Model weight hosting/downloads | Direct downloads from model repos |
| **RunPod/Vast.ai/Lambda** | Cloud GPU rental | Local GPU purchase (higher upfront cost) |
| **RIFE** | Frame interpolation | FILM (slower but smoother) |
| **FFmpeg** | Video encoding/compression | Hardware encoders (NVENC) |

---

## 📝 Change Log

### 2025-11-04: Documentation Review & Updates
- ✅ Replaced SVD references with Wan-AI/HunyuanVideo/Mochi shootout approach
- ✅ Added Phase 0 (Model Selection) to implementation roadmap
- ✅ Updated frame interpolation strategy (GPU-accelerated RIFE)
- ✅ Added objective metrics to shootout plan (SSIM, flash detection, VRAM tracking)
- ✅ Completed Setup Local AI with exact download commands and cloud GPU instructions
- ✅ Added Docker Compose configuration for consistent environments
- ✅ Updated decision gates with specific thresholds and criteria
- ✅ Created this README for document navigation

---

## 🚦 Next Actions

1. **Review this README** - Ensure strategy aligns with business goals
2. **Approve Phase 0 budget** - $50-150 for model shootout
3. **Select test keyframe** - Representative of typical content
4. **Run shootout** - Follow Setup Local AI guide
5. **Make model decision** - Based on weighted scoring
6. **Begin Phase 1** - Prototype with winning model

---

## 💡 Notes for Future Reference

- **Estimated Implementation Time:** 8-12 weeks from Phase 0 start to production
- **Team Size:** Can be implemented solo with documented guides
- **Reversibility:** Each phase gates the next - can stop after shootout if models don't meet quality bar
- **Scalability:** Linear - each additional GPU adds throughput proportionally

---

**Questions?** Refer to specific documents above, or review the Transition Review for strategic context.

