#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

# Ensure project root on sys.path when invoked directly
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import concurrent.futures
from functools import lru_cache
from typing import Tuple

from scripts.tools.prof import Profiler
from scripts.utils.companion_file_utils import (
    detect_stage,
    find_consecutive_stage_groups,
    move_file_with_all_companions,
    sort_image_files_by_timestamp_and_stage,
)
from scripts.utils.watchdog import Heartbeat, Watchdog, print_progress

# Minimal scaffold; actual logic will live in sibling modules

DEFAULTS = {
    "sandbox_root": "sandbox/mojo2",
    "images_glob": "**/*.png",
    "group_window": "5m",
    "dedupe": 0.90,
    "sharpness_metric": "lapvar",
    "sharpness_quantile": 0.95,
    "border_band": 1.0,
    "exposure_band": 2.0,
    "sample_size": 200,
    "seed": 1337,
}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="reducer",
        description="Sandbox-only Automation Reduction Experiments CLI",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--sandbox-root", default=DEFAULTS["sandbox_root"]) 
    common.add_argument("--images-glob", default=DEFAULTS["images_glob"]) 
    # Accept --time-window as an alias for --group-window
    common.add_argument("--group-window", "--time-window", dest="group_window", choices=["2m","5m","10m"], default=DEFAULTS["group_window"]) 
    # Accept --dedupe-threshold as an alias for --dedupe
    common.add_argument("--dedupe", "--dedupe-threshold", dest="dedupe", type=float, choices=[0.90,0.85,0.80], default=DEFAULTS["dedupe"]) 
    common.add_argument("--sharpness-metric", choices=["lapvar","tenengrad"], default=DEFAULTS["sharpness_metric"]) 
    common.add_argument("--sharpness-quantile", type=float, choices=[0.95,0.90,0.85,0.80], default=DEFAULTS["sharpness_quantile"]) 
    common.add_argument("--border-band", type=float, choices=[1.0,2.5,5.0], default=DEFAULTS["border_band"]) 
    common.add_argument("--exposure-band", type=float, choices=[2.0,2.5,3.0], default=DEFAULTS["exposure_band"]) 
    common.add_argument("--seed", type=int, default=DEFAULTS["seed"]) 
    # watchdog & progress flags
    common.add_argument("--max-runtime", type=float, default=900.0, help="Max wall time (seconds) before abort")
    common.add_argument("--stage-timeout", type=float, default=120.0, help="Per-phase timeout (seconds)")
    common.add_argument("--progress-interval", type=float, default=10.0, help="Seconds between progress prints")
    common.add_argument("--watchdog-threshold", type=float, default=120.0, help="No-progress stall threshold (seconds)")
    common.add_argument("--no-stack-dump", action="store_true", help="Disable stack dump on abort")

    # plan
    p_plan = sub.add_parser("plan", parents=[common], help="Create an execution plan")
    p_plan.add_argument("--variant", choices=["A","B","C"], required=True)
    p_plan.add_argument("--profile", choices=["conservative","balanced","aggressive"], required=True)
    p_plan.add_argument("--limit", type=int, default=None)

    # run
    p_run = sub.add_parser("run", parents=[common], help="Execute a plan (default dry-run)")
    p_run.add_argument("--variant", choices=["A","B","C"], required=True)
    p_run.add_argument("--profile", choices=["conservative","balanced","aggressive"], required=True)
    mode = p_run.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", default=True)
    mode.add_argument("--commit", action="store_true", default=False)
    p_run.add_argument("--limit", type=int, default=None)
    # testing hook to simulate hang
    p_run.add_argument("--simulate-hang", action="store_true", default=False)
    p_run.add_argument("--quiet", action="store_true", default=False, help="Reduce per-file logs")
    # investigation & profiling
    p_run.add_argument("--investigate", action="store_true", default=False, help="Enable profiling and periodic checkpoints")
    p_run.add_argument("--checkpoint-interval", type=float, default=300.0, help="Seconds between investigation checkpoints (JSONL)")
    # deterministic sharding (by group)
    p_run.add_argument("--shards", type=int, default=1, help="Total number of shards (groups are assigned deterministically)")
    p_run.add_argument("--shard-index", type=int, default=0, help="Shard index [0..shards-1]")
    # companion lookup optimization toggle
    p_run.add_argument("--use-stem-index", action="store_true", default=False, help="Use stem-indexed companion cache within directories")
    # async moves queue (write-behind) toggle
    p_run.add_argument("--async-moves", action="store_true", default=False, help="Enable async write-behind queue for file moves")
    p_run.add_argument("--move-workers", type=int, default=4, help="Number of worker threads for async moves")
    p_run.add_argument("--move-queue-size", type=int, default=128, help="Max outstanding move operations before waiting")
    # quality-aware selection
    p_run.add_argument("--quality-aware", action="store_true", default=False, help="Enable defect-aware selection within groups (fallback to lower stage if top is poor)")
    p_run.add_argument("--qa-thumb-size", type=int, default=256, help="Thumbnail max side for QA metrics")
    p_run.add_argument("--qa-clip-threshold", type=float, default=0.02, help="Exposure clipping threshold (fraction of pixels at 0 or 255)")

    # report
    p_rep = sub.add_parser("report", help="Emit markdown summary for a prior run")
    p_rep.add_argument("--run-id", required=True)
    p_rep.add_argument("--markdown", action="store_true", default=True)

    return parser.parse_args(argv)


def print_summary_panel(args):
    cfg = {
        "command": args.command,
        "variant": getattr(args, "variant", None),
        "profile": getattr(args, "profile", None),
        "sandbox_root": args.sandbox_root,
        "images_glob": args.images_glob,
        "group_window": args.group_window,
        "dedupe": args.dedupe,
        "sharpness_metric": args.sharpness_metric,
        "sharpness_quantile": args.sharpness_quantile,
        "border_band": args.border_band,
        "exposure_band": args.exposure_band,
        "mode": "commit" if getattr(args, "commit", False) else "dry-run",
        "seed": args.seed,
        "limit": getattr(args, "limit", None),
        "max_runtime": getattr(args, "max_runtime", None),
        "stage_timeout": getattr(args, "stage_timeout", None),
        "progress_interval": getattr(args, "progress_interval", None),
        "watchdog_threshold": getattr(args, "watchdog_threshold", None),
        "investigate": getattr(args, "investigate", False),
        "checkpoint_interval": getattr(args, "checkpoint_interval", None),
        "shards": getattr(args, "shards", 1),
        "shard_index": getattr(args, "shard_index", 0),
        "use_stem_index": getattr(args, "use_stem_index", False),
        "async_moves": getattr(args, "async_moves", False),
        "move_workers": getattr(args, "move_workers", None),
        "move_queue_size": getattr(args, "move_queue_size", None),
        "quality_aware": getattr(args, "quality_aware", False),
        "qa_thumb_size": getattr(args, "qa_thumb_size", None),
        "qa_clip_threshold": getattr(args, "qa_clip_threshold", None),
    }
    print(json.dumps(cfg, indent=2))


def ensure_sandbox_dirs(root: Path, run_id: str):
    (root / "metrics").mkdir(parents=True, exist_ok=True)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "runs" / run_id).mkdir(parents=True, exist_ok=True)


def cmd_plan(args):
    print_summary_panel(args)
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + f"_{args.variant}-{args.profile}"
    root = Path(args.sandbox_root)
    ensure_sandbox_dirs(root, run_id)
    manifest = {
        "run_id": run_id,
        "created": datetime.utcnow().isoformat() + "Z",
        "variant": args.variant,
        "profile": args.profile,
        "settings": {
            "group_window": args.group_window,
            "dedupe": args.dedupe,
            "sharpness_metric": args.sharpness_metric,
            "sharpness_quantile": args.sharpness_quantile,
            "border_band": args.border_band,
            "exposure_band": args.exposure_band,
        },
    }
    (root / "runs" / run_id / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Plan created: {run_id}")


def cmd_run(args):
    print_summary_panel(args)
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + f"_{args.variant}-{args.profile}"
    root = Path(args.sandbox_root)
    ensure_sandbox_dirs(root, run_id)
    # Setup heartbeat & watchdog
    hb = Heartbeat()
    abort_reason = {"reason": None}

    def on_abort(reason: str):
        abort_reason["reason"] = reason
        print(f"ABORT {run_id} reason={reason}", flush=True)
        print(f"ABORT {run_id} reason={reason}", file=sys.stderr, flush=True)

    wd = Watchdog(
        hb,
        start_time_utc=time.time(),
        max_runtime_sec=float(args.max_runtime),
        stall_threshold_sec=float(args.watchdog_threshold),
        poll_interval_sec=max(0.2, float(args.progress_interval) / 2.0),
        on_abort=on_abort,
        sandbox_root=root,
        run_id=run_id,
        write_stack=not bool(args.no_stack_dump),
    )
    wd.start()
    stop_event = None
    profiler = None
    try:
        import threading
        stop_event = threading.Event()
        printer = print_progress("[progress]", hb, interval_sec=float(args.progress_interval), stop_event=stop_event)

        if args.investigate:
            profiler = Profiler(Path(args.sandbox_root), run_id, checkpoint_interval_sec=float(args.checkpoint_interval))
            profiler.start_checkpoints(hb)

        # Apply optional companion optimization flag via environment
        if getattr(args, "use_stem_index", False):
            os.environ["COMPANION_USE_STEM_INDEX"] = "1"

        if args.simulate_hang:
            start_ts = time.time()
            while abort_reason["reason"] is None and time.time() - start_ts < (float(args.max_runtime) + 5):
                time.sleep(0.2)
        else:
            # Selection-and-move using centralized grouping utilities
            exclude_dirs = {"metrics", "reports", "runs", "logs", "selected", "delete", "crop"}
            # scan stage
            if profiler:
                profiler.start_stage("scan")
            image_files = [p for p in root.rglob("*.png") if not any(part in exclude_dirs for part in p.parts)]
            image_files = sort_image_files_by_timestamp_and_stage(image_files)
            if profiler:
                profiler.set_counter("n_files", len(image_files))
                profiler.end_stage("scan")
            hb.update(files_scanned=len(image_files), notes="scan")

            # group stage
            if profiler:
                profiler.start_stage("group")
            groups = find_consecutive_stage_groups(image_files)
            if profiler:
                profiler.set_counter("n_groups_total", len(groups))
                profiler.end_stage("group")
            hb.update(groups_built=len(groups), notes="group")

            group_iter = groups
            # deterministic sharding by group's first file path
            if int(getattr(args, "shards", 1)) > 1:
                import hashlib
                shard_k = int(args.shards)
                shard_i = int(args.shard_index)
                shard_i = max(0, min(shard_i, shard_k - 1))
                def group_hash(g):
                    if not g:
                        return 0
                    h = hashlib.sha1(str(g[0]).encode("utf-8")).hexdigest()
                    return int(h[:8], 16)
                group_iter = [g for g in groups if (group_hash(g) % shard_k) == shard_i]
            if profiler:
                profiler.set_counter("n_groups_shard", len(group_iter))
                avg_sz = (sum(len(g) for g in group_iter) / len(group_iter)) if group_iter else 0.0
                profiler.set_counter("avg_group_size", avg_sz)
            if args.limit:
                group_iter = group_iter[: int(args.limit)]

            # All outputs must remain under the sandbox root
            dst_selected = root / "selected"
            dst_delete = root / "delete"
            dst_crop = root / "crop"
            if args.commit:
                dst_selected.mkdir(parents=True, exist_ok=True)
                dst_delete.mkdir(parents=True, exist_ok=True)
                dst_crop.mkdir(parents=True, exist_ok=True)

            winners = 0
            total = 0
            # Reduce verbosity of per-file moves via env var for companion utils
            if args.quiet:
                os.environ["COMPANION_UTILS_QUIET"] = "1"
            # Time budget to end gracefully before watchdog aborts
            start_wall = time.time()
            time_budget_sec = float(args.max_runtime)
            # Optional async move executor
            executor = None
            pending = []  # list of futures
            move_queue_cap = int(getattr(args, "move_queue_size", 128))
            use_async_moves = bool(getattr(args, "async_moves", False))

            def _submit_move(path_src, dst_dir):
                # wrap to record duration on worker
                def _task():
                    t0 = time.time()
                    move_file_with_all_companions(path_src, dst_dir, dry_run=not args.commit)
                    return time.time() - t0
                return executor.submit(_task)

            if use_async_moves:
                executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(1, int(getattr(args, "move_workers", 4))))

            if profiler:
                profiler.start_stage("select")

            # ------------------------------ quality metrics helpers ------------------------------
            @lru_cache(maxsize=4096)
            def _qa_metrics_cached(path_str: str, mtime_ns: int, thumb_size: int) -> Tuple[float, float]:
                """Return (sharpness_score, clip_fraction). Pure-Python/Numpy on thumbnail; fail-open to zeros."""
                try:
                    import numpy as _np
                    from PIL import Image
                except Exception:
                    return 0.0, 0.0
                try:
                    with Image.open(path_str) as img:
                        img.thumbnail((thumb_size, thumb_size))
                        g = img.convert("L")
                        a = _np.asarray(g, dtype=_np.float32)
                        # Tenengrad (Sobel magnitude squared)
                        kx = _np.array([[1, 0, -1],[2,0,-2],[1,0,-1]], dtype=_np.float32)
                        ky = _np.array([[1, 2, 1],[0,0,0],[-1,-2,-1]], dtype=_np.float32)
                        # simple valid-convolution via correlate
                        def conv2(x, k):
                            from numpy.lib.stride_tricks import sliding_window_view
                            w = sliding_window_view(x, k.shape)
                            return (w * k).sum(axis=(-1,-2))
                        gx = conv2(a, kx)
                        gy = conv2(a, ky)
                        tenengrad = (gx*gx + gy*gy).mean()
                        # exposure clipping fraction
                        v = a.reshape(-1)
                        zeros = (v <= 1).sum()
                        highs = (v >= 254).sum()
                        clip_frac = float(zeros + highs) / float(v.size if v.size else 1)
                        return float(tenengrad), float(clip_frac)
                except Exception:
                    return 0.0, 0.0

            def pick_quality_aware(winner_path, losers):
                """Return the final chosen path; fallback if winner fails QA thresholds within group."""
                thumb_size = int(getattr(args, "qa_thumb_size", 256))
                clip_thr = float(getattr(args, "qa_clip_threshold", 0.02))
                # compute metrics for all in group
                candidates = [winner_path] + list(losers)
                metrics = []
                for pth in candidates:
                    try:
                        st = pth.stat()
                        s, c = _qa_metrics_cached(str(pth), getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9)), thumb_size)
                    except Exception:
                        s, c = 0.0, 0.0
                    metrics.append((pth, s, c))
                # group median sharpness as baseline
                import numpy as _np
                med_s = float(_np.median([m[1] for m in metrics])) if metrics else 0.0
                def passes(s, c):
                    return (s >= med_s) and (c <= clip_thr)
                # check winner first
                w_s = metrics[0][1] if metrics else 0.0
                w_c = metrics[0][2] if metrics else 1.0
                if passes(w_s, w_c):
                    return winner_path
                # else pick best alternative by stage order among those passing
                passing = [m[0] for m in metrics[1:] if passes(m[1], m[2])]
                if passing:
                    # choose highest stage among passing
                    stages = [(p, detect_stage(p.name)) for p in passing]
                    stages_sorted = sorted(stages, key=lambda it: {"1":1.0,"1.5":1.5,"2":2.0,"3":3.0}.get(it[1], 0.0))
                    return stages_sorted[-1][0]
                # if none pass, keep original winner
                return winner_path

            for idx, group in enumerate(group_iter):
                # group is a list of Paths sorted by stage asc; choose highest stage
                stages = [(p, detect_stage(p.name)) for p in group]
                stages_sorted = sorted(stages, key=lambda it: {"1":1.0,"1.5":1.5,"2":2.0,"3":3.0}.get(it[1], 0.0))
                winner_path, winner_stage = stages_sorted[-1]
                losers = [p for (p, s) in stages_sorted[:-1]]
                if getattr(args, "quality_aware", False):
                    try:
                        chosen = pick_quality_aware(winner_path, losers)
                        winner_path = chosen
                        # recompute losers as all others
                        losers = [p for p, _ in stages if p != winner_path]
                    except Exception:
                        pass
                total += len(group)
                winners += 1

                # time moves cumulatively under 'moves'
                if use_async_moves and executor is not None:
                    pending.append(_submit_move(winner_path, dst_selected))
                    for lp in losers:
                        pending.append(_submit_move(lp, dst_delete))
                    # backpressure when pending exceeds cap
                    if len(pending) >= move_queue_cap:
                        done, not_done = concurrent.futures.wait(pending, timeout=2.0, return_when=concurrent.futures.FIRST_COMPLETED)
                        pending = list(not_done)
                        if profiler and done:
                            for fut in done:
                                try:
                                    profiler.add_duration("moves", float(fut.result()))
                                except Exception:
                                    pass
                        # keep watchdog alive during backpressure waits
                        if (idx + 1) % 10 == 0:
                            hb.update(items_processed=idx + 1, notes="select_async")
                else:
                    _t0 = time.time()
                    move_file_with_all_companions(winner_path, dst_selected, dry_run=not args.commit)
                    for lp in losers:
                        move_file_with_all_companions(lp, dst_delete, dry_run=not args.commit)
                    if profiler:
                        profiler.add_duration("moves", time.time() - _t0)

                if (idx + 1) % 25 == 0:
                    hb.update(items_processed=idx + 1, notes="select")

                # Graceful stop when close to time budget (leave margin for wrap-up)
                elapsed = time.time() - start_wall
                if elapsed >= max(1.0, time_budget_sec - 2.0):
                    hb.update(items_processed=idx + 1, notes="time_budget_stop")
                    break
            # Flush any outstanding async moves
            if use_async_moves and executor is not None:
                try:
                    # Periodically wait with timeout to allow heartbeat updates
                    while pending:
                        done, not_done = concurrent.futures.wait(pending, timeout=2.0, return_when=concurrent.futures.FIRST_COMPLETED)
                        if profiler and done:
                            for fut in done:
                                try:
                                    profiler.add_duration("moves", float(fut.result()))
                                except Exception:
                                    pass
                        pending = list(not_done)
                        hb.update(items_processed=len(group_iter), notes="moves_flush")
                finally:
                    executor.shutdown(wait=True)
            if profiler:
                profiler.end_stage("select")
            hb.update(items_processed=len(group_iter), notes="select_done")

        if abort_reason["reason"] is not None:
            print(f"ABORT {run_id} reason={abort_reason['reason']}", flush=True)
            (root / "runs" / run_id / "manifest.json").write_text(json.dumps({"run_id": run_id, "args": vars(args), "aborted": abort_reason["reason"]}, indent=2))
            print(f"Run aborted: {run_id} reason={abort_reason['reason']}")
            sys.exit(2)

        elapsed_total = time.time() - wd.start_time_utc
        metrics_line = {
            "run_id": run_id,
            "variant": args.variant,
            "profile": args.profile,
            "input_count": int(locals().get("total", 0)),
            "groups": int(locals().get("winners", 0)),
            "winners": int(locals().get("winners", 0)),
            "percent_reduction": (1.0 - (locals().get("winners", 0) / locals().get("total", 1))) if locals().get("total", 0) else 0.0,
            "needs_crop": 0,
            "no_crop": int(locals().get("winners", 0)),
            "runtime_sec": float(elapsed_total),
            "errors": 0,
            "settings": {
                "group_window": args.group_window,
                "dedupe": args.dedupe,
                "sharpness_metric": args.sharpness_metric,
                "sharpness_quantile": args.sharpness_quantile,
                "border_band": args.border_band,
                "exposure_band": args.exposure_band,
            },
            "mode": "commit" if args.commit else "dry-run",
            "timestamp": datetime.utcnow().isoformat() + "Z",
        }
        with (root / "metrics" / "automation_metrics.jsonl").open("a") as f:
            f.write(json.dumps(metrics_line) + "\n")
        (root / "metrics" / f"samples_{run_id}.csv").write_text("run_id,group_id,winner_tp,winner_fp,winner_fn,crop_tp,crop_fp,crop_fn\n")
        (root / "reports" / f"summary_{run_id}.md").write_text(f"# Summary {run_id}\n\nSandbox selection run.\n")
        (root / "runs" / run_id / "manifest.json").write_text(json.dumps({"run_id": run_id, "args": vars(args)}, indent=2))
        (root / "logs" / f"filetracker_{run_id}.log").write_text("")
        if profiler:
            profiler.finish()
            profiler.stop_checkpoints()
            summary_path = profiler.write_summary()
            print(f"Investigation summary: {summary_path}")
        print(f"Run completed: {run_id}")
    finally:
        try:
            if stop_event:
                stop_event.set()
        except Exception:
            pass
        try:
            wd.stop()
        except Exception:
            pass


def cmd_report(args):
    root = Path(DEFAULTS["sandbox_root"])  # report reads from default unless moved
    run_id = args.run_id
    summary = root / "reports" / f"summary_{run_id}.md"
    if summary.exists():
        print(summary.read_text())
    else:
        print(f"Summary not found for run: {run_id}")


def main(argv=None):
    args = parse_args(argv)
    random.seed(args.seed)
    if args.command == "plan":
        return cmd_plan(args)
    if args.command == "run":
        return cmd_run(args)
    if args.command == "report":
        return cmd_report(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())


