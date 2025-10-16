#!/usr/bin/env python3
import argparse
import os
import tarfile
import time
import shutil
from pathlib import Path


EXCLUDE_DIRS = {"metrics", "reports", "runs", "logs"}


def _should_exclude(relative_path: Path) -> bool:
    # Exclude any path that contains an excluded top-level directory segment
    return any(part in EXCLUDE_DIRS for part in relative_path.parts)


def save_snapshot(root: Path, out_path: Path) -> None:
    root = root.resolve()
    out_path = out_path.resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Create tar archive excluding EXCLUDE_DIRS
    with tarfile.open(out_path, mode="w") as tar:
        for dirpath, dirnames, filenames in os.walk(root):
            rel_dir = Path(dirpath).resolve().relative_to(root)
            # Prune excluded directories from traversal
            dirnames[:] = [d for d in dirnames if (Path(rel_dir, d).parts and not _should_exclude(Path(rel_dir, d)))]

            for filename in filenames:
                rel_file = Path(rel_dir, filename)
                if _should_exclude(rel_file):
                    continue
                abs_file = root / rel_file
                # Skip the output archive itself if it lives under root
                if abs_file == out_path:
                    continue
                tar.add(abs_file, arcname=str(rel_file))

    print(f"Saved snapshot of {root} to {out_path}")
    print(f"Excluded directories: {', '.join(sorted(EXCLUDE_DIRS))}")


def restore_snapshot(root: Path, in_path: Path) -> None:
    root = root.resolve()
    in_path = in_path.resolve()
    if not in_path.exists():
        raise FileNotFoundError(f"Snapshot not found: {in_path}")

    # Ensure root exists
    root.mkdir(parents=True, exist_ok=True)

    # Remove everything under root except excluded dirs and the snapshot file itself
    for entry in root.iterdir():
        name = entry.name
        if name in EXCLUDE_DIRS:
            # Keep excluded dirs as-is
            continue
        if entry.resolve() == in_path:
            continue
        try:
            if entry.is_symlink() or entry.is_file():
                entry.unlink(missing_ok=True)
            elif entry.is_dir():
                shutil.rmtree(entry)
        except Exception as e:
            print(f"Warning: failed to remove {entry}: {e}")

    with tarfile.open(in_path, mode="r") as tar:
        tar.extractall(path=root)

    print(f"Restored snapshot from {in_path} into {root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sandbox snapshot save/restore tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_save = subparsers.add_parser("save", help="Save a snapshot to a tar archive")
    p_save.add_argument("--root", required=True, help="Root directory to snapshot")
    p_save.add_argument("--out", required=True, help="Output tar path (e.g., .baseline.tar)")

    p_restore = subparsers.add_parser("restore", help="Restore from a tar archive snapshot")
    p_restore.add_argument("--root", required=True, help="Root directory to restore into")
    p_restore.add_argument("--in", dest="in_path", required=True, help="Input tar path to restore from")

    args = parser.parse_args()

    if args.command == "save":
        save_snapshot(Path(args.root), Path(args.out))
    elif args.command == "restore":
        restore_snapshot(Path(args.root), Path(args.in_path))
    else:
        parser.error("Unknown command")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import argparse
import os
import sys
import tarfile
from pathlib import Path


def _assert_under_root(path: Path, root: Path) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except Exception:
        raise RuntimeError(f"Refusing to operate outside sandbox root: {path}")


def save_baseline(root: Path, out_tar: Path) -> None:
    root = root.resolve()
    _assert_under_root(root, root)
    out_tar = out_tar.resolve()
    if not root.exists():
        raise FileNotFoundError(f"Sandbox root not found: {root}")

    exclude = {"metrics", "reports", "runs", "logs"}
    # Ensure parent exists
    out_tar.parent.mkdir(parents=True, exist_ok=True)

    # Create tar at root/.baseline.tar
    with tarfile.open(out_tar, "w:gz") as tf:
        for p in root.iterdir():
            if p.name in exclude:
                continue
            if p.name == out_tar.name:
                continue
            tf.add(p, arcname=p.name)

    print(f"Saved baseline: {out_tar}")


def _safe_remove_tree(target: Path, preserve: set) -> None:
    for entry in target.iterdir():
        if entry.name in preserve:
            continue
        if entry.is_dir():
            # Recursively remove directory
            for sub in sorted(entry.rglob("*"), reverse=True):
                try:
                    if sub.is_file() or sub.is_symlink():
                        sub.unlink(missing_ok=True)
                    elif sub.is_dir():
                        sub.rmdir()
                except Exception:
                    pass
            try:
                entry.rmdir()
            except Exception:
                pass
        else:
            try:
                entry.unlink(missing_ok=True)
            except Exception:
                pass


def restore_baseline(root: Path, in_tar: Path) -> None:
    root = root.resolve()
    _assert_under_root(root, root)
    in_tar = in_tar.resolve()
    if not in_tar.exists():
        raise FileNotFoundError(f"Baseline tar not found: {in_tar}")

    preserve = {"metrics", "reports", "runs", "logs", in_tar.name}
    _safe_remove_tree(root, preserve)

    with tarfile.open(in_tar, "r:gz") as tf:
        tf.extractall(root)

    print(f"Restored baseline from: {in_tar}")


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="snapshot", description="Sandbox snapshot save/restore")
    sub = p.add_subparsers(dest="command", required=True)

    sp_save = sub.add_parser("save", help="Create baseline tar")
    sp_save.add_argument("--root", required=True)
    sp_save.add_argument("--out", required=True)

    sp_restore = sub.add_parser("restore", help="Restore from baseline tar")
    sp_restore.add_argument("--root", required=True)
    sp_restore.add_argument("--in", dest="in_file", required=True)

    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "save":
        save_baseline(Path(args.root), Path(args.out))
        return 0
    if args.command == "restore":
        restore_baseline(Path(args.root), Path(args.in_file))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())


