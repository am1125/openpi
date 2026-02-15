#!/usr/bin/env python3
"""
Convert LIBERO-format HDF5 trajectory files to video.

Reads image observations from HDF5 (data/demo_*/obs/agentview_rgb or agentview_image)
and writes MP4 files, one per demo or a single selected demo.

Usage:
    # All demos from one file → one video per demo in output_dir
    python openpi/examples/libero/hdf5_to_video.py /path/to/demo.hdf5 --output_dir videos/ --all

    # Single demo by index
    python openpi/examples/libero/hdf5_to_video.py /path/to/demo.hdf5 --output_dir videos/ --demo-index 0

    # Directory of HDF5 files, first 3 demos per file
    python openpi/examples/libero/hdf5_to_video.py /path/to/dir/ --output_dir videos/ --all --max-demos-per-file 3
"""

import argparse
import sys
from pathlib import Path

import h5py
import imageio
import numpy as np


def _normalize_image(img: np.ndarray) -> np.ndarray:
    """Convert to uint8 (H, W, 3) for video."""
    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)
    if len(img.shape) == 2:
        img = np.stack([img, img, img], axis=-1)
    elif len(img.shape) == 3:
        if img.shape[0] in (3, 4):
            img = img.transpose(1, 2, 0)
        if img.shape[2] == 4:
            img = img[:, :, :3]
    return img


def get_image_key(obs_group) -> str | None:
    """Return first available main camera key in obs group."""
    for key in ("agentview_rgb", "agentview_image"):
        if key in obs_group:
            return key
    return None


def extract_demo_images(f: h5py.File, demo_name: str, camera: str | None = None) -> np.ndarray | None:
    """Load images for one demo. Returns (N, H, W, 3) uint8 or None if no images."""
    demo_group = f["data"][demo_name]
    obs_group = demo_group["obs"]
    key = camera or get_image_key(obs_group)
    if key is None:
        return None
    if key not in obs_group:
        return None
    images = np.array(obs_group[key][:])
    out = np.array([_normalize_image(img) for img in images])
    return out


def hdf5_to_video(
    hdf5_path: Path,
    output_dir: Path,
    demo_index: int | None = None,
    all_demos: bool = False,
    camera: str | None = None,
    fps: int = 30,
    max_demos_per_file: int | None = None,
    filename: str | None = None,
) -> list[Path]:
    """
    Convert HDF5 demo(s) to MP4.

    Args:
        hdf5_path: Path to .hdf5/.h5 file or directory of such files.
        output_dir: Directory to write videos.
        demo_index: If set, convert only this demo index (per file).
        all_demos: If True, convert every demo (subject to max_demos_per_file).
        camera: Obs key for images (e.g. 'agentview_rgb', 'eye_in_hand_rgb'). Auto if None.
        fps: Frames per second for output video.
        max_demos_per_file: When using --all, limit demos per file (default: no limit).
        filename: For single-demo mode, optional output filename (e.g. 'out.mp4').

    Returns:
        List of created video paths.
    """
    if not all_demos and demo_index is None:
        demo_index = 0

    if hdf5_path.is_file():
        files = [hdf5_path] if hdf5_path.suffix in (".hdf5", ".h5") else []
    else:
        files = sorted(hdf5_path.glob("*.hdf5")) + sorted(hdf5_path.glob("*.h5"))

    if not files:
        raise FileNotFoundError(f"No HDF5 files found at {hdf5_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for file_path in files:
        with h5py.File(file_path, "r") as f:
            if "data" not in f:
                print(f"Skip {file_path.name}: no 'data' group", file=sys.stderr)
                continue
            demos = sorted(f["data"].keys())
            if not demos:
                print(f"Skip {file_path.name}: no demos", file=sys.stderr)
                continue

            if all_demos:
                indices = list(range(len(demos)))
                if max_demos_per_file is not None:
                    indices = indices[: max_demos_per_file]
            else:
                idx = demo_index if demo_index is not None else 0
                if idx < 0 or idx >= len(demos):
                    print(f"Skip {file_path.name}: demo_index {idx} out of range [0, {len(demos)-1}]", file=sys.stderr)
                    continue
                indices = [idx]

            for i in indices:
                demo_name = demos[i]
                images = extract_demo_images(f, demo_name, camera)
                if images is None or len(images) == 0:
                    print(f"Skip {file_path.name} / {demo_name}: no images", file=sys.stderr)
                    continue

                if all_demos or filename is None:
                    out_name = f"{file_path.stem}_{demo_name}.mp4"
                else:
                    out_name = filename if filename.endswith(".mp4") else f"{filename}.mp4"

                out_path = output_dir / out_name
                print(f"Writing {len(images)} frames to {out_path} ({fps} fps)")
                with imageio.get_writer(out_path, fps=fps) as writer:
                    for img in images:
                        writer.append_data(img)
                created.append(out_path)

    return created


def main():
    parser = argparse.ArgumentParser(
        description="Convert LIBERO HDF5 trajectories to video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "hdf5_path",
        type=str,
        help="Path to .hdf5/.h5 file or directory containing HDF5 files",
    )
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        required=True,
        help="Directory to save output video(s)",
    )
    parser.add_argument(
        "--demo-index",
        type=int,
        default=None,
        help="Convert only this demo index (default: 0 if neither --demo-index nor --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Convert all demos (one video per demo)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default=None,
        help="Image key (e.g. agentview_rgb, eye_in_hand_rgb). Auto if not set.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="Frames per second (default: 30)",
    )
    parser.add_argument(
        "--max-demos-per-file",
        type=int,
        default=None,
        help="When using --all, max number of demos to convert per HDF5 file",
    )
    parser.add_argument(
        "--filename",
        type=str,
        default=None,
        help="Output filename for single-demo mode (e.g. rollout.mp4)",
    )
    args = parser.parse_args()

    try:
        paths = hdf5_to_video(
            Path(args.hdf5_path),
            Path(args.output_dir),
            demo_index=args.demo_index,
            all_demos=args.all,
            camera=args.camera,
            fps=args.fps,
            max_demos_per_file=args.max_demos_per_file,
            filename=args.filename,
        )
        if paths:
            print(f"Created {len(paths)} video(s):")
            for p in paths:
                print(f"  {p}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
