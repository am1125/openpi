"""
Scatter plot of (x, z) body positions for every initial state in a LIBERO dataset.

For each demo: load the scene (model_xml), set the sim to that initial state,
then read world position (x, y, z) for every body and plot (x, y). Each body
is a different color.

Uses the same load_trajectory_data() as evaluate_trajectories.py.

Usage:
  python openpi/examples/libero/heatmap_initial_states.py \
    --dataset-path /path/to/libero_processed/libero_spatial_openvla_processed/<task_name>_demo.hdf5 \
    --task-suite-name libero_spatial \
    --output scatter_xy.png
"""

import dataclasses
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tyro

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from evaluate_trajectories import load_trajectory_data


LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    dataset_path: str  # Path to the task HDF5
    task_suite_name: str = "libero_spatial"  # Suite to resolve task and create env
    output: str = "scatter_initial_states_xy.png"  # Output figure path
    max_trajectories: int | None = None  # Limit number of trajectories (None = all)
    fig_width: float = 10.0
    fig_height: float = 8.0
    alpha: float = 0.5  # Point transparency
    point_size: float = 15.0  # Scatter point size
    margin: float = 0.15  # Fractional margin around data for axis limits (e.g. 0.15 = 15% padding each side)
    exclude_robot_gripper: bool = False  # If True, do not plot bodies whose name starts with "robot" or "gripper"
    separate_plots: bool = True  # If True, save two files: one density-only, one scatter-only (names get _density / _scatter)
    show_density: bool = True  # Show 2D density (histogram) behind scatter
    density_bins: int = 80  # Bins per axis for density histogram (smaller bins = finer grid)
    density_alpha: float = 0.35  # Transparency of density layer
    density_cmap: str = "viridis"  # Colormap for density (e.g. Blues, Greys, viridis)
    density_vmax: float = 20  # Max value for density color scale (bins with more counts saturate)


def _get_task_and_env(dataset_path: pathlib.Path, task_suite_name: str, model_xml: str):
    """Resolve task from dataset path and create env."""
    task_name = dataset_path.stem.replace("_demo", "").replace("_rollouts", "")
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()

    task_id = None
    for i in range(task_suite.n_tasks):
        t = task_suite.get_task(i)
        if t.name == task_name:
            task_id = i
            break
    if task_id is None:
        raise ValueError(f"Task name '{task_name}' not found in suite '{task_suite_name}'")

    task = task_suite.get_task(task_id)
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=task_bddl_file,
        camera_heights=LIBERO_ENV_RESOLUTION,
        camera_widths=LIBERO_ENV_RESOLUTION,
    )
    return task, env


def main(args: Args) -> None:
    dataset_path = pathlib.Path(args.dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    trajectories = load_trajectory_data(str(dataset_path))
    if not trajectories:
        raise ValueError(f"No trajectories in {dataset_path}")

    if args.max_trajectories is not None:
        trajectories = trajectories[: args.max_trajectories]

    _, env = _get_task_and_env(dataset_path, args.task_suite_name, trajectories[0][2])

    # body_id -> list of (x, y) across all initial states
    points_by_body: dict[int, list[tuple[float, float]]] = {}
    body_names = None

    for demo_name, initial_state, model_xml in trajectories:
        env.reset()
        env.reset_from_xml_string(model_xml)
        env.sim.reset()
        env.sim.set_state_from_flattened(initial_state)
        env.sim.forward()

        if body_names is None:
            try:
                names = getattr(env.sim.model, "body_names", None)
                body_names = list(names) if names is not None else []
            except Exception:
                body_names = []

        nbody = env.sim.model.nbody
        for body_id in range(nbody):
            x, y, z = env.sim.data.body_xpos[body_id]
            if body_id not in points_by_body:
                points_by_body[body_id] = []
            points_by_body[body_id].append((float(x), float(y)))

    env.close()

    if args.exclude_robot_gripper and body_names:
        def _skip(name: str) -> bool:
            n = (name or "").lower()
            return n.startswith("robot") or n.startswith("gripper")
        points_by_body = {bid: pts for bid, pts in points_by_body.items() if not _skip(body_names[bid] if bid < len(body_names) else "")}

    # Flatten all points for density
    all_x = []
    all_y = []
    for points in points_by_body.values():
        for x, y in points:
            all_x.append(x)
            all_y.append(y)
    all_x = np.array(all_x)
    all_y = np.array(all_y)

    # Distinct markers and colors so each body is easy to tell apart
    markers = ["o", "s", "^", "v", "<", ">", "D", "P", "p", "*", "X", "h", "H", "d", "1", "2", "3", "4", "8", "."]
    n_markers = len(markers)
    n_colors = 40

    task_name = dataset_path.stem.replace("_demo", "").replace("_rollouts", "")
    title_base = f"Initial state body positions (x, y) — {task_name}\n{len(trajectories)} demos × {len(points_by_body)} bodies"
    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Axis limits from data with margin
    if len(all_x) > 0 and len(all_y) > 0:
        x_min, x_max = np.min(all_x), np.max(all_x)
        y_min, y_max = np.min(all_y), np.max(all_y)
        x_range = max(x_max - x_min, 0.01)
        y_range = max(y_max - y_min, 0.01)
        pad_x = x_range * args.margin
        pad_y = y_range * args.margin
        x_lim = (x_min - pad_x, x_max + pad_x)
        y_lim = (y_min - pad_y, y_max + pad_y)
    else:
        x_lim = y_lim = None

    def _set_axes(ax):
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_aspect("equal")
        if x_lim is not None and y_lim is not None:
            ax.set_xlim(*x_lim)
            ax.set_ylim(*y_lim)

    if args.separate_plots:
        # Plot 1: density only
        if args.show_density and len(all_x) > 0:
            H, xedges, yedges = np.histogram2d(all_x, all_y, bins=args.density_bins)
            H = H.T
            H_masked = np.ma.masked_where(H <= 0, H)
            cmap = plt.get_cmap(args.density_cmap).copy()
            cmap.set_bad(color="none", alpha=0)
            fig_d, ax_d = plt.subplots(figsize=(args.fig_width, args.fig_height))
            im = ax_d.pcolormesh(
                xedges, yedges, H_masked,
                cmap=cmap, alpha=args.density_alpha, zorder=0,
                shading="flat",
                vmin=0, vmax=args.density_vmax,
            )
            cbar = plt.colorbar(im, ax=ax_d, shrink=0.6, label="Point count")
            cbar.ax.tick_params(labelsize=8)
            _set_axes(ax_d)
            ax_d.set_title(f"{title_base} — density")
            plt.tight_layout()
            density_path = out_path.parent / f"{out_path.stem}_density{out_path.suffix}"
            plt.savefig(density_path, dpi=150)
            plt.close()
            print(f"Saved density plot to {density_path}")

        # Plot 2: scatter only
        fig_s, ax_s = plt.subplots(figsize=(args.fig_width, args.fig_height))
        for i, (body_id, points) in enumerate(sorted(points_by_body.items())):
            xs, ys = zip(*points)
            label = body_names[body_id] if body_names and body_id < len(body_names) else f"body_{body_id}"
            cidx = i % n_colors
            color = plt.cm.tab20(cidx / 19.0) if cidx < 20 else plt.cm.tab20b((cidx - 20) / 19.0)
            marker = markers[i % n_markers]
            ax_s.scatter(xs, ys, c=[color], marker=marker, label=label, alpha=args.alpha, s=args.point_size, edgecolors="black", linewidths=0.3, zorder=1)
        _set_axes(ax_s)
        ax_s.set_title(f"{title_base} — scatter")
        n_legend = len(points_by_body)
        if n_legend <= 25:
            ax_s.legend(loc="best", fontsize=8)
        else:
            ax_s.legend(loc="best", fontsize=6, ncol=2)
        plt.tight_layout()
        scatter_path = out_path.parent / f"{out_path.stem}_scatter{out_path.suffix}"
        plt.savefig(scatter_path, dpi=150)
        plt.close()
        print(f"Saved scatter plot to {scatter_path}")
    else:
        # Combined: density + scatter in one figure
        fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height))
        if args.show_density and len(all_x) > 0:
            H, xedges, yedges = np.histogram2d(all_x, all_y, bins=args.density_bins)
            H = H.T
            H_masked = np.ma.masked_where(H <= 0, H)
            cmap = plt.get_cmap(args.density_cmap).copy()
            cmap.set_bad(color="none", alpha=0)
            im = ax.pcolormesh(
                xedges, yedges, H_masked,
                cmap=cmap, alpha=args.density_alpha, zorder=0,
                shading="flat",
                vmin=0, vmax=args.density_vmax,
            )
            cbar = plt.colorbar(im, ax=ax, shrink=0.6, label="Point count")
            cbar.ax.tick_params(labelsize=8)
        for i, (body_id, points) in enumerate(sorted(points_by_body.items())):
            xs, ys = zip(*points)
            label = body_names[body_id] if body_names and body_id < len(body_names) else f"body_{body_id}"
            cidx = i % n_colors
            color = plt.cm.tab20(cidx / 19.0) if cidx < 20 else plt.cm.tab20b((cidx - 20) / 19.0)
            marker = markers[i % n_markers]
            ax.scatter(xs, ys, c=[color], marker=marker, label=label, alpha=args.alpha, s=args.point_size, edgecolors="black", linewidths=0.3, zorder=1)
        _set_axes(ax)
        ax.set_title(title_base)
        n_legend = len(points_by_body)
        if n_legend <= 25:
            ax.legend(loc="best", fontsize=8)
        else:
            ax.legend(loc="best", fontsize=6, ncol=2)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    tyro.cli(main)
