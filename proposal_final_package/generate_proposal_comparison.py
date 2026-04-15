from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from generate_generalized_ekf_figures import (
    build_distance_controls_sensor_only,
    build_imu_heading_series,
    run_local_frame_ekf,
)
from run_ekf_distance_fusion_experiments import (
    build_distance_controls,
    load_distance,
    run_distance_wheel_baseline,
)
from run_ekf_fusion_experiments import (
    Trajectory,
    apply_gyro_calibration,
    build_mag_heading_series,
    calibrate_gyro_to_wheel,
    estimate_heading_from_path,
    evaluate_trajectory,
    load_ground_truth,
    load_signals,
    resolve_default_paths,
    wrap_angle,
    yaw_from_quaternion,
)


def build_imu_only_mag_heading(
    mag_df: pd.DataFrame,
    gyro_time: np.ndarray,
    gyro_relative_heading: np.ndarray,
    alignment_window_s: float = 30.0,
) -> pd.DataFrame:
    samples = mag_df[["mx", "my", "mz"]].to_numpy()
    center = samples.mean(axis=0)
    centered = samples - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    plane_basis = vh[:2].T
    uv = centered @ plane_basis

    half_range = 0.5 * (uv.max(axis=0) - uv.min(axis=0))
    half_range[half_range == 0.0] = 1.0
    uv_cal = uv * (np.mean(half_range) / half_range)

    mag_t = mag_df["t"].to_numpy()
    ref_end = float(mag_t[0] + alignment_window_s)
    ref_mask = (mag_t >= mag_t[0]) & (mag_t <= ref_end)
    ref_heading = np.interp(mag_t[ref_mask], gyro_time, gyro_relative_heading)

    best_corr = -np.inf
    best_heading: np.ndarray | None = None
    for order in ((0, 1), (1, 0)):
        for sx in (1.0, -1.0):
            for sy in (1.0, -1.0):
                candidate = np.unwrap(np.arctan2(sy * uv_cal[:, order[1]], sx * uv_cal[:, order[0]]))
                candidate_ref = np.interp(mag_t[ref_mask], mag_t, candidate)
                candidate_ref = candidate_ref - candidate_ref[0]
                corr = float(np.corrcoef(ref_heading, candidate_ref)[0, 1])
                if np.isnan(corr):
                    corr = -np.inf
                if corr > best_corr:
                    best_corr = corr
                    best_heading = candidate

    if best_heading is None:
        raise ValueError("Failed to derive a magnetometer heading for the IMU-only baseline.")

    return pd.DataFrame({"t": mag_t, "heading": best_heading, "u": uv_cal[:, 0], "v": uv_cal[:, 1]})


def run_imu_only_simulation(
    name: str,
    reach_imu: pd.DataFrame,
    gt: pd.DataFrame,
    accel_bias_window_s: float = 2.0,
) -> Trajectory:
    imu_t = reach_imu["t"].to_numpy()
    accel_raw = reach_imu[["acc_x", "acc_y", "acc_z"]].to_numpy().copy()
    quat_xyzw = reach_imu[["ori_qx", "ori_qy", "ori_qz", "ori_qw"]].to_numpy().copy()
    yaw_from_imu = yaw_from_quaternion(
        quat_xyzw[:, 0],
        quat_xyzw[:, 1],
        quat_xyzw[:, 2],
        quat_xyzw[:, 3],
    )

    t0 = max(float(imu_t[0]), float(gt["t"].iloc[0]))
    t_end = min(float(imu_t[-1]), float(gt["t"].iloc[-1]))
    if t_end <= t0:
        raise ValueError("No overlapping interval for the IMU-only simulation.")

    rotations = R.from_quat(quat_xyzw)
    accel_world = rotations.apply(accel_raw)
    accel_world -= np.array([0.0, 0.0, 9.81])

    yaw = estimate_heading_from_path(gt, t0)
    x = float(np.interp(t0, gt["t"], gt["x"]))
    y = float(np.interp(t0, gt["t"], gt["y"]))
    yaw_offset = yaw - float(np.interp(t0, imu_t, yaw_from_imu))
    yaw_rot = np.array(
        [
            [math.cos(yaw_offset), -math.sin(yaw_offset)],
            [math.sin(yaw_offset), math.cos(yaw_offset)],
        ]
    )
    accel_xy = (yaw_rot @ accel_world[:, :2].T).T

    bias_end = float(t0 + accel_bias_window_s)
    bias_mask = (imu_t >= t0) & (imu_t <= bias_end)
    if np.any(bias_mask):
        accel_xy -= np.median(accel_xy[bias_mask], axis=0)

    start_idx = int(np.searchsorted(imu_t, t0, side="left"))
    end_idx = int(np.searchsorted(imu_t, t_end, side="right"))
    vx = 0.0
    vy = 0.0

    t_hist = [t0]
    x_hist = [x]
    y_hist = [y]
    yaw_hist = [yaw]

    for idx in range(start_idx + 1, end_idx):
        dt = float(imu_t[idx] - imu_t[idx - 1])
        if dt <= 0.0 or dt > 0.1:
            continue

        ax_world, ay_world = accel_xy[idx]
        x += vx * dt + 0.5 * ax_world * dt * dt
        y += vy * dt + 0.5 * ay_world * dt * dt
        vx += ax_world * dt
        vy += ay_world * dt
        yaw = float(wrap_angle(yaw_from_imu[idx] + yaw_offset))

        if imu_t[idx] - t_hist[-1] >= 0.1:
            t_hist.append(float(imu_t[idx]))
            x_hist.append(x)
            y_hist.append(y)
            yaw_hist.append(yaw)

    return Trajectory(
        name=name,
        t=np.asarray(t_hist),
        x=np.asarray(x_hist),
        y=np.asarray(y_hist),
        yaw=np.unwrap(np.asarray(yaw_hist)),
    )


def get_gt_zoom_limits(gt: pd.DataFrame, margin: float = 15.0) -> tuple[tuple[float, float], tuple[float, float]]:
    gt_x = gt["x"].to_numpy()
    gt_y = gt["y"].to_numpy()
    return (
        (float(gt_x.min() - margin), float(gt_x.max() + margin)),
        (float(gt_y.min() - margin), float(gt_y.max() + margin)),
    )


def get_traj_zoom_limits(
    traj: Trajectory,
    margin: float = 15.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        (float(np.min(traj.x) - margin), float(np.max(traj.x) + margin)),
        (float(np.min(traj.y) - margin), float(np.max(traj.y) + margin)),
    )


def translate_trajectory(traj: Trajectory, dx: float, dy: float) -> Trajectory:
    return Trajectory(
        name=traj.name,
        t=traj.t.copy(),
        x=traj.x + dx,
        y=traj.y + dy,
        yaw=traj.yaw.copy(),
    )


def plot_proposal_comparison_zoom(
    gt: pd.DataFrame,
    trajectories: list[Trajectory],
    output_path: Path,
    *,
    zoom_reference: Trajectory,
) -> None:
    colors = {
        "Wheel Odometry": "#1f77b4",
        "IMU": "#ff7f0e",
        "EKF Fusion (IMU + Wheel Odometry)": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    gt_x = gt["x"].to_numpy()
    gt_y = gt["y"].to_numpy()
    zoom_xlim, zoom_ylim = get_traj_zoom_limits(zoom_reference)

    ax.plot(gt_x, gt_y, color="black", linewidth=2.4, label="Ground Truth")
    for traj in trajectories:
        ax.plot(
            traj.x,
            traj.y,
            linewidth=1.8,
            color=colors.get(traj.name, None),
            label=traj.name,
        )
    ax.scatter(gt_x[0], gt_y[0], c="green", s=70, label="Start", zorder=5)
    ax.scatter(gt_x[-1], gt_y[-1], c="red", s=70, label="End", zorder=5)
    ax.set_xlabel("X / East (m)")
    ax.set_ylabel("Y / North (m)")
    ax.set_xlim(*zoom_xlim)
    ax.set_ylim(*zoom_ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Ground Truth, IMU, Wheel Odometry, and EKF Fusion (Zoom Near Ground Truth)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_single_method_vs_gt(
    gt: pd.DataFrame,
    traj: Trajectory,
    output_path: Path,
    *,
    title: str | None = None,
    zoom_to_gt: bool = False,
    legend_outside: bool = False,
) -> None:
    color_map = {
        "Wheel Odometry": "#1f77b4",
        "IMU": "#ff7f0e",
        "EKF Fusion (IMU + Wheel Odometry)": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    gt_x = gt["x"].to_numpy()
    gt_y = gt["y"].to_numpy()

    ax.plot(gt_x, gt_y, color="black", linewidth=2.4, label="Ground Truth")
    ax.plot(
        traj.x,
        traj.y,
        linewidth=2.0,
        color=color_map.get(traj.name, "#d62728"),
        label=traj.name,
    )
    ax.scatter(gt_x[0], gt_y[0], c="green", s=70, label="Start", zorder=5)
    ax.scatter(gt_x[-1], gt_y[-1], c="red", s=70, label="End", zorder=5)
    ax.set_xlabel("X / East (m)")
    ax.set_ylabel("Y / North (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)

    if zoom_to_gt:
        zoom_xlim, zoom_ylim = get_gt_zoom_limits(gt)
        ax.set_xlim(*zoom_xlim)
        ax.set_ylim(*zoom_ylim)

    ax.set_title(title or f"{traj.name} vs Ground Truth")
    if legend_outside:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    else:
        ax.legend(loc="best")

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_summary(
    metrics_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "Proposal-aligned comparison setup",
        "Ground Truth: Rosario v2 MINS trajectory, shown in the figure as Ground Truth.",
        "Wheel Odometry: /distance-based Ackermann dead reckoning.",
        "IMU: IMU-only v1 dead reckoning using the IMU's own orientation and acceleration, with only the initial pose aligned to Ground Truth.",
        "EKF Fusion (IMU + Wheel Odometry): Fully GT-Free wheel-IMU EKF using IMU heading initialization and sensor-only wheel/gyro calibration.",
        "For direct overlay in the proposal figures, the EKF trajectory is translated by a constant initial-position offset only; no GT heading alignment is used.",
        "",
        "Metrics versus Ground Truth",
    ]
    for _, row in metrics_df.iterrows():
        lines.append(
            f"{row['method']}: pos_rmse_m={row['pos_rmse_m']:.3f}, "
            f"final_pos_err_m={row['final_pos_err_m']:.3f}, "
            f"yaw_rmse_deg={row['yaw_rmse_deg']:.3f}"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the proposal-style trajectory comparison figure.")
    parser.add_argument("--bag-dir", type=Path, default=None, help="Path to the ROS2 bag directory.")
    parser.add_argument("--ground-truth", type=Path, default=None, help="Path to the MINS TUM CSV.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for the final proposal figure and metrics.")
    args = parser.parse_args()

    default_bag, default_gt = resolve_default_paths()
    bag_dir = (args.bag_dir or default_bag).resolve()
    gt_path = (args.ground_truth or default_gt).resolve()
    output_dir = (args.output_dir or (Path(__file__).resolve().parent / "proposal_results")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gt = load_ground_truth(gt_path)
    odom, _, reach_imu, mag = load_signals(bag_dir)
    distance_df = load_distance(bag_dir)
    distance_controls, _ = build_distance_controls(distance_df, gt)

    raw_mag_heading, _, _ = build_mag_heading_series(mag, odom)
    reach_calibration = calibrate_gyro_to_wheel(reach_imu, odom)
    reach_imu_calibrated = apply_gyro_calibration(reach_imu, reach_calibration)
    fully_gt_free_controls, _ = build_distance_controls_sensor_only(distance_df, odom)
    init_heading_df = build_imu_heading_series(reach_imu)

    wheel_traj = run_distance_wheel_baseline("Wheel Odometry", distance_controls, gt)
    imu_traj = run_imu_only_simulation(
        "IMU",
        reach_imu,
        gt,
    )
    ekf_local_traj = run_local_frame_ekf(
        "EKF Fusion (IMU + Wheel Odometry)",
        reach_imu_calibrated,
        fully_gt_free_controls,
        init_heading_df,
        raw_mag_heading,
    )
    ekf_x0 = float(np.interp(ekf_local_traj.t[0], gt["t"], gt["x"]))
    ekf_y0 = float(np.interp(ekf_local_traj.t[0], gt["t"], gt["y"]))
    ekf_traj = translate_trajectory(ekf_local_traj, ekf_x0, ekf_y0)

    trajectories = [wheel_traj, imu_traj, ekf_traj]
    metrics = [{"method": traj.name, **evaluate_trajectory(traj, gt)} for traj in trajectories]
    metrics_df = pd.DataFrame(metrics)
    metrics_path = output_dir / "proposal_comparison_metrics.csv"
    try:
        metrics_df.to_csv(metrics_path, index=False)
    except PermissionError:
        fallback_metrics = output_dir / "proposal_comparison_metrics_unlocked_copy.csv"
        metrics_df.to_csv(fallback_metrics, index=False)
        print(f"Warning: could not overwrite {metrics_path}; wrote metrics to {fallback_metrics} instead.")

    plot_single_method_vs_gt(
        gt,
        imu_traj,
        output_dir / "imu_vs_ground_truth.png",
        title="IMU vs Ground Truth",
    )
    plot_single_method_vs_gt(
        gt,
        imu_traj,
        output_dir / "imu_vs_ground_truth_zoom.png",
        title="IMU vs Ground Truth (Zoom Near Ground Truth)",
        zoom_to_gt=True,
    )
    plot_single_method_vs_gt(
        gt,
        wheel_traj,
        output_dir / "wheel_odometry_vs_ground_truth.png",
        title="Wheel Odometry vs Ground Truth",
    )
    plot_single_method_vs_gt(
        gt,
        ekf_traj,
        output_dir / "ekf_fusion_vs_ground_truth.png",
        title="EKF Fusion (IMU + Wheel Odometry) vs Ground Truth",
        legend_outside=True,
    )
    plot_proposal_comparison_zoom(
        gt,
        trajectories,
        output_dir / "proposal_trajectory_comparison.png",
        zoom_reference=wheel_traj,
    )
    summary_path = output_dir / "proposal_comparison_summary.txt"
    try:
        save_summary(metrics_df, summary_path)
    except PermissionError:
        fallback_summary = output_dir / "proposal_comparison_summary_unlocked_copy.txt"
        save_summary(metrics_df, fallback_summary)
        print(f"Warning: could not overwrite {summary_path}; wrote summary to {fallback_summary} instead.")

    print("Bag:", bag_dir)
    print("Ground truth:", gt_path)
    print("Output directory:", output_dir)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
