from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R

from trajectory_estimation_utils import (
    Trajectory,
    add_calibrated_yaw_rate_to_imu,
    build_distance_sensor_model_from_ground_truth,
    build_distance_sensor_model_from_odom,
    compute_trajectory_error_metrics_against_ground_truth,
    compute_yaw_from_quaternion_series,
    estimate_heading_from_ground_truth_path,
    estimate_heading_series_from_magnetometer,
    extract_heading_series_from_imu_orientation,
    find_default_dataset_paths,
    fit_imu_yaw_rate_to_odom_yaw_rate,
    load_distance_sensor_topic,
    load_ground_truth_trajectory,
    load_imu_odom_and_mag_topics,
    run_wheel_imu_ekf_in_local_frame,
    run_wheel_odometry_dead_reckoning,
    wrap_angle_to_pi,
)

def run_imu_only_dead_reckoning(
    name: str,
    reach_imu: pd.DataFrame,
    gt: pd.DataFrame,
    accel_bias_window_s: float = 2.0,
) -> Trajectory:
    imu_t = reach_imu["t"].to_numpy()
    accel_raw = reach_imu[["acc_x", "acc_y", "acc_z"]].to_numpy().copy()
    quat_xyzw = reach_imu[["ori_qx", "ori_qy", "ori_qz", "ori_qw"]].to_numpy().copy()
    yaw_from_imu = compute_yaw_from_quaternion_series(
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

    yaw = estimate_heading_from_ground_truth_path(gt, t0)
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
        yaw = float(wrap_angle_to_pi(yaw_from_imu[idx] + yaw_offset))

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


def compute_ground_truth_axis_limits(
    gt: pd.DataFrame,
    margin: float = 15.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    gt_x = gt["x"].to_numpy()
    gt_y = gt["y"].to_numpy()
    return (
        (float(gt_x.min() - margin), float(gt_x.max() + margin)),
        (float(gt_y.min() - margin), float(gt_y.max() + margin)),
    )


def compute_trajectory_axis_limits(
    traj: Trajectory,
    margin: float = 15.0,
) -> tuple[tuple[float, float], tuple[float, float]]:
    return (
        (float(np.min(traj.x) - margin), float(np.max(traj.x) + margin)),
        (float(np.min(traj.y) - margin), float(np.max(traj.y) + margin)),
    )


def translate_trajectory_by_offset(traj: Trajectory, dx: float, dy: float) -> Trajectory:
    return Trajectory(
        name=traj.name,
        t=traj.t.copy(),
        x=traj.x + dx,
        y=traj.y + dy,
        yaw=traj.yaw.copy(),
    )


def plot_all_trajectories_zoomed_to_wheel_range(
    gt: pd.DataFrame,
    trajectories: list[Trajectory],
    output_path: Path,
    *,
    zoom_reference: Trajectory,
) -> None:
    colors = {
        "Wheel Odometry": "#1f77b4",
        "IMU Only": "#ff7f0e",
        "EKF Fusion (IMU + Wheel Odometry)": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    gt_x = gt["x"].to_numpy()
    gt_y = gt["y"].to_numpy()
    zoom_xlim, zoom_ylim = compute_trajectory_axis_limits(zoom_reference)

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
    ax.set_title("Ground Truth, IMU Only, Wheel Odometry, and EKF Fusion (Zoom to Wheel Range)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_single_trajectory_vs_ground_truth(
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
        "IMU Only": "#ff7f0e",
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
        zoom_xlim, zoom_ylim = compute_ground_truth_axis_limits(gt)
        ax.set_xlim(*zoom_xlim)
        ax.set_ylim(*zoom_ylim)

    ax.set_title(title or f"{traj.name} vs Ground Truth")
    if legend_outside:
        ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    else:
        ax.legend(loc="best")

    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def compute_yaw_error_time_series(
    gt: pd.DataFrame,
    traj: Trajectory,
) -> tuple[np.ndarray, np.ndarray]:
    gt_t = gt["t"].to_numpy()
    mask = (gt_t >= traj.t[0]) & (gt_t <= traj.t[-1])
    eval_t = gt_t[mask]
    gt_yaw = gt.loc[mask, "yaw"].to_numpy()
    pred_yaw = np.interp(eval_t, traj.t, traj.yaw)
    yaw_err_deg = np.degrees(wrap_angle_to_pi(pred_yaw - gt_yaw))
    return eval_t - eval_t[0], yaw_err_deg


def plot_single_yaw_error_vs_ground_truth(
    gt: pd.DataFrame,
    traj: Trajectory,
    output_path: Path,
) -> None:
    color_map = {
        "Wheel Odometry": "#1f77b4",
        "IMU Only": "#ff7f0e",
        "EKF Fusion (IMU + Wheel Odometry)": "#d62728",
    }
    time_rel, yaw_err_deg = compute_yaw_error_time_series(gt, traj)

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    ax.plot(
        time_rel,
        yaw_err_deg,
        linewidth=1.8,
        color=color_map.get(traj.name, "#d62728"),
        label=traj.name,
    )
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Time Since Start (s)")
    ax.set_ylabel("Yaw Error (deg)")
    ax.set_title(f"{traj.name} Yaw Error vs Ground Truth")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_all_yaw_error_comparison(
    gt: pd.DataFrame,
    trajectories: list[Trajectory],
    output_path: Path,
) -> None:
    color_map = {
        "Wheel Odometry": "#1f77b4",
        "IMU Only": "#ff7f0e",
        "EKF Fusion (IMU + Wheel Odometry)": "#d62728",
    }

    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for traj in trajectories:
        time_rel, yaw_err_deg = compute_yaw_error_time_series(gt, traj)
        ax.plot(
            time_rel,
            yaw_err_deg,
            linewidth=1.6,
            color=color_map.get(traj.name, None),
            label=traj.name,
        )

    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    ax.set_xlabel("Time Since Start (s)")
    ax.set_ylabel("Yaw Error (deg)")
    ax.set_title("Yaw Error Comparison vs Ground Truth")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_comparison_summary(
    metrics_df: pd.DataFrame,
    output_path: Path,
) -> None:
    lines = [
        "Proposal-aligned comparison setup",
        "Ground Truth: Rosario v2 MINS trajectory, shown in the figure as Ground Truth.",
        "Wheel Odometry: /distance-based Ackermann dead reckoning.",
        "IMU Only: IMU-only v1 dead reckoning using the IMU's own orientation and acceleration, with only the initial pose aligned to Ground Truth.",
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
    parser = argparse.ArgumentParser(
        description="Generate the final trajectory comparison figures for the project."
    )
    parser.add_argument("--bag-dir", type=Path, default=None, help="Path to the ROS2 bag directory.")
    parser.add_argument("--ground-truth", type=Path, default=None, help="Path to the MINS TUM CSV.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the final trajectory figures and metrics.",
    )
    args = parser.parse_args()

    default_bag, default_gt = find_default_dataset_paths()
    bag_dir = (args.bag_dir or default_bag).resolve()
    gt_path = (args.ground_truth or default_gt).resolve()
    output_dir = (args.output_dir or (Path(__file__).resolve().parent / "trajectory_comparison_results")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ground_truth_df = load_ground_truth_trajectory(gt_path)
    odom_df, reach_imu_df, mag_df = load_imu_odom_and_mag_topics(bag_dir)
    distance_topic_df = load_distance_sensor_topic(bag_dir)
    ground_truth_calibrated_distance_controls = build_distance_sensor_model_from_ground_truth(
        distance_topic_df,
        ground_truth_df,
    )
    magnetometer_heading_df = estimate_heading_series_from_magnetometer(mag_df, odom_df)
    imu_yaw_rate_calibration = fit_imu_yaw_rate_to_odom_yaw_rate(reach_imu_df, odom_df)
    calibrated_reach_imu_df = add_calibrated_yaw_rate_to_imu(reach_imu_df, imu_yaw_rate_calibration)
    odom_calibrated_distance_controls = build_distance_sensor_model_from_odom(distance_topic_df, odom_df)
    imu_orientation_heading_df = extract_heading_series_from_imu_orientation(reach_imu_df)

    wheel_odometry_trajectory = run_wheel_odometry_dead_reckoning(
        "Wheel Odometry",
        ground_truth_calibrated_distance_controls,
        ground_truth_df,
    )
    imu_only_trajectory = run_imu_only_dead_reckoning(
        "IMU Only",
        reach_imu_df,
        ground_truth_df,
    )
    ekf_local_frame_trajectory = run_wheel_imu_ekf_in_local_frame(
        "EKF Fusion (IMU + Wheel Odometry)",
        calibrated_reach_imu_df,
        odom_calibrated_distance_controls,
        imu_orientation_heading_df,
        magnetometer_heading_df,
    )
    ekf_x0 = float(np.interp(ekf_local_frame_trajectory.t[0], ground_truth_df["t"], ground_truth_df["x"]))
    ekf_y0 = float(np.interp(ekf_local_frame_trajectory.t[0], ground_truth_df["t"], ground_truth_df["y"]))
    ekf_fusion_trajectory = translate_trajectory_by_offset(ekf_local_frame_trajectory, ekf_x0, ekf_y0)

    trajectories = [wheel_odometry_trajectory, imu_only_trajectory, ekf_fusion_trajectory]
    metrics = [
        {"method": traj.name, **compute_trajectory_error_metrics_against_ground_truth(traj, ground_truth_df)}
        for traj in trajectories
    ]
    metrics_df = pd.DataFrame(metrics)
    metrics_path = output_dir / "trajectory_comparison_metrics.csv"
    try:
        metrics_df.to_csv(metrics_path, index=False)
    except PermissionError:
        fallback_metrics = output_dir / "trajectory_comparison_metrics_unlocked_copy.csv"
        metrics_df.to_csv(fallback_metrics, index=False)
        print(f"Warning: could not overwrite {metrics_path}; wrote metrics to {fallback_metrics} instead.")

    plot_single_trajectory_vs_ground_truth(
        ground_truth_df,
        imu_only_trajectory,
        output_dir / "trajectory_imu_only_vs_ground_truth.png",
        title="IMU Only vs Ground Truth",
    )
    plot_single_trajectory_vs_ground_truth(
        ground_truth_df,
        imu_only_trajectory,
        output_dir / "trajectory_imu_only_zoom_near_ground_truth.png",
        title="IMU Only vs Ground Truth (Zoom Near Ground Truth)",
        zoom_to_gt=True,
    )
    plot_single_trajectory_vs_ground_truth(
        ground_truth_df,
        wheel_odometry_trajectory,
        output_dir / "trajectory_wheel_odometry_vs_ground_truth.png",
        title="Wheel Odometry vs Ground Truth",
    )
    plot_single_trajectory_vs_ground_truth(
        ground_truth_df,
        ekf_fusion_trajectory,
        output_dir / "trajectory_ekf_fusion_vs_ground_truth.png",
        title="EKF Fusion (IMU + Wheel Odometry) vs Ground Truth",
        legend_outside=True,
    )
    plot_all_trajectories_zoomed_to_wheel_range(
        ground_truth_df,
        trajectories,
        output_dir / "trajectory_comparison_zoom_to_wheel_odometry.png",
        zoom_reference=wheel_odometry_trajectory,
    )
    plot_single_yaw_error_vs_ground_truth(
        ground_truth_df,
        imu_only_trajectory,
        output_dir / "yaw_error_imu_only_vs_ground_truth.png",
    )
    plot_single_yaw_error_vs_ground_truth(
        ground_truth_df,
        wheel_odometry_trajectory,
        output_dir / "yaw_error_wheel_odometry_vs_ground_truth.png",
    )
    plot_single_yaw_error_vs_ground_truth(
        ground_truth_df,
        ekf_fusion_trajectory,
        output_dir / "yaw_error_ekf_fusion_vs_ground_truth.png",
    )
    plot_all_yaw_error_comparison(
        ground_truth_df,
        trajectories,
        output_dir / "yaw_error_all_methods_vs_ground_truth.png",
    )
    summary_path = output_dir / "trajectory_comparison_summary.txt"
    try:
        write_comparison_summary(metrics_df, summary_path)
    except PermissionError:
        fallback_summary = output_dir / "trajectory_comparison_summary_unlocked_copy.txt"
        write_comparison_summary(metrics_df, fallback_summary)
        print(f"Warning: could not overwrite {summary_path}; wrote summary to {fallback_summary} instead.")

    print("Bag:", bag_dir)
    print("Ground truth:", gt_path)
    print("Output directory:", output_dir)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
