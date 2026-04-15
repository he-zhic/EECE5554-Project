from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader

from run_ekf_fusion_experiments import (
    Trajectory,
    apply_gyro_calibration,
    build_mag_heading_series,
    calibrate_gyro_to_wheel,
    estimate_heading_from_path,
    evaluate_trajectory,
    load_ground_truth,
    load_signals,
    plot_mag_calibration,
    plot_yaw_errors,
    resolve_default_paths,
    save_metrics,
    wrap_angle,
)


@dataclass
class DistanceCalibration:
    meters_per_pulse: float
    yaw_gain: float
    yaw_bias: float
    calibration_duration_s: float
    pulse_window: int


@dataclass
class HeadingAlignment:
    offset_rad: float
    rmse_deg: float
    mean_abs_deg: float
    samples: int


def load_distance(bag_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, float]] = []
    with AnyReader([bag_dir]) as reader:
        conns = [conn for conn in reader.connections if conn.topic == "/distance"]
        for conn, timestamp, rawdata in reader.messages(conns):
            msg = reader.deserialize(rawdata, conn.msgtype)
            rows.append(
                {
                    "t": timestamp * 1e-9,
                    "pulses": float(msg.pulses),
                    "angle": float(msg.angle),
                    "direction": float(msg.direction),
                }
            )
    return pd.DataFrame(rows).sort_values("t").reset_index(drop=True)


def estimate_heading_offset_to_gt(
    signal_t: np.ndarray,
    signal_heading: np.ndarray,
    gt: pd.DataFrame,
    start_t: float,
    calibration_duration_s: float,
) -> HeadingAlignment:
    end_t = min(float(signal_t[-1]), float(gt["t"].iloc[-1]), start_t + calibration_duration_s)
    gt_mask = (gt["t"] >= start_t) & (gt["t"] <= end_t)
    eval_t = gt.loc[gt_mask, "t"].to_numpy()

    if len(eval_t) < 5:
        return HeadingAlignment(offset_rad=0.0, rmse_deg=0.0, mean_abs_deg=0.0, samples=int(len(eval_t)))

    gt_heading = np.asarray([estimate_heading_from_path(gt, float(t)) for t in eval_t])
    signal_interp = np.interp(eval_t, signal_t, signal_heading)
    diff = wrap_angle(gt_heading - signal_interp)
    offset = float(np.arctan2(np.mean(np.sin(diff)), np.mean(np.cos(diff))))
    residual_deg = np.degrees(wrap_angle(signal_interp + offset - gt_heading))
    return HeadingAlignment(
        offset_rad=offset,
        rmse_deg=float(np.sqrt(np.mean(residual_deg**2))),
        mean_abs_deg=float(np.mean(np.abs(residual_deg))),
        samples=int(len(eval_t)),
    )


def build_distance_controls(
    distance_df: pd.DataFrame,
    gt: pd.DataFrame,
    calibration_duration_s: float = 300.0,
    pulse_window: int = 15,
) -> tuple[pd.DataFrame, DistanceCalibration]:
    df = distance_df.copy()
    df["dt"] = df["t"].diff().fillna(0.0)
    df["dp"] = df["pulses"].diff().fillna(0.0)

    cal_end = float(df["t"].iloc[0] + calibration_duration_s)
    dist_mask = df["t"] <= cal_end
    gt_mask = gt["t"] <= cal_end

    if dist_mask.sum() < 2 or gt_mask.sum() < 2:
        raise ValueError("Not enough data inside the calibration window.")

    gt_path = np.concatenate(
        [[0.0], np.cumsum(np.hypot(np.diff(gt["x"].to_numpy()), np.diff(gt["y"].to_numpy())))]
    )
    gt_yaw = gt["yaw"].to_numpy()
    gt_yaw_rate = np.gradient(gt_yaw, gt["t"].to_numpy())

    pulse_delta = float(df.loc[dist_mask, "pulses"].iloc[-1] - df.loc[dist_mask, "pulses"].iloc[0])
    gt_delta = float(gt_path[gt_mask.to_numpy()][-1] - gt_path[gt_mask.to_numpy()][0])
    if abs(pulse_delta) < 1e-9:
        raise ValueError("Pulse count did not change in the calibration segment.")

    meters_per_pulse = gt_delta / pulse_delta

    raw_pulse_rate = df["direction"] * df["dp"] / df["dt"].replace(0.0, np.nan)
    smoothed_rate = raw_pulse_rate.rolling(pulse_window, center=True, min_periods=1).mean().fillna(0.0)
    df["speed"] = meters_per_pulse * smoothed_rate
    df["delta_s"] = meters_per_pulse * df["direction"] * df["dp"]

    steering_base = df["speed"] * np.tan(df["angle"])
    interp_gt_yaw_rate = np.interp(df["t"], gt["t"], gt_yaw_rate)
    valid = (df["t"] <= cal_end) & np.isfinite(interp_gt_yaw_rate) & (np.abs(steering_base) > 1e-4)

    design = np.column_stack([steering_base[valid], np.ones(int(valid.sum()))])
    yaw_gain, yaw_bias = np.linalg.lstsq(design, interp_gt_yaw_rate[valid], rcond=None)[0]
    df["wheel_yaw_rate"] = yaw_gain * steering_base + yaw_bias

    wheel_heading = np.zeros(len(df), dtype=float)
    wheel_heading[0] = estimate_heading_from_path(gt, float(df["t"].iloc[0]))
    for idx in range(1, len(df)):
        dt = float(df["t"].iloc[idx] - df["t"].iloc[idx - 1])
        wheel_heading[idx] = wheel_heading[idx - 1] + 0.5 * (
            df["wheel_yaw_rate"].iloc[idx] + df["wheel_yaw_rate"].iloc[idx - 1]
        ) * dt
    df["wheel_heading"] = wheel_heading

    calibration = DistanceCalibration(
        meters_per_pulse=float(meters_per_pulse),
        yaw_gain=float(yaw_gain),
        yaw_bias=float(yaw_bias),
        calibration_duration_s=calibration_duration_s,
        pulse_window=pulse_window,
    )
    return df, calibration


def run_distance_wheel_baseline(name: str, distance_controls: pd.DataFrame, gt: pd.DataFrame) -> Trajectory:
    t = distance_controls["t"].to_numpy()
    delta_s = distance_controls["delta_s"].to_numpy()
    yaw_rate = distance_controls["wheel_yaw_rate"].to_numpy()

    x = np.zeros_like(t)
    y = np.zeros_like(t)
    yaw = np.zeros_like(t)

    x[0] = float(gt["x"].iloc[0])
    y[0] = float(gt["y"].iloc[0])
    yaw[0] = estimate_heading_from_path(gt, float(t[0]))

    for idx in range(1, len(t)):
        dt = float(t[idx] - t[idx - 1])
        mean_yaw_rate = 0.5 * (yaw_rate[idx] + yaw_rate[idx - 1])
        heading_mid = yaw[idx - 1] + 0.5 * mean_yaw_rate * dt
        yaw[idx] = yaw[idx - 1] + mean_yaw_rate * dt
        x[idx] = x[idx - 1] + delta_s[idx] * math.cos(heading_mid)
        y[idx] = y[idx - 1] + delta_s[idx] * math.sin(heading_mid)

    return Trajectory(name=name, t=t, x=x, y=y, yaw=np.unwrap(yaw))


def run_distance_ekf(
    name: str,
    imu_df: pd.DataFrame,
    distance_controls: pd.DataFrame,
    gt: pd.DataFrame,
    mag_df: pd.DataFrame | None = None,
    heading_calibration_duration_s: float = 300.0,
    speed_noise: float = 0.30,
    wheel_yaw_noise: float = 0.40,
    mag_yaw_noise: float = 0.35,
    process_yaw_noise: float = 0.12,
    process_v_noise: float = 0.45,
) -> Trajectory:
    imu_t = imu_df["t"].to_numpy()
    imu_rate = imu_df["yaw_rate"].to_numpy()
    wheel_t = distance_controls["t"].to_numpy()
    wheel_v = distance_controls["speed"].to_numpy()
    wheel_heading = distance_controls["wheel_heading"].to_numpy()

    mag_t = mag_df["t"].to_numpy() if mag_df is not None else np.array([], dtype=float)
    mag_yaw = mag_df["heading"].to_numpy() if mag_df is not None else np.array([], dtype=float)

    t0_candidates = [imu_t[0], wheel_t[0], gt["t"].iloc[0]]
    if len(mag_t):
        t0_candidates.append(mag_t[0])
    t_end_candidates = [imu_t[-1], wheel_t[-1], gt["t"].iloc[-1]]
    if len(mag_t):
        t_end_candidates.append(mag_t[-1])

    t0 = max(t0_candidates)
    t_end = min(t_end_candidates)
    if t_end <= t0:
        raise ValueError(f"No overlapping interval for {name}.")

    yaw0 = estimate_heading_from_path(gt, t0)
    x0 = float(np.interp(t0, gt["t"], gt["x"]))
    y0 = float(np.interp(t0, gt["t"], gt["y"]))
    v0 = float(np.interp(t0, wheel_t, wheel_v))

    state = np.array([x0, y0, yaw0, v0], dtype=float)
    cov = np.diag([0.05, 0.05, 0.12, 0.25]) ** 2

    current_rate = float(np.interp(t0, imu_t, imu_rate))
    wheel_heading_offset = estimate_heading_offset_to_gt(
        wheel_t,
        wheel_heading,
        gt,
        t0,
        heading_calibration_duration_s,
    ).offset_rad
    mag_offset = (
        estimate_heading_offset_to_gt(
            mag_t,
            mag_yaw,
            gt,
            t0,
            heading_calibration_duration_s,
        ).offset_rad
        if len(mag_t)
        else 0.0
    )

    imu_idx = int(np.searchsorted(imu_t, t0, side="right"))
    wheel_idx = int(np.searchsorted(wheel_t, t0, side="left"))
    mag_idx = int(np.searchsorted(mag_t, t0, side="left")) if len(mag_t) else 0

    t_hist = [t0]
    x_hist = [state[0]]
    y_hist = [state[1]]
    yaw_hist = [state[2]]
    current_t = t0

    while True:
        next_imu = imu_t[imu_idx] if imu_idx < len(imu_t) else np.inf
        next_wheel = wheel_t[wheel_idx] if wheel_idx < len(wheel_t) else np.inf
        next_mag = mag_t[mag_idx] if len(mag_t) and mag_idx < len(mag_t) else np.inf
        next_t = min(next_imu, next_wheel, next_mag)
        if not np.isfinite(next_t) or next_t > t_end:
            break

        dt = float(max(0.0, next_t - current_t))
        x_prev, y_prev, yaw_prev, v_prev = state
        state[0] = x_prev + v_prev * math.cos(yaw_prev) * dt
        state[1] = y_prev + v_prev * math.sin(yaw_prev) * dt
        state[2] = float(wrap_angle(yaw_prev + current_rate * dt))

        jacobian = np.array(
            [
                [1.0, 0.0, -v_prev * math.sin(yaw_prev) * dt, math.cos(yaw_prev) * dt],
                [0.0, 1.0, v_prev * math.cos(yaw_prev) * dt, math.sin(yaw_prev) * dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        process_cov = np.diag(
            [
                1e-4 * max(dt, 1e-3),
                1e-4 * max(dt, 1e-3),
                (process_yaw_noise * max(dt, 1e-3)) ** 2,
                (process_v_noise * max(dt, 1e-3)) ** 2,
            ]
        )
        cov = jacobian @ cov @ jacobian.T + process_cov

        while imu_idx < len(imu_t) and abs(imu_t[imu_idx] - next_t) < 1e-9:
            current_rate = float(imu_rate[imu_idx])
            imu_idx += 1

        while wheel_idx < len(wheel_t) and abs(wheel_t[wheel_idx] - next_t) < 1e-9:
            speed_meas = float(wheel_v[wheel_idx])
            speed_h = np.array([[0.0, 0.0, 0.0, 1.0]])
            speed_r = np.array([[speed_noise**2]])
            speed_innovation = np.array([[speed_meas - state[3]]])
            speed_s = speed_h @ cov @ speed_h.T + speed_r
            speed_k = cov @ speed_h.T @ np.linalg.inv(speed_s)
            state = state + (speed_k @ speed_innovation).ravel()
            cov = (np.eye(4) - speed_k @ speed_h) @ cov

            yaw_meas = float(wheel_heading[wheel_idx] + wheel_heading_offset)
            yaw_h = np.array([[0.0, 0.0, 1.0, 0.0]])
            yaw_r = np.array([[wheel_yaw_noise**2]])
            yaw_innovation = wrap_angle(yaw_meas - state[2])
            yaw_s = yaw_h @ cov @ yaw_h.T + yaw_r
            yaw_k = cov @ yaw_h.T @ np.linalg.inv(yaw_s)
            state = state + (yaw_k.flatten() * yaw_innovation)
            state[2] = float(wrap_angle(state[2]))
            cov = (np.eye(4) - yaw_k @ yaw_h) @ cov
            wheel_idx += 1

        while len(mag_t) and mag_idx < len(mag_t) and abs(mag_t[mag_idx] - next_t) < 1e-9:
            measurement = float(mag_yaw[mag_idx] + mag_offset)
            mag_h = np.array([[0.0, 0.0, 1.0, 0.0]])
            mag_r = np.array([[mag_yaw_noise**2]])
            mag_innovation = wrap_angle(measurement - state[2])
            mag_s = mag_h @ cov @ mag_h.T + mag_r
            mag_k = cov @ mag_h.T @ np.linalg.inv(mag_s)
            state = state + (mag_k.flatten() * mag_innovation)
            state[2] = float(wrap_angle(state[2]))
            cov = (np.eye(4) - mag_k @ mag_h) @ cov
            mag_idx += 1

        current_t = next_t
        t_hist.append(current_t)
        x_hist.append(state[0])
        y_hist.append(state[1])
        yaw_hist.append(state[2])

    return Trajectory(
        name=name,
        t=np.asarray(t_hist),
        x=np.asarray(x_hist),
        y=np.asarray(y_hist),
        yaw=np.unwrap(np.asarray(yaw_hist)),
    )


def plot_distance_trajectories(gt: pd.DataFrame, trajectories: list[Trajectory], output_path: Path) -> None:
    plt.figure(figsize=(10, 8))
    plt.plot(gt["x"], gt["y"], color="black", linewidth=2.2, label="MINS ground truth")
    for traj in trajectories:
        plt.plot(traj.x, traj.y, linewidth=1.6, label=traj.name)
    plt.scatter(gt["x"].iloc[0], gt["y"].iloc[0], c="green", s=60, label="Start", zorder=5)
    plt.scatter(gt["x"].iloc[-1], gt["y"].iloc[-1], c="red", s=60, label="End", zorder=5)
    plt.xlabel("X / East (m)")
    plt.ylabel("Y / North (m)")
    plt.title("Rosario v2: /distance-based Wheel-IMU EKF Comparison")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def build_summary(
    distance_cal: DistanceCalibration,
    rs_calibration,
    reach_calibration,
    heading_alignments: dict[str, HeadingAlignment],
    output_path: Path,
) -> None:
    lines = [
        "Distance control calibration",
        f"meters_per_pulse = {distance_cal.meters_per_pulse:.8f}",
        f"yaw_gain = {distance_cal.yaw_gain:.8f}",
        f"yaw_bias = {distance_cal.yaw_bias:.8f}",
        f"calibration_duration_s = {distance_cal.calibration_duration_s:.1f}",
        f"pulse_window = {distance_cal.pulse_window}",
        "",
        "Gyro axis calibration against /odom yaw rate",
        f"Realsense: axis={rs_calibration.axis}, sign={rs_calibration.sign}, scale={rs_calibration.scale:.6f}, bias={rs_calibration.bias:.6f}, corr={rs_calibration.corr:.4f}",
        f"Reach1: axis={reach_calibration.axis}, sign={reach_calibration.sign}, scale={reach_calibration.scale:.6f}, bias={reach_calibration.bias:.6f}, corr={reach_calibration.corr:.4f}",
        "",
        "Heading alignment to MINS ground truth over calibration window",
    ]
    for name, alignment in heading_alignments.items():
        lines.append(
            f"{name}: offset_deg = {math.degrees(alignment.offset_rad):.6f}, "
            f"rmse_deg = {alignment.rmse_deg:.6f}, mean_abs_deg = {alignment.mean_abs_deg:.6f}, "
            f"samples = {alignment.samples}"
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run /distance-based Ackermann wheel + IMU EKF experiments on Rosario v2.")
    parser.add_argument("--bag-dir", type=Path, default=None, help="Path to the ROS2 bag directory.")
    parser.add_argument("--ground-truth", type=Path, default=None, help="Path to the MINS TUM CSV.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for plots and metrics.")
    parser.add_argument("--calibration-duration", type=float, default=300.0, help="Seconds from the start used to calibrate /distance.")
    parser.add_argument("--heading-calibration-duration", type=float, default=300.0, help="Seconds from the start used to estimate constant heading offsets.")
    parser.add_argument("--pulse-window", type=int, default=15, help="Rolling window size for pulse-rate smoothing.")
    args = parser.parse_args()

    default_bag, default_gt = resolve_default_paths()
    bag_dir = (args.bag_dir or default_bag).resolve()
    gt_path = (args.ground_truth or default_gt).resolve()
    output_dir = (args.output_dir or (Path(__file__).resolve().parent / "ekf_results_distance")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gt = load_ground_truth(gt_path)
    odom, rs_imu, reach_imu, mag = load_signals(bag_dir)
    distance_df = load_distance(bag_dir)
    distance_controls, distance_cal = build_distance_controls(
        distance_df,
        gt,
        calibration_duration_s=args.calibration_duration,
        pulse_window=args.pulse_window,
    )

    rs_calibration = calibrate_gyro_to_wheel(rs_imu, odom)
    reach_calibration = calibrate_gyro_to_wheel(reach_imu, odom)
    rs_imu = apply_gyro_calibration(rs_imu, rs_calibration)
    reach_imu = apply_gyro_calibration(reach_imu, reach_calibration)

    raw_mag_heading, cal_mag_heading, _ = build_mag_heading_series(mag, odom)
    heading_alignments = {
        "wheel_heading": estimate_heading_offset_to_gt(
            distance_controls["t"].to_numpy(),
            distance_controls["wheel_heading"].to_numpy(),
            gt,
            float(distance_controls["t"].iloc[0]),
            args.heading_calibration_duration,
        ),
        "raw_mag_heading": estimate_heading_offset_to_gt(
            raw_mag_heading["t"].to_numpy(),
            raw_mag_heading["heading"].to_numpy(),
            gt,
            float(raw_mag_heading["t"].iloc[0]),
            args.heading_calibration_duration,
        ),
        "calibrated_mag_heading": estimate_heading_offset_to_gt(
            cal_mag_heading["t"].to_numpy(),
            cal_mag_heading["heading"].to_numpy(),
            gt,
            float(cal_mag_heading["t"].iloc[0]),
            args.heading_calibration_duration,
        ),
    }

    wheel_only = run_distance_wheel_baseline("Wheel-only Ackermann (/distance)", distance_controls, gt)
    ekf_rs = run_distance_ekf(
        "EKF + Realsense 6-axis + /distance",
        rs_imu,
        distance_controls,
        gt,
        heading_calibration_duration_s=args.heading_calibration_duration,
        wheel_yaw_noise=0.35,
    )
    ekf_reach_raw = run_distance_ekf(
        "EKF + Reach 9-axis raw mag + /distance",
        reach_imu,
        distance_controls,
        gt,
        mag_df=raw_mag_heading,
        heading_calibration_duration_s=args.heading_calibration_duration,
        wheel_yaw_noise=0.35,
        mag_yaw_noise=0.40,
    )
    ekf_reach_cal = run_distance_ekf(
        "EKF + Reach 9-axis calibrated mag + /distance",
        reach_imu,
        distance_controls,
        gt,
        mag_df=cal_mag_heading,
        heading_calibration_duration_s=args.heading_calibration_duration,
        wheel_yaw_noise=0.45,
        mag_yaw_noise=0.15,
    )

    trajectories = [wheel_only, ekf_rs, ekf_reach_raw, ekf_reach_cal]
    metrics = [{"method": traj.name, **evaluate_trajectory(traj, gt)} for traj in trajectories]
    metrics_df = save_metrics(metrics, output_dir / "distance_ekf_metrics.csv")

    plot_distance_trajectories(gt, trajectories, output_dir / "distance_ekf_trajectory_comparison.png")
    plot_yaw_errors(gt, trajectories, output_dir / "distance_ekf_yaw_error_comparison.png")
    plot_mag_calibration(raw_mag_heading, cal_mag_heading, output_dir / "distance_magnetometer_calibration.png")
    build_summary(
        distance_cal,
        rs_calibration,
        reach_calibration,
        heading_alignments,
        output_dir / "distance_experiment_summary.txt",
    )

    print("Bag:", bag_dir)
    print("Ground truth:", gt_path)
    print("Output directory:", output_dir)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
