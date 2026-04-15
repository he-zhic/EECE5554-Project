from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from run_ekf_distance_fusion_experiments import load_distance
from run_ekf_fusion_experiments import (
    Trajectory,
    apply_gyro_calibration,
    build_mag_heading_series,
    calibrate_gyro_to_wheel,
    evaluate_trajectory,
    load_ground_truth,
    load_signals,
    resolve_default_paths,
    wrap_angle,
    yaw_from_quaternion,
)


@dataclass
class SensorOnlyCalibration:
    meters_per_pulse: float
    yaw_gain: float
    yaw_bias: float
    gyro_axis: str
    gyro_sign: int
    gyro_scale: float
    gyro_bias: float
    pulse_window: int


@dataclass
class FixedOfflineParameters:
    pulses_per_turn: float
    wheel_circumference_m: float
    wheelbase_m: float
    steering_sign: float
    yaw_bias: float
    gyro_axis: str
    gyro_sign: int
    gyro_scale: float
    gyro_bias: float
    pulse_window: int


def build_distance_controls_sensor_only(
    distance_df: pd.DataFrame,
    odom_df: pd.DataFrame,
    pulse_window: int = 15,
) -> tuple[pd.DataFrame, SensorOnlyCalibration]:
    df = distance_df.copy()
    df["dt"] = df["t"].diff().fillna(0.0)
    df["dp"] = df["pulses"].diff().fillna(0.0)

    raw_pulse_rate = df["direction"] * df["dp"] / df["dt"].replace(0.0, np.nan)
    smoothed_rate = raw_pulse_rate.rolling(pulse_window, center=True, min_periods=1).mean().fillna(0.0)

    odom_path = np.concatenate(
        [[0.0], np.cumsum(np.hypot(np.diff(odom_df["x"].to_numpy()), np.diff(odom_df["y"].to_numpy())))]
    )
    pulse_path = np.concatenate([[0.0], np.cumsum(np.abs(df["dp"].to_numpy()[1:]))])
    pulse_delta = float(pulse_path[-1] - pulse_path[0])
    odom_delta = float(odom_path[-1] - odom_path[0])
    if abs(pulse_delta) < 1e-9:
        raise ValueError("Pulse count did not change enough to estimate meters_per_pulse.")

    meters_per_pulse = odom_delta / pulse_delta
    df["speed"] = meters_per_pulse * smoothed_rate
    df["delta_s"] = meters_per_pulse * df["direction"] * df["dp"]

    steering_base = df["speed"] * np.tan(df["angle"])
    odom_dx = np.gradient(odom_df["x"].to_numpy(), odom_df["t"].to_numpy())
    odom_dy = np.gradient(odom_df["y"].to_numpy(), odom_df["t"].to_numpy())
    odom_heading = np.unwrap(np.arctan2(odom_dy, odom_dx))
    odom_yaw_rate = np.gradient(odom_heading, odom_df["t"].to_numpy())
    interp_yaw_rate = np.interp(df["t"], odom_df["t"], odom_yaw_rate)
    valid = np.isfinite(steering_base) & np.isfinite(interp_yaw_rate) & (np.abs(steering_base) > 1e-4)
    if int(valid.sum()) < 10:
        raise ValueError("Not enough valid samples to estimate the steering-to-yaw mapping from /odom.")

    design = np.column_stack([steering_base[valid], np.ones(int(valid.sum()))])
    yaw_gain, yaw_bias = np.linalg.lstsq(design, interp_yaw_rate[valid], rcond=None)[0]
    df["wheel_yaw_rate"] = yaw_gain * steering_base + yaw_bias

    wheel_heading = np.zeros(len(df), dtype=float)
    for idx in range(1, len(df)):
        dt = float(df["t"].iloc[idx] - df["t"].iloc[idx - 1])
        wheel_heading[idx] = wheel_heading[idx - 1] + 0.5 * (
            df["wheel_yaw_rate"].iloc[idx] + df["wheel_yaw_rate"].iloc[idx - 1]
        ) * dt
    df["wheel_heading"] = wheel_heading

    return df, SensorOnlyCalibration(
        meters_per_pulse=float(meters_per_pulse),
        yaw_gain=float(yaw_gain),
        yaw_bias=float(yaw_bias),
        gyro_axis="",
        gyro_sign=1,
        gyro_scale=1.0,
        gyro_bias=0.0,
        pulse_window=pulse_window,
    )


def apply_fixed_gyro_calibration(imu_df: pd.DataFrame, params: FixedOfflineParameters) -> pd.DataFrame:
    out = imu_df.copy()
    signed = params.gyro_sign * out[params.gyro_axis].to_numpy()
    out["yaw_rate"] = params.gyro_scale * signed + params.gyro_bias
    return out


def build_distance_controls_fixed(
    distance_df: pd.DataFrame,
    params: FixedOfflineParameters,
) -> pd.DataFrame:
    df = distance_df.copy()
    df["dt"] = df["t"].diff().fillna(0.0)
    df["dp"] = df["pulses"].diff().fillna(0.0)

    raw_pulse_rate = df["direction"] * df["dp"] / df["dt"].replace(0.0, np.nan)
    smoothed_rate = raw_pulse_rate.rolling(params.pulse_window, center=True, min_periods=1).mean().fillna(0.0)

    meters_per_pulse = params.wheel_circumference_m / params.pulses_per_turn
    df["speed"] = meters_per_pulse * smoothed_rate
    df["delta_s"] = meters_per_pulse * df["direction"] * df["dp"]

    steering_base = df["speed"] * np.tan(df["angle"])
    df["wheel_yaw_rate"] = params.steering_sign * steering_base / params.wheelbase_m + params.yaw_bias

    wheel_heading = np.zeros(len(df), dtype=float)
    for idx in range(1, len(df)):
        dt = float(df["t"].iloc[idx] - df["t"].iloc[idx - 1])
        wheel_heading[idx] = wheel_heading[idx - 1] + 0.5 * (
            df["wheel_yaw_rate"].iloc[idx] + df["wheel_yaw_rate"].iloc[idx - 1]
        ) * dt
    df["wheel_heading"] = wheel_heading
    return df


def build_imu_heading_series(reach_imu: pd.DataFrame) -> pd.DataFrame:
    yaw = yaw_from_quaternion(
        reach_imu["ori_qx"].to_numpy(),
        reach_imu["ori_qy"].to_numpy(),
        reach_imu["ori_qz"].to_numpy(),
        reach_imu["ori_qw"].to_numpy(),
    )
    return pd.DataFrame({"t": reach_imu["t"].to_numpy(), "heading": yaw})


def run_local_frame_ekf(
    name: str,
    imu_df: pd.DataFrame,
    distance_controls: pd.DataFrame,
    init_heading_df: pd.DataFrame,
    mag_df: pd.DataFrame,
    speed_noise: float = 0.30,
    wheel_yaw_noise: float = 0.30,
    mag_yaw_noise: float = 0.08,
    process_yaw_noise: float = 0.08,
    process_v_noise: float = 0.25,
) -> Trajectory:
    imu_t = imu_df["t"].to_numpy()
    imu_rate = imu_df["yaw_rate"].to_numpy()
    wheel_t = distance_controls["t"].to_numpy()
    wheel_v = distance_controls["speed"].to_numpy()
    wheel_heading = distance_controls["wheel_heading"].to_numpy()
    init_t = init_heading_df["t"].to_numpy()
    init_heading = init_heading_df["heading"].to_numpy()
    mag_t = mag_df["t"].to_numpy()
    mag_heading = mag_df["heading"].to_numpy()

    t0 = max(float(imu_t[0]), float(wheel_t[0]), float(init_t[0]), float(mag_t[0]))
    t_end = min(float(imu_t[-1]), float(wheel_t[-1]), float(init_t[-1]), float(mag_t[-1]))
    if t_end <= t0:
        raise ValueError(f"No overlapping interval for {name}.")

    yaw0 = float(np.interp(t0, init_t, init_heading))
    state = np.array([0.0, 0.0, yaw0, float(np.interp(t0, wheel_t, wheel_v))], dtype=float)
    cov = np.diag([0.05, 0.05, 0.12, 0.25]) ** 2
    current_rate = float(np.interp(t0, imu_t, imu_rate))

    # Local-frame initialization: start at (0,0) and use only IMU heading at t0.
    wheel_heading_offset = yaw0 - float(np.interp(t0, wheel_t, wheel_heading))
    mag_offset = yaw0 - float(np.interp(t0, mag_t, mag_heading))

    imu_idx = int(np.searchsorted(imu_t, t0, side="right"))
    wheel_idx = int(np.searchsorted(wheel_t, t0, side="left"))
    mag_idx = int(np.searchsorted(mag_t, t0, side="left"))

    t_hist = [t0]
    x_hist = [state[0]]
    y_hist = [state[1]]
    yaw_hist = [state[2]]
    current_t = t0

    while True:
        next_imu = imu_t[imu_idx] if imu_idx < len(imu_t) else np.inf
        next_wheel = wheel_t[wheel_idx] if wheel_idx < len(wheel_t) else np.inf
        next_mag = mag_t[mag_idx] if mag_idx < len(mag_t) else np.inf
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
            ],
            dtype=float,
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
            speed_h = np.array([[0.0, 0.0, 0.0, 1.0]])
            speed_r = np.array([[speed_noise**2]])
            speed_innovation = np.array([[float(wheel_v[wheel_idx]) - state[3]]])
            speed_s = speed_h @ cov @ speed_h.T + speed_r
            speed_k = cov @ speed_h.T @ np.linalg.inv(speed_s)
            state = state + (speed_k @ speed_innovation).ravel()
            cov = (np.eye(4) - speed_k @ speed_h) @ cov

            yaw_h = np.array([[0.0, 0.0, 1.0, 0.0]])
            yaw_r = np.array([[wheel_yaw_noise**2]])
            yaw_meas = float(wheel_heading[wheel_idx] + wheel_heading_offset)
            yaw_innovation = wrap_angle(yaw_meas - state[2])
            yaw_s = yaw_h @ cov @ yaw_h.T + yaw_r
            yaw_k = cov @ yaw_h.T @ np.linalg.inv(yaw_s)
            state = state + (yaw_k.flatten() * yaw_innovation)
            state[2] = float(wrap_angle(state[2]))
            cov = (np.eye(4) - yaw_k @ yaw_h) @ cov
            wheel_idx += 1

        while mag_idx < len(mag_t) and abs(mag_t[mag_idx] - next_t) < 1e-9:
            mag_h = np.array([[0.0, 0.0, 1.0, 0.0]])
            mag_r = np.array([[mag_yaw_noise**2]])
            mag_meas = float(mag_heading[mag_idx] + mag_offset)
            mag_innovation = wrap_angle(mag_meas - state[2])
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


def shift_ground_truth_to_origin(gt: pd.DataFrame, t0: float) -> pd.DataFrame:
    out = gt.copy()
    x0 = float(np.interp(t0, out["t"], out["x"]))
    y0 = float(np.interp(t0, out["t"], out["y"]))
    out["x"] = out["x"] - x0
    out["y"] = out["y"] - y0
    return out


def plot_vs_ground_truth(
    gt: pd.DataFrame,
    traj: Trajectory,
    output_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    gt_x = gt["x"].to_numpy()
    gt_y = gt["y"].to_numpy()

    for ax in axes:
        ax.plot(gt_x, gt_y, color="black", linewidth=2.4, label="Ground Truth (translated to start at origin)")
        ax.plot(traj.x, traj.y, linewidth=2.0, color="#d62728", label=traj.name)
        ax.scatter(0.0, 0.0, c="green", s=70, label="Start", zorder=5)
        ax.scatter(traj.x[-1], traj.y[-1], c="red", s=70, label="End", zorder=5)
        ax.set_xlabel("X / East-like (m)")
        ax.set_ylabel("Y / North-like (m)")
        ax.grid(True, alpha=0.3)

    margin = 15.0
    axes[1].set_xlim(float(gt_x.min() - margin), float(gt_x.max() + margin))
    axes[1].set_ylim(float(gt_y.min() - margin), float(gt_y.max() + margin))

    axes[0].set_title(title)
    axes[1].set_title("Zoom Near Ground Truth")
    axes[0].set_aspect("equal", adjustable="box")
    axes[1].set_aspect("auto")
    axes[0].legend(loc="best")
    axes[1].legend(loc="best")
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_summary(
    output_path: Path,
    sensor_metrics: dict[str, float],
    fixed_metrics: dict[str, float],
    sensor_cal: SensorOnlyCalibration,
    fixed_params: FixedOfflineParameters,
    mag_mode: str,
) -> None:
    lines = [
        "Generalized EKF figures",
        "Both figures start at (0,0) and use IMU heading for initialization.",
        "Ground Truth is translated to the same origin only for plotting and evaluation.",
        "No GT heading alignment is used in either figure.",
        f"Magnetometer update source = {mag_mode}",
        "",
        "Figure 1: Fully GT-free EKF",
        "Wheel scale, wheel yaw mapping, and gyro yaw-rate calibration are estimated only from IMU + wheel data on this bag.",
        f"meters_per_pulse = {sensor_cal.meters_per_pulse:.10f}",
        f"yaw_gain = {sensor_cal.yaw_gain:.10f}",
        f"yaw_bias = {sensor_cal.yaw_bias:.10f}",
        f"gyro_axis = {sensor_cal.gyro_axis}",
        f"gyro_sign = {sensor_cal.gyro_sign}",
        f"gyro_scale = {sensor_cal.gyro_scale:.10f}",
        f"gyro_bias = {sensor_cal.gyro_bias:.10f}",
        "",
        "Figure 2: Fixed-parameter EKF",
        "Wheel and gyro parameters are frozen as one-time offline constants and reused without per-bag re-estimation.",
        f"pulses_per_turn = {fixed_params.pulses_per_turn:.1f}",
        f"wheel_circumference_m = {fixed_params.wheel_circumference_m:.10f}",
        f"wheelbase_m = {fixed_params.wheelbase_m:.10f}",
        f"steering_sign = {fixed_params.steering_sign:.1f}",
        f"yaw_bias = {fixed_params.yaw_bias:.10f}",
        f"gyro_axis = {fixed_params.gyro_axis}",
        f"gyro_sign = {fixed_params.gyro_sign}",
        f"gyro_scale = {fixed_params.gyro_scale:.10f}",
        f"gyro_bias = {fixed_params.gyro_bias:.10f}",
        "",
        "Metrics",
        (
            "Fully GT-free EKF: "
            f"pos_rmse_m={sensor_metrics['pos_rmse_m']:.3f}, "
            f"final_pos_err_m={sensor_metrics['final_pos_err_m']:.3f}, "
            f"yaw_rmse_deg={sensor_metrics['yaw_rmse_deg']:.3f}"
        ),
        (
            "Fixed-parameter EKF: "
            f"pos_rmse_m={fixed_metrics['pos_rmse_m']:.3f}, "
            f"final_pos_err_m={fixed_metrics['final_pos_err_m']:.3f}, "
            f"yaw_rmse_deg={fixed_metrics['yaw_rmse_deg']:.3f}"
        ),
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate more general EKF-vs-ground-truth figures without GT heading initialization.")
    parser.add_argument("--bag-dir", type=Path, default=None, help="Path to the ROS2 bag directory.")
    parser.add_argument("--ground-truth", type=Path, default=None, help="Path to the MINS TUM CSV.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for the output figures.")
    parser.add_argument("--pulse-window", type=int, default=15, help="Rolling window for pulse-rate smoothing.")
    parser.add_argument("--mag-mode", choices=("raw", "cal"), default="raw", help="Which magnetometer heading series to use.")
    args = parser.parse_args()

    default_bag, default_gt = resolve_default_paths()
    bag_dir = (args.bag_dir or default_bag).resolve()
    gt_path = (args.ground_truth or default_gt).resolve()
    output_dir = (args.output_dir or (Path(__file__).resolve().parent / "proposal_results")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gt = load_ground_truth(gt_path)
    odom, _, reach_imu, mag = load_signals(bag_dir)
    distance_df = load_distance(bag_dir)
    init_heading_df = build_imu_heading_series(reach_imu)

    raw_mag_heading, cal_mag_heading, _ = build_mag_heading_series(mag, odom)
    mag_heading = cal_mag_heading if args.mag_mode == "cal" else raw_mag_heading

    # Variant 1: fully GT-free after evaluation setup. Initialization uses only IMU heading and origin.
    reach_sensor_cal = calibrate_gyro_to_wheel(reach_imu, odom)
    reach_imu_sensor = apply_gyro_calibration(reach_imu, reach_sensor_cal)
    sensor_controls, sensor_cal = build_distance_controls_sensor_only(distance_df, odom, pulse_window=args.pulse_window)
    sensor_cal.gyro_axis = reach_sensor_cal.axis
    sensor_cal.gyro_sign = reach_sensor_cal.sign
    sensor_cal.gyro_scale = reach_sensor_cal.scale
    sensor_cal.gyro_bias = reach_sensor_cal.bias

    fully_gt_free_traj = run_local_frame_ekf(
        "Fully GT-Free EKF Fusion (IMU + Wheel Odometry)",
        reach_imu_sensor,
        sensor_controls,
        init_heading_df,
        mag_heading,
    )
    gt_for_fully_gt_free = shift_ground_truth_to_origin(gt, float(fully_gt_free_traj.t[0]))
    fully_gt_free_metrics = evaluate_trajectory(fully_gt_free_traj, gt_for_fully_gt_free)

    # Variant 2: freeze wheel + gyro constants as one-time offline parameters and reuse them.
    fixed_params = FixedOfflineParameters(
        pulses_per_turn=46.0,
        wheel_circumference_m=46.0 * 0.0371315668,
        wheelbase_m=1.0 / abs(-0.6110833579),
        steering_sign=-1.0,
        yaw_bias=-0.0161618484,
        gyro_axis="gyro_z",
        gyro_sign=-1,
        gyro_scale=0.750160,
        gyro_bias=-0.015767,
        pulse_window=args.pulse_window,
    )
    reach_imu_fixed = apply_fixed_gyro_calibration(reach_imu, fixed_params)
    fixed_controls = build_distance_controls_fixed(distance_df, fixed_params)
    fixed_param_traj = run_local_frame_ekf(
        "Fixed-Parameter EKF Fusion (IMU + Wheel Odometry)",
        reach_imu_fixed,
        fixed_controls,
        init_heading_df,
        mag_heading,
    )
    gt_for_fixed = shift_ground_truth_to_origin(gt, float(fixed_param_traj.t[0]))
    fixed_param_metrics = evaluate_trajectory(fixed_param_traj, gt_for_fixed)

    metrics_df = pd.DataFrame(
        [
            {"method": fully_gt_free_traj.name, "mag_mode": args.mag_mode, **fully_gt_free_metrics},
            {"method": fixed_param_traj.name, "mag_mode": args.mag_mode, **fixed_param_metrics},
        ]
    )
    metrics_path = output_dir / "generalized_ekf_metrics.csv"
    summary_path = output_dir / "generalized_ekf_summary.txt"
    fully_gt_free_fig = output_dir / "fully_gt_free_ekf_vs_ground_truth.png"
    fixed_param_fig = output_dir / "fixed_parameter_ekf_vs_ground_truth.png"

    metrics_df.to_csv(metrics_path, index=False)
    plot_vs_ground_truth(
        gt_for_fully_gt_free,
        fully_gt_free_traj,
        fully_gt_free_fig,
        "Fully GT-Free EKF Fusion vs Ground Truth",
    )
    plot_vs_ground_truth(
        gt_for_fixed,
        fixed_param_traj,
        fixed_param_fig,
        "Fixed-Parameter EKF Fusion vs Ground Truth",
    )
    save_summary(
        summary_path,
        fully_gt_free_metrics,
        fixed_param_metrics,
        sensor_cal,
        fixed_params,
        args.mag_mode,
    )

    print("Bag:", bag_dir)
    print("Ground truth:", gt_path)
    print("Output directory:", output_dir)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
