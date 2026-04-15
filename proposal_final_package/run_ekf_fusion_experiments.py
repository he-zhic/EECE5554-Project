from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader


def wrap_angle(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def yaw_from_quaternion(qx: np.ndarray, qy: np.ndarray, qz: np.ndarray, qw: np.ndarray) -> np.ndarray:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.unwrap(np.arctan2(siny_cosp, cosy_cosp))


def resolve_default_paths() -> tuple[Path, Path]:
    project_dir = Path(__file__).resolve().parent
    search_roots = [project_dir, project_dir.parent]
    bag_candidates: list[Path] = []
    gt_candidates: list[Path] = []
    for root in search_roots:
        bag_candidates.extend(root.rglob("2023-12-26-15-10-15_ros2"))
        gt_candidates.extend(root.rglob("2023-12-26-15-10-15_mins_tum.csv"))
    bag_candidates = sorted(set(bag_candidates))
    gt_candidates = sorted(set(gt_candidates))
    if not bag_candidates or not gt_candidates:
        raise FileNotFoundError("Could not locate the Rosario ROS2 bag folder or MINS ground-truth CSV.")
    return bag_candidates[0], gt_candidates[0]


def load_ground_truth(csv_path: Path) -> pd.DataFrame:
    gt = pd.read_csv(
        csv_path,
        sep=r"\s+",
        header=None,
        names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"],
    )
    quat_yaw = yaw_from_quaternion(gt["qx"], gt["qy"], gt["qz"], gt["qw"])
    dx = np.gradient(gt["x"].to_numpy(), gt["t"].to_numpy())
    dy = np.gradient(gt["y"].to_numpy(), gt["t"].to_numpy())
    path_yaw = np.unwrap(np.arctan2(dy, dx))
    gt["yaw_quat"] = quat_yaw
    gt["yaw"] = path_yaw
    return gt


def estimate_heading_from_path(gt: pd.DataFrame, t: float, min_travel_m: float = 2.0) -> float:
    gt_t = gt["t"].to_numpy()
    gt_x = gt["x"].to_numpy()
    gt_y = gt["y"].to_numpy()

    x0 = float(np.interp(t, gt_t, gt_x))
    y0 = float(np.interp(t, gt_t, gt_y))
    start_idx = int(np.searchsorted(gt_t, t, side="left"))

    if start_idx >= len(gt_t) - 1:
        return float(np.interp(t, gt_t, gt["yaw"].to_numpy()))

    accum = 0.0
    prev_x = x0
    prev_y = y0
    for idx in range(start_idx + 1, len(gt_t)):
        curr_x = gt_x[idx]
        curr_y = gt_y[idx]
        accum += float(np.hypot(curr_x - prev_x, curr_y - prev_y))
        if accum >= min_travel_m:
            return float(np.arctan2(curr_y - y0, curr_x - x0))
        prev_x = curr_x
        prev_y = curr_y

    return float(np.interp(t, gt_t, gt["yaw"].to_numpy()))


def load_signals(bag_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    odom_rows: list[dict[str, float]] = []
    rs_rows: list[dict[str, float]] = []
    reach_rows: list[dict[str, float]] = []
    mag_rows: list[dict[str, float]] = []

    targets = {"/odom", "/realsense/imu", "/reach_1/imu", "/reach_1/imu/mag"}

    with AnyReader([bag_dir]) as reader:
        conns = [conn for conn in reader.connections if conn.topic in targets]
        for conn, timestamp, rawdata in reader.messages(conns):
            msg = reader.deserialize(rawdata, conn.msgtype)
            t = timestamp * 1e-9
            if conn.topic == "/odom":
                odom_rows.append(
                    {
                        "t": t,
                        "x": float(msg.pose.pose.position.x),
                        "y": float(msg.pose.pose.position.y),
                        "vx": float(msg.twist.twist.linear.x),
                        "vyaw": float(msg.twist.twist.angular.z),
                    }
                )
            elif conn.topic == "/realsense/imu":
                rs_rows.append(
                    {
                        "t": t,
                        "gyro_x": float(msg.angular_velocity.x),
                        "gyro_y": float(msg.angular_velocity.y),
                        "gyro_z": float(msg.angular_velocity.z),
                        "acc_x": float(msg.linear_acceleration.x),
                        "acc_y": float(msg.linear_acceleration.y),
                        "acc_z": float(msg.linear_acceleration.z),
                    }
                )
            elif conn.topic == "/reach_1/imu":
                q = msg.orientation
                reach_rows.append(
                    {
                        "t": t,
                        "gyro_x": float(msg.angular_velocity.x),
                        "gyro_y": float(msg.angular_velocity.y),
                        "gyro_z": float(msg.angular_velocity.z),
                        "acc_x": float(msg.linear_acceleration.x),
                        "acc_y": float(msg.linear_acceleration.y),
                        "acc_z": float(msg.linear_acceleration.z),
                        "ori_qx": float(q.x),
                        "ori_qy": float(q.y),
                        "ori_qz": float(q.z),
                        "ori_qw": float(q.w),
                    }
                )
            elif conn.topic == "/reach_1/imu/mag":
                mag = msg.magnetic_field
                mag_rows.append(
                    {
                        "t": t,
                        "mx": float(mag.x),
                        "my": float(mag.y),
                        "mz": float(mag.z),
                    }
                )

    odom = pd.DataFrame(odom_rows).sort_values("t").reset_index(drop=True)
    rs = pd.DataFrame(rs_rows).sort_values("t").reset_index(drop=True)
    reach = pd.DataFrame(reach_rows).sort_values("t").reset_index(drop=True)
    mag = pd.DataFrame(mag_rows).sort_values("t").reset_index(drop=True)
    return odom, rs, reach, mag


def integrate_wheel_heading(odom: pd.DataFrame) -> np.ndarray:
    t = odom["t"].to_numpy()
    yaw_rate = odom["vyaw"].to_numpy()
    heading = np.zeros_like(t)
    if len(t) > 1:
        heading[1:] = np.cumsum(0.5 * (yaw_rate[1:] + yaw_rate[:-1]) * np.diff(t))
    return heading


@dataclass
class GyroCalibration:
    axis: str
    sign: int
    scale: float
    bias: float
    corr: float


def calibrate_gyro_to_wheel(imu_df: pd.DataFrame, odom_df: pd.DataFrame) -> GyroCalibration:
    odom_t = odom_df["t"].to_numpy()
    wheel_yaw_rate = odom_df["vyaw"].to_numpy()
    best: GyroCalibration | None = None

    for axis in ("gyro_x", "gyro_y", "gyro_z"):
        raw = imu_df[axis].to_numpy()
        interp = np.interp(odom_t, imu_df["t"].to_numpy(), raw)
        for sign in (1, -1):
            signed = sign * interp
            corr = float(np.corrcoef(signed, wheel_yaw_rate)[0, 1])
            design = np.column_stack([signed, np.ones_like(signed)])
            scale, bias = np.linalg.lstsq(design, wheel_yaw_rate, rcond=None)[0]
            candidate = GyroCalibration(axis=axis, sign=sign, scale=float(scale), bias=float(bias), corr=corr)
            if best is None or candidate.corr > best.corr:
                best = candidate

    assert best is not None
    return best


def apply_gyro_calibration(imu_df: pd.DataFrame, calibration: GyroCalibration) -> pd.DataFrame:
    out = imu_df.copy()
    signed = calibration.sign * out[calibration.axis].to_numpy()
    out["yaw_rate"] = calibration.scale * signed + calibration.bias
    return out


def orient_heading_to_reference(heading: np.ndarray, time: np.ndarray, ref_time: np.ndarray, ref_heading: np.ndarray) -> np.ndarray:
    best_heading: np.ndarray | None = None
    best_score = -np.inf

    base = ref_heading - ref_heading[0]
    for order in ((0, 1), (1, 0)):
        for sx in (1.0, -1.0):
            for sy in (1.0, -1.0):
                candidate = np.unwrap(np.arctan2(sy * heading[:, order[1]], sx * heading[:, order[0]]))
                interp = np.interp(ref_time, time, candidate)
                aligned = interp - interp[0]
                corr = float(np.corrcoef(aligned, base)[0, 1])
                if corr > best_score:
                    best_score = corr
                    best_heading = candidate

    assert best_heading is not None
    return best_heading


@dataclass
class MagnetometerCalibration:
    center: np.ndarray
    plane_basis: np.ndarray
    scale: np.ndarray


def build_mag_heading_series(mag_df: pd.DataFrame, odom_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, MagnetometerCalibration]:
    samples = mag_df[["mx", "my", "mz"]].to_numpy()
    center = samples.mean(axis=0)
    centered = samples - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    plane_basis = vh[:2].T
    uv = centered @ plane_basis

    half_range = 0.5 * (uv.max(axis=0) - uv.min(axis=0))
    half_range[half_range == 0.0] = 1.0
    average_half_range = np.mean(half_range)
    scale = average_half_range / half_range
    uv_cal = uv * scale

    wheel_heading = integrate_wheel_heading(odom_df)
    raw_heading = orient_heading_to_reference(uv, mag_df["t"].to_numpy(), odom_df["t"].to_numpy(), wheel_heading)
    cal_heading = orient_heading_to_reference(uv_cal, mag_df["t"].to_numpy(), odom_df["t"].to_numpy(), wheel_heading)

    raw_df = pd.DataFrame({"t": mag_df["t"], "heading": raw_heading, "u": uv[:, 0], "v": uv[:, 1]})
    cal_df = pd.DataFrame({"t": mag_df["t"], "heading": cal_heading, "u": uv_cal[:, 0], "v": uv_cal[:, 1]})
    calibration = MagnetometerCalibration(center=center, plane_basis=plane_basis, scale=scale)
    return raw_df, cal_df, calibration


@dataclass
class Trajectory:
    name: str
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray


def interpolate_gt_state(gt: pd.DataFrame, t: float) -> tuple[float, float, float]:
    gt_t = gt["t"].to_numpy()
    x0 = float(np.interp(t, gt_t, gt["x"].to_numpy()))
    y0 = float(np.interp(t, gt_t, gt["y"].to_numpy()))
    yaw0 = estimate_heading_from_path(gt, t)
    return x0, y0, yaw0


def run_wheel_only_baseline(name: str, odom_df: pd.DataFrame, gt: pd.DataFrame) -> Trajectory:
    t = odom_df["t"].to_numpy()
    v = odom_df["vx"].to_numpy()
    yaw_rate = odom_df["vyaw"].to_numpy()

    x = np.zeros_like(t)
    y = np.zeros_like(t)
    yaw = np.zeros_like(t)

    x[0], y[0], yaw[0] = interpolate_gt_state(gt, float(t[0]))
    for idx in range(1, len(t)):
        dt = t[idx] - t[idx - 1]
        yaw[idx] = wrap_angle(yaw[idx - 1] + 0.5 * (yaw_rate[idx] + yaw_rate[idx - 1]) * dt)
        speed = 0.5 * (v[idx] + v[idx - 1])
        x[idx] = x[idx - 1] + speed * math.cos(yaw[idx - 1]) * dt
        y[idx] = y[idx - 1] + speed * math.sin(yaw[idx - 1]) * dt

    return Trajectory(name=name, t=t, x=x, y=y, yaw=np.unwrap(yaw))


def run_planar_ekf(
    name: str,
    imu_df: pd.DataFrame,
    odom_df: pd.DataFrame,
    gt: pd.DataFrame,
    mag_df: pd.DataFrame | None = None,
    speed_noise: float = 0.12,
    yaw_noise: float = 0.30,
    process_yaw_noise: float = 0.10,
    process_v_noise: float = 0.40,
) -> Trajectory:
    imu_t = imu_df["t"].to_numpy()
    imu_rate = imu_df["yaw_rate"].to_numpy()
    wheel_t = odom_df["t"].to_numpy()
    wheel_v = odom_df["vx"].to_numpy()

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

    x0, y0, yaw0 = interpolate_gt_state(gt, t0)
    v0 = float(np.interp(t0, wheel_t, wheel_v))
    current_rate = float(np.interp(t0, imu_t, imu_rate))
    mag_offset = 0.0
    if len(mag_t):
        mag_offset = yaw0 - float(np.interp(t0, mag_t, mag_yaw))

    state = np.array([x0, y0, yaw0, v0], dtype=float)
    cov = np.diag([0.05, 0.05, 0.10, 0.20]) ** 2

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

        dt = float(next_t - current_t)
        if dt < 0.0:
            dt = 0.0

        x_prev, y_prev, yaw_prev, v_prev = state
        state[0] = x_prev + v_prev * math.cos(yaw_prev) * dt
        state[1] = y_prev + v_prev * math.sin(yaw_prev) * dt
        state[2] = wrap_angle(yaw_prev + current_rate * dt)

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
            measurement = float(wheel_v[wheel_idx])
            h = np.array([[0.0, 0.0, 0.0, 1.0]])
            r = np.array([[speed_noise**2]])
            innovation = np.array([[measurement - state[3]]])
            s = h @ cov @ h.T + r
            k = cov @ h.T @ np.linalg.inv(s)
            state = state + (k @ innovation).ravel()
            cov = (np.eye(4) - k @ h) @ cov
            wheel_idx += 1

        while len(mag_t) and mag_idx < len(mag_t) and abs(mag_t[mag_idx] - next_t) < 1e-9:
            measurement = float(mag_yaw[mag_idx] + mag_offset)
            h = np.array([[0.0, 0.0, 1.0, 0.0]])
            r = np.array([[yaw_noise**2]])
            innovation = wrap_angle(measurement - state[2])
            s = h @ cov @ h.T + r
            k = cov @ h.T @ np.linalg.inv(s)
            state = state + (k.flatten() * innovation)
            state[2] = float(wrap_angle(state[2]))
            cov = (np.eye(4) - k @ h) @ cov
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


def evaluate_trajectory(traj: Trajectory, gt: pd.DataFrame) -> dict[str, float]:
    gt_t = gt["t"].to_numpy()
    mask = (gt_t >= traj.t[0]) & (gt_t <= traj.t[-1])
    eval_t = gt_t[mask]

    gt_x = gt.loc[mask, "x"].to_numpy()
    gt_y = gt.loc[mask, "y"].to_numpy()
    gt_yaw = gt.loc[mask, "yaw"].to_numpy()

    pred_x = np.interp(eval_t, traj.t, traj.x)
    pred_y = np.interp(eval_t, traj.t, traj.y)
    pred_yaw = np.interp(eval_t, traj.t, traj.yaw)

    pos_err = np.hypot(pred_x - gt_x, pred_y - gt_y)
    yaw_err = wrap_angle(pred_yaw - gt_yaw)
    yaw_err_deg = np.degrees(yaw_err)

    return {
        "samples": float(len(eval_t)),
        "duration_s": float(eval_t[-1] - eval_t[0]),
        "pos_rmse_m": float(np.sqrt(np.mean(pos_err**2))),
        "pos_mean_m": float(np.mean(pos_err)),
        "pos_max_m": float(np.max(pos_err)),
        "final_pos_err_m": float(pos_err[-1]),
        "yaw_rmse_rad": float(np.sqrt(np.mean(yaw_err**2))),
        "yaw_rmse_deg": float(np.degrees(np.sqrt(np.mean(yaw_err**2)))),
        "yaw_max_abs_deg": float(np.max(np.abs(yaw_err_deg))),
        "final_yaw_err_deg": float(np.degrees(yaw_err[-1])),
    }


def save_metrics(metrics: list[dict[str, float]], output_path: Path) -> pd.DataFrame:
    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(output_path, index=False)
    return metrics_df


def plot_trajectories(gt: pd.DataFrame, trajectories: list[Trajectory], output_path: Path) -> None:
    plt.figure(figsize=(10, 8))
    plt.plot(gt["x"], gt["y"], color="black", linewidth=2.2, label="MINS ground truth")

    colors = {
        "Wheel-only baseline": "#d62728",
        "EKF + Realsense 6-axis IMU": "#ff7f0e",
        "EKF + Reach 9-axis IMU (raw mag)": "#2ca02c",
        "EKF + Reach 9-axis IMU (calibrated mag)": "#1f77b4",
    }
    for traj in trajectories:
        plt.plot(traj.x, traj.y, linewidth=1.6, color=colors.get(traj.name, None), label=traj.name)

    plt.scatter(gt["x"].iloc[0], gt["y"].iloc[0], c="green", s=60, label="Start", zorder=5)
    plt.scatter(gt["x"].iloc[-1], gt["y"].iloc[-1], c="red", s=60, label="End", zorder=5)
    plt.xlabel("X / East (m)")
    plt.ylabel("Y / North (m)")
    plt.title("Rosario v2: Wheel-IMU EKF Trajectory Comparison")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_yaw_errors(gt: pd.DataFrame, trajectories: list[Trajectory], output_path: Path) -> None:
    plt.figure(figsize=(11, 6))
    gt_t = gt["t"].to_numpy()
    gt_yaw = gt["yaw"].to_numpy()

    for traj in trajectories:
        mask = (gt_t >= traj.t[0]) & (gt_t <= traj.t[-1])
        eval_t = gt_t[mask]
        pred_yaw = np.interp(eval_t, traj.t, traj.yaw)
        yaw_err_deg = np.degrees(wrap_angle(pred_yaw - gt_yaw[mask]))
        plt.plot(eval_t - eval_t[0], yaw_err_deg, linewidth=1.4, label=traj.name)

    plt.axhline(0.0, color="black", linewidth=1.0, alpha=0.6)
    plt.xlabel("Time since alignment start (s)")
    plt.ylabel("Yaw error (deg)")
    plt.title("Yaw Error vs Ground Truth")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_mag_calibration(raw_mag: pd.DataFrame, cal_mag: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    axes[0].scatter(raw_mag["u"], raw_mag["v"], s=4, alpha=0.35, color="#8c564b")
    axes[0].set_title("Raw projected magnetometer")
    axes[0].set_xlabel("u")
    axes[0].set_ylabel("v")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)

    axes[1].scatter(cal_mag["u"], cal_mag["v"], s=4, alpha=0.35, color="#1f77b4")
    axes[1].set_title("Ellipse-calibrated magnetometer")
    axes[1].set_xlabel("u_cal")
    axes[1].set_ylabel("v_cal")
    axes[1].axis("equal")
    axes[1].grid(True, alpha=0.3)

    fig.suptitle("Reach 9-axis magnetometer plane before/after calibration", fontsize=14)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_summary(
    rs_cal: GyroCalibration,
    reach_cal: GyroCalibration,
    mag_cal: MagnetometerCalibration,
    output_path: Path,
) -> None:
    lines = [
        "Selected gyro calibration against /odom yaw rate",
        f"Realsense: axis={rs_cal.axis}, sign={rs_cal.sign}, scale={rs_cal.scale:.6f}, bias={rs_cal.bias:.6f}, corr={rs_cal.corr:.4f}",
        f"Reach1: axis={reach_cal.axis}, sign={reach_cal.sign}, scale={reach_cal.scale:.6f}, bias={reach_cal.bias:.6f}, corr={reach_cal.corr:.4f}",
        "",
        "Magnetometer plane calibration",
        f"center = [{mag_cal.center[0]:.8e}, {mag_cal.center[1]:.8e}, {mag_cal.center[2]:.8e}]",
        f"plane basis row 0 = [{mag_cal.plane_basis[0, 0]:.8f}, {mag_cal.plane_basis[0, 1]:.8f}]",
        f"plane basis row 1 = [{mag_cal.plane_basis[1, 0]:.8f}, {mag_cal.plane_basis[1, 1]:.8f}]",
        f"plane basis row 2 = [{mag_cal.plane_basis[2, 0]:.8f}, {mag_cal.plane_basis[2, 1]:.8f}]",
        f"ellipse scale = [{mag_cal.scale[0]:.6f}, {mag_cal.scale[1]:.6f}]",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run wheel + IMU EKF fusion experiments on Rosario v2 data.")
    parser.add_argument("--bag-dir", type=Path, default=None, help="Path to the ROS2 bag directory.")
    parser.add_argument("--ground-truth", type=Path, default=None, help="Path to the MINS TUM CSV.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for plots and metrics.")
    args = parser.parse_args()

    default_bag, default_gt = resolve_default_paths()
    bag_dir = (args.bag_dir or default_bag).resolve()
    gt_path = (args.ground_truth or default_gt).resolve()
    output_dir = (args.output_dir or (Path(__file__).resolve().parent / "ekf_results")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    gt = load_ground_truth(gt_path)
    odom, rs_imu, reach_imu, mag = load_signals(bag_dir)

    rs_calibration = calibrate_gyro_to_wheel(rs_imu, odom)
    reach_calibration = calibrate_gyro_to_wheel(reach_imu, odom)
    rs_imu = apply_gyro_calibration(rs_imu, rs_calibration)
    reach_imu = apply_gyro_calibration(reach_imu, reach_calibration)

    raw_mag_heading, cal_mag_heading, mag_calibration = build_mag_heading_series(mag, odom)

    wheel_only = run_wheel_only_baseline("Wheel-only baseline", odom, gt)
    ekf_rs = run_planar_ekf("EKF + Realsense 6-axis IMU", rs_imu, odom, gt)
    ekf_reach_raw = run_planar_ekf(
        "EKF + Reach 9-axis IMU (raw mag)",
        reach_imu,
        odom,
        gt,
        mag_df=raw_mag_heading,
        yaw_noise=0.45,
    )
    ekf_reach_cal = run_planar_ekf(
        "EKF + Reach 9-axis IMU (calibrated mag)",
        reach_imu,
        odom,
        gt,
        mag_df=cal_mag_heading,
        yaw_noise=0.20,
    )

    trajectories = [wheel_only, ekf_rs, ekf_reach_raw, ekf_reach_cal]
    metrics = [{"method": traj.name, **evaluate_trajectory(traj, gt)} for traj in trajectories]
    metrics_df = save_metrics(metrics, output_dir / "ekf_metrics.csv")

    plot_trajectories(gt, trajectories, output_dir / "ekf_trajectory_comparison.png")
    plot_yaw_errors(gt, trajectories, output_dir / "ekf_yaw_error_comparison.png")
    plot_mag_calibration(raw_mag_heading, cal_mag_heading, output_dir / "magnetometer_calibration.png")
    build_summary(rs_calibration, reach_calibration, mag_calibration, output_dir / "experiment_summary.txt")

    print("Bag:", bag_dir)
    print("Ground truth:", gt_path)
    print("Output directory:", output_dir)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
