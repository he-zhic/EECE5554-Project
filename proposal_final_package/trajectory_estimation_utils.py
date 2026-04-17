from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rosbags.highlevel import AnyReader


def wrap_angle_to_pi(angle: float | np.ndarray) -> float | np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def compute_yaw_from_quaternion_series(
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    qw: np.ndarray,
) -> np.ndarray:
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return np.unwrap(np.arctan2(siny_cosp, cosy_cosp))


@dataclass
class Trajectory:
    name: str
    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    yaw: np.ndarray


@dataclass
class ImuYawRateCalibration:
    axis: str
    sign: int
    scale: float
    bias: float
    corr: float


def find_default_dataset_paths() -> tuple[Path, Path]:
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


def load_ground_truth_trajectory(csv_path: Path) -> pd.DataFrame:
    gt = pd.read_csv(
        csv_path,
        sep=r"\s+",
        header=None,
        names=["t", "x", "y", "z", "qx", "qy", "qz", "qw"],
    )
    quat_yaw = compute_yaw_from_quaternion_series(gt["qx"], gt["qy"], gt["qz"], gt["qw"])
    dx = np.gradient(gt["x"].to_numpy(), gt["t"].to_numpy())
    dy = np.gradient(gt["y"].to_numpy(), gt["t"].to_numpy())
    gt["yaw_quat"] = quat_yaw
    gt["yaw"] = np.unwrap(np.arctan2(dy, dx))
    return gt


def estimate_heading_from_ground_truth_path(
    gt: pd.DataFrame,
    t: float,
    min_travel_m: float = 2.0,
) -> float:
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


def load_imu_odom_and_mag_topics(bag_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    odom_rows: list[dict[str, float]] = []
    reach_rows: list[dict[str, float]] = []
    mag_rows: list[dict[str, float]] = []

    targets = {"/odom", "/reach_1/imu", "/reach_1/imu/mag"}
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
    reach = pd.DataFrame(reach_rows).sort_values("t").reset_index(drop=True)
    mag = pd.DataFrame(mag_rows).sort_values("t").reset_index(drop=True)
    return odom, reach, mag


def load_distance_sensor_topic(bag_dir: Path) -> pd.DataFrame:
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


def fit_imu_yaw_rate_to_odom_yaw_rate(
    imu_df: pd.DataFrame,
    odom_df: pd.DataFrame,
) -> ImuYawRateCalibration:
    odom_t = odom_df["t"].to_numpy()
    wheel_yaw_rate = odom_df["vyaw"].to_numpy()
    best: ImuYawRateCalibration | None = None

    for axis in ("gyro_x", "gyro_y", "gyro_z"):
        raw = imu_df[axis].to_numpy()
        interp = np.interp(odom_t, imu_df["t"].to_numpy(), raw)
        for sign in (1, -1):
            signed = sign * interp
            corr = float(np.corrcoef(signed, wheel_yaw_rate)[0, 1])
            design = np.column_stack([signed, np.ones_like(signed)])
            scale, bias = np.linalg.lstsq(design, wheel_yaw_rate, rcond=None)[0]
            candidate = ImuYawRateCalibration(
                axis=axis,
                sign=sign,
                scale=float(scale),
                bias=float(bias),
                corr=corr,
            )
            if best is None or candidate.corr > best.corr:
                best = candidate

    assert best is not None
    return best


def add_calibrated_yaw_rate_to_imu(
    imu_df: pd.DataFrame,
    calibration: ImuYawRateCalibration,
) -> pd.DataFrame:
    out = imu_df.copy()
    signed = calibration.sign * out[calibration.axis].to_numpy()
    out["yaw_rate"] = calibration.scale * signed + calibration.bias
    return out


def integrate_odom_yaw_rate_to_heading(odom: pd.DataFrame) -> np.ndarray:
    t = odom["t"].to_numpy()
    yaw_rate = odom["vyaw"].to_numpy()
    heading = np.zeros_like(t)
    if len(t) > 1:
        heading[1:] = np.cumsum(0.5 * (yaw_rate[1:] + yaw_rate[:-1]) * np.diff(t))
    return heading


def orient_planar_heading_to_reference(
    heading_xy: np.ndarray,
    time: np.ndarray,
    ref_time: np.ndarray,
    ref_heading: np.ndarray,
) -> np.ndarray:
    best_heading: np.ndarray | None = None
    best_score = -np.inf
    base = ref_heading - ref_heading[0]

    for order in ((0, 1), (1, 0)):
        for sx in (1.0, -1.0):
            for sy in (1.0, -1.0):
                candidate = np.unwrap(np.arctan2(sy * heading_xy[:, order[1]], sx * heading_xy[:, order[0]]))
                interp = np.interp(ref_time, time, candidate)
                aligned = interp - interp[0]
                corr = float(np.corrcoef(aligned, base)[0, 1])
                if corr > best_score:
                    best_score = corr
                    best_heading = candidate

    assert best_heading is not None
    return best_heading


def estimate_heading_series_from_magnetometer(
    mag_df: pd.DataFrame,
    odom_df: pd.DataFrame,
) -> pd.DataFrame:
    samples = mag_df[["mx", "my", "mz"]].to_numpy()
    center = samples.mean(axis=0)
    centered = samples - center
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    plane_basis = vh[:2].T
    uv = centered @ plane_basis

    wheel_heading = integrate_odom_yaw_rate_to_heading(odom_df)
    raw_heading = orient_planar_heading_to_reference(
        uv,
        mag_df["t"].to_numpy(),
        odom_df["t"].to_numpy(),
        wheel_heading,
    )
    return pd.DataFrame({"t": mag_df["t"], "heading": raw_heading})


def build_distance_sensor_model_from_ground_truth(
    distance_df: pd.DataFrame,
    gt: pd.DataFrame,
    calibration_duration_s: float = 300.0,
    pulse_window: int = 15,
) -> pd.DataFrame:
    df = distance_df.copy()
    df["dt"] = df["t"].diff().fillna(0.0)
    df["dp"] = df["pulses"].diff().fillna(0.0)

    cal_end = float(df["t"].iloc[0] + calibration_duration_s)
    dist_mask = df["t"] <= cal_end
    gt_mask = gt["t"] <= cal_end

    gt_path = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(gt["x"].to_numpy()), np.diff(gt["y"].to_numpy())))])
    gt_yaw_rate = np.gradient(gt["yaw"].to_numpy(), gt["t"].to_numpy())

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
    wheel_heading[0] = estimate_heading_from_ground_truth_path(gt, float(df["t"].iloc[0]))
    for idx in range(1, len(df)):
        dt = float(df["t"].iloc[idx] - df["t"].iloc[idx - 1])
        wheel_heading[idx] = wheel_heading[idx - 1] + 0.5 * (
            df["wheel_yaw_rate"].iloc[idx] + df["wheel_yaw_rate"].iloc[idx - 1]
        ) * dt
    df["wheel_heading"] = wheel_heading
    return df


def run_wheel_odometry_dead_reckoning(
    name: str,
    distance_controls: pd.DataFrame,
    gt: pd.DataFrame,
) -> Trajectory:
    t = distance_controls["t"].to_numpy()
    delta_s = distance_controls["delta_s"].to_numpy()
    yaw_rate = distance_controls["wheel_yaw_rate"].to_numpy()

    x = np.zeros_like(t)
    y = np.zeros_like(t)
    yaw = np.zeros_like(t)
    x[0] = float(gt["x"].iloc[0])
    y[0] = float(gt["y"].iloc[0])
    yaw[0] = estimate_heading_from_ground_truth_path(gt, float(t[0]))

    for idx in range(1, len(t)):
        dt = float(t[idx] - t[idx - 1])
        mean_yaw_rate = 0.5 * (yaw_rate[idx] + yaw_rate[idx - 1])
        heading_mid = yaw[idx - 1] + 0.5 * mean_yaw_rate * dt
        yaw[idx] = yaw[idx - 1] + mean_yaw_rate * dt
        x[idx] = x[idx - 1] + delta_s[idx] * math.cos(heading_mid)
        y[idx] = y[idx - 1] + delta_s[idx] * math.sin(heading_mid)

    return Trajectory(name=name, t=t, x=x, y=y, yaw=np.unwrap(yaw))


def build_distance_sensor_model_from_odom(
    distance_df: pd.DataFrame,
    odom_df: pd.DataFrame,
    pulse_window: int = 15,
) -> pd.DataFrame:
    df = distance_df.copy()
    df["dt"] = df["t"].diff().fillna(0.0)
    df["dp"] = df["pulses"].diff().fillna(0.0)

    raw_pulse_rate = df["direction"] * df["dp"] / df["dt"].replace(0.0, np.nan)
    smoothed_rate = raw_pulse_rate.rolling(pulse_window, center=True, min_periods=1).mean().fillna(0.0)

    odom_path = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(odom_df["x"].to_numpy()), np.diff(odom_df["y"].to_numpy())))])
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
    return df


def extract_heading_series_from_imu_orientation(reach_imu: pd.DataFrame) -> pd.DataFrame:
    yaw = compute_yaw_from_quaternion_series(
        reach_imu["ori_qx"].to_numpy(),
        reach_imu["ori_qy"].to_numpy(),
        reach_imu["ori_qz"].to_numpy(),
        reach_imu["ori_qw"].to_numpy(),
    )
    return pd.DataFrame({"t": reach_imu["t"].to_numpy(), "heading": yaw})


def run_wheel_imu_ekf_in_local_frame(
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
        state[2] = float(wrap_angle_to_pi(yaw_prev + current_rate * dt))

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
            yaw_innovation = wrap_angle_to_pi(yaw_meas - state[2])
            yaw_s = yaw_h @ cov @ yaw_h.T + yaw_r
            yaw_k = cov @ yaw_h.T @ np.linalg.inv(yaw_s)
            state = state + (yaw_k.flatten() * yaw_innovation)
            state[2] = float(wrap_angle_to_pi(state[2]))
            cov = (np.eye(4) - yaw_k @ yaw_h) @ cov
            wheel_idx += 1

        while mag_idx < len(mag_t) and abs(mag_t[mag_idx] - next_t) < 1e-9:
            mag_h = np.array([[0.0, 0.0, 1.0, 0.0]])
            mag_r = np.array([[mag_yaw_noise**2]])
            mag_meas = float(mag_heading[mag_idx] + mag_offset)
            mag_innovation = wrap_angle_to_pi(mag_meas - state[2])
            mag_s = mag_h @ cov @ mag_h.T + mag_r
            mag_k = cov @ mag_h.T @ np.linalg.inv(mag_s)
            state = state + (mag_k.flatten() * mag_innovation)
            state[2] = float(wrap_angle_to_pi(state[2]))
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


def compute_trajectory_error_metrics_against_ground_truth(
    traj: Trajectory,
    gt: pd.DataFrame,
) -> dict[str, float]:
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
    yaw_err = wrap_angle_to_pi(pred_yaw - gt_yaw)
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
