import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from ahrs.filters import Madgwick


# ============================================================
# 1) Network Definition
# ============================================================
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size, stride=1, padding=padding
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(identity)

        out = self.relu(out + identity)
        return out


class AngleResNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(8, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = ResidualBlock1D(32, 32, kernel_size=3, stride=1)
        self.layer2 = ResidualBlock1D(32, 64, kernel_size=3, stride=1)

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.global_avg_pool(out).squeeze(-1)
        out = self.fc(out)
        return out


# ============================================================
# 2) Utility Functions
# ============================================================
def convert_row_index(v: int, row_mode: str) -> int:
    if row_mode == "pandas_index":
        return int(v)
    if row_mode == "excel_row_with_header":
        return int(v - 2)
    raise ValueError("row_mode must be 'pandas_index' or 'excel_row_with_header'")


def deg2rad_if_needed(x, unit: str):
    if unit.lower() in ["deg", "deg/s", "dps"]:
        return np.deg2rad(x)
    return x


def acc_to_mps2_if_needed(a, unit: str, g0=9.80665):
    if unit.lower() in ["g", "gee"]:
        return a * g0
    return a


def quat_gravity_body(q):
    w, x, y, z = q
    vx = 2.0 * (x * z - w * y)
    vy = 2.0 * (w * x + y * z)
    vz = w * w - x * x - y * y + z * z
    return np.array([vx, vy, vz], dtype=np.float64)


def integrate_xy(v, w, dt):
    n = len(v)
    x = np.zeros(n, dtype=np.float64)
    y = np.zeros(n, dtype=np.float64)
    psi = np.zeros(n, dtype=np.float64)

    for i in range(1, n):
        psi[i] = psi[i - 1] + w[i] * dt
        x[i] = x[i - 1] + v[i] * np.cos(psi[i]) * dt
        y[i] = y[i - 1] + v[i] * np.sin(psi[i]) * dt

    return x, y, psi


def mad_despike_and_smooth(x, despike_win=21, thresh=3.5, smooth_win=11):
    s = pd.Series(np.asarray(x, dtype=np.float64))
    med = s.rolling(despike_win, center=True, min_periods=1).median()
    mad = (s - med).abs().rolling(despike_win, center=True, min_periods=1).median()
    mad_sigma = 1.4826 * mad.to_numpy(dtype=np.float64)
    mad_sigma = np.maximum(mad_sigma, 1e-12)

    diff = s.to_numpy(dtype=np.float64) - med.to_numpy(dtype=np.float64)
    is_spike = np.abs(diff) > (thresh * mad_sigma)

    y = s.to_numpy(dtype=np.float64).copy()
    y[is_spike] = med.to_numpy(dtype=np.float64)[is_spike]

    y_s = (
        pd.Series(y)
        .rolling(smooth_win, center=True, min_periods=1)
        .mean()
        .to_numpy(dtype=np.float64)
    )
    return y_s

# ============================================================
# 3) Main Pipeline
# ============================================================
def main(cfg: dict):
    # Config
    xlsx_path = cfg["xlsx"]
    start_row = int(cfg["start"])
    end_row = int(cfg["end"])
    row_mode = cfg.get("row_mode", "pandas_index")
    model_ckpt = cfg.get("model_ckpt", "angle_resnet1d_best.pth")
    output_xlsx = cfg.get("output", "results.xlsx")

    imu_rate = float(cfg.get("imu_rate", 200.0))
    net_rate = float(cfg.get("net_rate", 50.0))
    win = int(cfg.get("window", 12))

    gyro_unit = cfg.get("gyro_unit", "deg")
    acc_unit = cfg.get("acc_unit", "m/s2")

    q_var = float(cfg.get("q_var", 0.5 ** 2))
    r_var = float(cfg.get("r_var", 0.8 ** 2))

    mad_despike_win = int(cfg.get("mad_despike_win", 21))
    mad_thresh = float(cfg.get("mad_thresh", 3.5))
    mad_smooth_win = int(cfg.get("mad_smooth_win", 11))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    dt_imu = 1.0 / imu_rate
    dt_net = 1.0 / net_rate

    df_feat = pd.read_excel(xlsx_path, sheet_name=3)
    df_truth = pd.read_excel(xlsx_path, sheet_name=1)
    df_imu = pd.read_excel(xlsx_path, sheet_name=2)

    start_idx = convert_row_index(start_row, row_mode)
    end_idx = convert_row_index(end_row, row_mode)
    K = end_idx - start_idx + 1

    feat_start = start_idx - (win - 1)
    feat_end = end_idx
    X_feat = df_feat.iloc[feat_start: feat_end + 1, :8].to_numpy(dtype=np.float32)
    X_windows = np.stack([X_feat[i: i + win, :] for i in range(K)], axis=0).astype(np.float32)

    v_gt = df_truth.iloc[start_idx: end_idx + 1, 4].to_numpy(dtype=np.float64)
    w_gt = df_truth.iloc[start_idx: end_idx + 1, 5].to_numpy(dtype=np.float64)  # rad/s assumed

    imu_start = start_idx * 4
    imu_len_expected = K * 4
    imu_end = imu_start + imu_len_expected - 1

    ax_raw = df_imu.iloc[imu_start: imu_end + 1, 2].to_numpy(dtype=np.float64)
    ay_raw = df_imu.iloc[imu_start: imu_end + 1, 3].to_numpy(dtype=np.float64)
    az_raw = df_imu.iloc[imu_start: imu_end + 1, 4].to_numpy(dtype=np.float64)
    gx_raw = df_imu.iloc[imu_start: imu_end + 1, 5].to_numpy(dtype=np.float64)
    gy_raw = df_imu.iloc[imu_start: imu_end + 1, 6].to_numpy(dtype=np.float64)
    gz_raw = df_imu.iloc[imu_start: imu_end + 1, 7].to_numpy(dtype=np.float64)

    gx_raw = deg2rad_if_needed(gx_raw, gyro_unit)
    gy_raw = deg2rad_if_needed(gy_raw, gyro_unit)
    gz_raw = deg2rad_if_needed(gz_raw, gyro_unit)
    ax_raw = acc_to_mps2_if_needed(ax_raw, acc_unit)
    ay_raw = acc_to_mps2_if_needed(ay_raw, acc_unit)
    az_raw = acc_to_mps2_if_needed(az_raw, acc_unit)

    N_imu = len(ax_raw)
    t_imu = np.arange(N_imu) * dt_imu
    t_50 = np.arange(K) * dt_net

    acc_b = np.column_stack([az_raw, ay_raw, ax_raw])
    gyr_b = np.column_stack([gz_raw, gy_raw, gx_raw])
    omega_imu = -gyr_b[:, 2].copy()

    ckpt = torch.load(model_ckpt, map_location=device, weights_only=False)
    model = AngleResNet1D().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    x_norm = bool(ckpt.get("X_NORM", ckpt.get("x_norm", False)))
    x_mean = ckpt.get("x_mean", None)
    x_std = ckpt.get("x_std", None)
    y_shift = float(ckpt.get("y_shift", 0.0))
    y_scale = float(ckpt.get("y_scale", 1.0))

    X_in = X_windows.astype(np.float32)
    if x_norm and x_mean is not None and x_std is not None:
        X_in = (X_in - np.asarray(x_mean, dtype=np.float32)) / np.asarray(x_std, dtype=np.float32)

    X_in_t = torch.from_numpy(X_in).to(device)
    with torch.no_grad():
        v_net_norm = model(X_in_t).cpu().numpy().squeeze(1)
    v_net = v_net_norm * y_scale + y_shift

    madgwick = Madgwick(sampleperiod=dt_imu)
    q = np.zeros((N_imu, 4), dtype=np.float64)
    q[0] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    for i in range(1, N_imu):
        qi = madgwick.updateIMU(q[i - 1], gyr=gyr_b[i], acc=acc_b[i])
        q[i] = qi if qi is not None else q[i - 1]

    g0 = 9.80665
    g_b = np.zeros_like(acc_b, dtype=np.float64)
    for i in range(N_imu):
        g_dir = quat_gravity_body(q[i])
        g_b[i] = g_dir * g0

    a_lin_b = acc_b - g_b
    a_fwd = a_lin_b[:, 0]

    # H) Kalman
    v_est = float(v_net[0])
    P = 1.0
    Q = float(q_var)
    R = float(r_var)
    v_fused_imu = np.zeros(N_imu, dtype=np.float64)
    # v_fused_50 isn't strictly needed for the final loop but kept for debug if needed
    v_fused_50 = np.zeros(K, dtype=np.float64)
    net_k = 0
    for i in range(N_imu):
        v_est = v_est + a_fwd[i] * dt_imu
        P = P + Q
        if (i % 4 == 3) and (net_k < K):
            z = float(v_net[net_k])
            Kgain = P / (P + R)
            v_est = v_est + Kgain * (z - v_est)
            P = (1.0 - Kgain) * P
            v_fused_50[net_k] = v_est
            net_k += 1
        v_fused_imu[i] = v_est

    v_final_imu = mad_despike_and_smooth(v_fused_imu, mad_despike_win, mad_thresh, mad_smooth_win)
    v_gt_mad_50 = mad_despike_and_smooth(v_gt, mad_despike_win, mad_thresh, mad_smooth_win)
    v_final_50 = v_final_imu[3::4]

    min_len = min(len(v_final_50), K)
    v_final_50 = v_final_50[:min_len]
    v_gt_mad_50 = v_gt_mad_50[:min_len]
    t_50 = t_50[:min_len]
    K = min_len

    v_imu_pure = np.zeros(N_imu, dtype=np.float64)
    v_imu_pure[0] = float(v_gt_mad_50[0])
    for i in range(1, N_imu):
        v_imu_pure[i] = v_imu_pure[i - 1] + a_fwd[i] * dt_imu

    # Trajectories
    x_kal, y_kal, psi_kal = integrate_xy(v_final_imu, omega_imu, dt_imu)
    x_imu, y_imu, psi_imu = integrate_xy(v_imu_pure, omega_imu, dt_imu)

    # GT Trajectory
    x_gt_50, y_gt_50, psi_gt_50 = integrate_xy(v_gt_mad_50, w_gt[:K], dt_net)
    x_gt_imu = np.interp(t_imu, t_50, x_gt_50)
    y_gt_imu = np.interp(t_imu, t_50, y_gt_50)

    # ============================================================
    # >>> Calculate Errors (200Hz)
    # ============================================================
    err_x_kal = x_kal - x_gt_imu
    err_y_kal = y_kal - y_gt_imu

    err_x_imu = x_imu - x_gt_imu
    err_y_imu = y_imu - y_gt_imu

    # ============================================================
    # Plot Trajectory
    # ============================================================
    plt.figure(figsize=(8, 7))
    plt.plot(x_kal, y_kal, label="Kalman (Raw Prediction)")
    plt.plot(x_gt_50, y_gt_50, label="GT")
    plt.scatter(0, 0, label="Start", color='red')
    plt.legend()
    plt.title("Trajectory Comparison (No GT Correction)")
    plt.axis('equal')
    plt.grid(True, linestyle="--")
    plt.show()

if __name__ == "__main__":
    cfg = {
        "xlsx": "processed_all_data_square.xlsx",
        "start": 1239,
        "end": 3239,
        "row_mode": "excel_row_with_header",
        "model_ckpt": "angle_resnet1d_best.pth",
        "output": "results_pure.xlsx",

        "imu_rate": 200.0,
        "net_rate": 50.0,
        "window": 12,

        "gyro_unit": "deg",
        "acc_unit": "m/s2",

        "q_var": 0.5 ** 2,
        "r_var": 0.8 ** 2,
        "gt_omega_is_deg": False,

        "mad_despike_win": 21,
        "mad_thresh": 3.5,
        "mad_smooth_win": 11,
    }
    main(cfg)