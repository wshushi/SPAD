import numpy as np
import math
import matplotlib.pyplot as plt
import os

# Safely import the custom SPAD model, or mock it if missing
try:
    from spad_model import spad_measure
except ImportError:
    print("Warning: spad_model not found. Using simulated SPAD noise.")
    def spad_measure(true_z, params):
        return true_z + np.random.normal(0, 0.05)

tick_hz = 700
dt      = 1.0 / tick_hz
sim_time = 20.0
total_ticks = int(sim_time * tick_hz)

GRAVITY = 9.81 # m/s^2

# C_LIGHT = 3e8   # speed of light [m/s]
C_LIGHT = 2.25e8 # speed of light in water [m/s]


spad_interval = int(tick_hz/7)
SPAD_DEFAULT_PARAMS = dict(
    T_HO           = 10e-9,
    PDP            = 0.30,
    DCR            = 1e3,
    N_pulses       = 600,
    T_window       = 400e-9,
    dt_bin         = 100e-12,
    lambda_sig_0   = 10000,
    pulse_sigma_s  = 0.42e-9,
    lambda_bg_rate = 20e6,
)

# see section 3.9 of datasheet for sampling rate
baro_interval = int(tick_hz/27)
def get_baro_reading(true_altitude):
    white_noise = np.random.normal(0, 0.11)
    BARO_FIXED_BIAS = np.random.normal(0, 0.66)
    raw_imu = true_altitude + white_noise + BARO_FIXED_BIAS
    return raw_imu

def get_true_reading(t, mode, offset, amplitude, omega, gradient=0.0, phase=0.0,t_turn=10.0, gradient2=-0.5):
    if mode == "SIN":
        z = offset + amplitude * math.sin(omega * t)
        a = -amplitude * (omega**2) * math.sin(omega * t)

    elif mode == "MSIN":
        amp2 = amplitude * 0.33
        omega2 = omega * 2.4
        z = offset + amplitude * math.sin(omega * t) + amp2 * math.cos(omega2 * t)
        a = -amplitude * (omega**2) * math.sin(omega * t) - amp2 * (omega2**2) * math.cos(omega2 * t)

    elif mode == "DIP":
        t0 = 20.0
        sigma = 2.0
        dt_maneuver = t - t0
        exp_term = math.exp(-(dt_maneuver**2) / (2 * sigma**2))
        z = offset - amplitude * exp_term
        a = (amplitude / sigma**2) * exp_term * (1.0 - (dt_maneuver**2) / sigma**2)

    elif mode == "UW":
        # 1. Calculate the theoretical trajectory
        z = (offset + gradient * t) + amplitude * math.sin(omega * t + phase)
        a = -amplitude * (omega**2) * math.sin(omega * t + phase)

        # 2. BOUNDARY CHECK: Prevent negative distance
        # A negative z means the sensor has crashed into the seabed.
        # This prevents the ToF (2 * z / C_LIGHT) from becoming negative and crashing the SPAD histogram array.
        if z <= 0.0:
            z = 1e-6  # Clamp to a near-zero positive value (1 micrometer)
            a = 0.0   # Acceleration stops because the vehicle/target cannot move further down
    elif mode == "UW2":
        # 1. Calculate the base trend (the seabed ridge)
        if t <= t_turn:
            # First leg: going up
            base_z = offset + gradient * t
        else:
            # Second leg: going down.
            # We must start at the exact height the first leg finished at to prevent a gap.
            peak_z = offset + gradient * t_turn
            base_z = peak_z + gradient2 * (t - t_turn)

        # 2. Add the seabed ripples (sine wave)
        z = base_z + amplitude * math.sin(omega * t + phase)

        # 3. Calculate acceleration
        # The second derivative of straight lines is 0, so 'a' only depends on the sine wave.
        a = -amplitude * (omega**2) * math.sin(omega * t + phase)

        # 4. BOUNDARY CHECK: Prevent negative distance (crashing into seabed)
        if z <= 0.0:
            z = 1e-6  # Clamp to a near-zero positive value
            a = 0.0   # Acceleration stops

    else:
        raise ValueError(f"Unknown flight profile mode: {mode}")

    return z, a

accel_interval = int(tick_hz/100)
def get_acc_reading(true_a):
    white_noise = np.random.normal(0, 0.015)
    ACCEL_FIXED_BIAS = np.random.normal(0, 0.2)
    return true_a + white_noise + ACCEL_FIXED_BIAS


def main():
    # --- 1. Initialize Data Storage Arrays ---
    t_history = []
    true_z_history = []
    true_a_history = []

    accel_history = []
    accel_t_history = []

    spad_history = []
    spad_t_history = []

    measure_t_history = []
    res_z_history = []
    res_v_history = []
    res_a_history = []

    # Simulation parameters
    offset      = 20.0
    amplitude   = 2
    omega = 1.0 # Lowered slightly for cleaner visual plots
    max_range  = offset + amplitude + 5.0
    spad_params = {**SPAD_DEFAULT_PARAMS,
                   'T_window': 2.0 * max_range / C_LIGHT}

    # --- State Initialization ---
    # State Vector X = [z, v, a]^T
    X = np.array([[offset],
                  [0.0],
                  [0.0]])

    # Initial Covariance (P) - High uncertainty at the start
    P = np.eye(3) * 100.0

    # State Transition Matrix (F)
    F = np.array([[1, dt, 0.5 * dt**2],
                  [0,  1, dt],
                  [0,  0,  1]])

    # Process Noise (Q) - Tuning parameters
    var_accel_process = 0.2
    Q = np.array([[0.1, 0, 0],
                  [0, var_accel_process * dt**2, var_accel_process * dt],
                  [0, var_accel_process * dt, var_accel_process]])

    # Measurement Noise Variances (R values)
    R_spad = 0.05 ** 2   # Variance of SPAD noise
    R_accel = 0.015 ** 2 # Variance of Accel noise

    # --- 2. Run the Simulation ---
    for tick in range(total_ticks):
        t = tick * dt

        # ==========================================
        # STEP A: SOURCE OF TRUTH
        # ==========================================
        true_z, true_a = get_true_reading(t, "UW", offset, amplitude, omega, -1.5)
        t_history.append(t)
        true_z_history.append(true_z)
        true_a_history.append(true_a)

        # ==========================================
        # STEP B: EKF PREDICT STEP
        # ==========================================
        X = F @ X
        P = F @ P @ F.T + Q

        # ==========================================
        # STEP C: EKF UPDATE STEP (Sensor Fusion)
        # ==========================================
        has_spad = (tick % spad_interval == 0)
        has_accel = (tick % accel_interval == 0)

        if has_spad or has_accel:
            H_rows = []
            Z_rows = []
            R_diag = []

            if has_spad:
                # SPAD measures Z (index 0)
                H_rows.append([1.0, 0.0, 0.0])
                spad_val = spad_measure(true_z, spad_params)
                Z_rows.append([spad_val])
                R_diag.append(R_spad)

                spad_t_history.append(t)
                spad_history.append(spad_val)

            if has_accel:
                # Accel measures A (index 2)
                H_rows.append([0.0, 0.0, 1.0])
                accel_val = get_acc_reading(true_a)
                Z_rows.append([accel_val])
                R_diag.append(R_accel)

                accel_t_history.append(t)
                accel_history.append(accel_val)

            # Convert to numpy arrays
            H = np.array(H_rows)
            Z = np.array(Z_rows)
            R = np.diag(R_diag)

            # Kalman Equations
            y = Z - (H @ X)
            S = H @ P @ H.T + R
            K = P @ H.T @ np.linalg.inv(S)
            X = X + (K @ y)
            I = np.eye(3)
            P = (I - K @ H) @ P

        # Store Estimated State for Plotting
        measure_t_history.append(t)
        res_z_history.append(X[0, 0])
        res_v_history.append(X[1, 0])
        res_a_history.append(X[2, 0])

    # --- 3. Generate Plots ---
    print("Simulation complete. Generating plots...")

    # Create 3 subplots: SPAD Altitude, Accelerometer, and Final Result
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Top Plot: Altitude (Truth vs SPAD)
    ax1.plot(t_history, true_z_history, 'k-', linewidth=2, label='True Depth')
    ax1.scatter(spad_t_history, spad_history, color='red', marker='x', s=60, label='SPAD Measurements')
    ax1.set_ylabel('Depth (m)')
    ax1.set_title('Sensor: Sparse SPAD Altitude Measurements')
    ax1.legend()
    ax1.grid(True)

    # Middle Plot: Acceleration (Truth vs Accelerometer)
    ax2.plot(t_history, true_a_history, 'k-', linewidth=2, label='True Acceleration')
    ax2.plot(accel_t_history, accel_history, 'g-', alpha=0.5, label='Accelerometer (BMI088)')
    ax2.set_ylabel('Acceleration (m/s^2)')
    ax2.set_title('Sensor: Noisy Accelerometer Readings')
    ax2.legend()
    ax2.grid(True)

    # Bottom Plot: The Kalman Filter Result
    ax3.plot(t_history, true_z_history, 'k-', linewidth=2, label='True Depth')
    ax3.plot(measure_t_history, res_z_history, 'b-', linewidth=2, alpha=0.8, label='EKF Estimated Depth')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Depth (m)')
    ax3.set_title('Result: Kalman Filter Fusion (SPAD + Accel)')
    ax3.legend()
    ax3.grid(True)

    plt.tight_layout()
    plt.savefig('low.png', dpi=150)
    print("Plot saved to output.png")
    os.startfile('low.png')  # Windows only

if __name__ == "__main__":
    main()
