import pandas as pd
from matplotlib import pyplot as plt
import os
from scipy.fft import rfft, rfftfreq
import numpy as np
import yaml
from scipy.optimize import curve_fit


class Device:
    def __init__(self, path, sensors, flip=False):
        self.path = path
        self.df = pd.DataFrame()

        for f in os.listdir(path):
            if f.endswith(".csv"):
                filepath = os.path.join(path, f)
                df = pd.read_csv(filepath, delimiter=";")

                df = df[sensors + ["time"]]
                self.df = pd.concat([self.df, df], ignore_index=True)

        self.df["time"] = pd.to_datetime(
            self.df["time"], format="%Y/%m/%d %H:%M:%S:%f"
        )
        self.df = self.df.sort_values("time").reset_index(drop=True)
        if flip: 
            self.df[sensors] *= -1.0

        self.fs = self.compute_sampling_frequency(self.df["time"])
        self.sensors = sensors

        # ---- FIX: single temporal grouping key ----
        self.df["day_hour"] = self.df["time"].dt.floor("h")

    @staticmethod
    def compute_sampling_frequency(time_df):
        all_dts = time_df.diff().dt.total_seconds().iloc[1:]
        dt = all_dts.mode()[0]
        return 1 / dt

    @staticmethod
    def get_all_day_hours(df, min_ts=None, max_ts=None):
        ts = df["day_hour"].unique()
        if min_ts is not None:
            ts = ts[ts >= min_ts]
        if max_ts is not None:
            ts = ts[ts <= max_ts]
        return np.sort(ts)


# ================= CONFIG =================
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

os.makedirs("figures", exist_ok=True)

devices = {}
models = config.get("models", [])
for idx, model in enumerate(models):
    devices[model["name"]] = Device(
        model["data_folder"], model["sensors"], flip=model['flip_values']
    )



# =========================================================
# VISUALIZE 2 SECONDS OF DATA PER DAY-HOUR
# =========================================================
if config["analysis"].get("time_visualization"):
    n_seconds = 2
    os.makedirs("figures/2_seconds", exist_ok=True)

    for dev_name, dev in devices.items():
        os.makedirs(f"figures/2_seconds/{dev_name}", exist_ok=True)

        for dh in Device.get_all_day_hours(dev.df):
            df_h = dev.df[dev.df["day_hour"] == dh]

            if df_h.empty:
                continue

            for s in dev.sensors:
                n = int(n_seconds * dev.fs)
                t0 = df_h["time"].iloc[0]

                plt.plot(
                    (df_h["time"].iloc[:n] - t0).dt.total_seconds(),
                    df_h[s].iloc[:n],
                )

                plt.title(f"{dev_name} – {dh}")
                plt.xlabel("time [s]")
                plt.ylabel("values")
                plt.grid()
                dh_ts = pd.to_datetime(dh)
                dh_str = dh_ts.strftime("%Y-%m-%d_%H-%M")
                plt.savefig(
                    f"figures/2_seconds/{dev_name}/{dh_str}_sens_{s}.png"
                )
                plt.close()

# =========================================================
# OFFSET OVER TIME
# =========================================================
if config["analysis"].get("offset"):
    os.makedirs("figures/offset", exist_ok=True)

    for dev_name, dev in devices.items():
        os.makedirs(f"figures/offset/{dev_name}", exist_ok=True)

        out = (
            dev.df
            .groupby("day_hour")[dev.sensors]
            .mean()
            .reset_index()
        )

        for s in dev.sensors:
            plt.plot(out["day_hour"], out[s], "o--")

        plt.xlabel("time")
        plt.ylabel("offset")
        plt.grid()
        plt.tight_layout()
        plt.savefig(f"figures/offset/{dev_name}/offset.png")
        plt.close()

# =========================================================
# SPECTRAL ANALYSIS PER DAY-HOUR
# =========================================================
if config["analysis"].get("spectrum"):
    os.makedirs("figures/spectrum", exist_ok=True)

    for dev_name, dev in devices.items():
        os.makedirs(f"figures/spectrum/{dev_name}", exist_ok=True)

        for dh in Device.get_all_day_hours(dev.df):
            df_h = dev.df[dev.df["day_hour"] == dh]

            for s in dev.sensors:
                x = df_h[s].to_numpy()
                x = x - x.mean()

                X = rfft(x)
                freqs = rfftfreq(len(x), 1 / dev.fs)

                peak_freq = freqs[ np.argmax(np.abs(X)) ]

                plt.subplots(2, 1, 1)
                plt.plot(freqs, np.abs(X), alpha=0.6)
                plt.yscale("log")
                plt.xlim(0.5, 60)
                plt.ylim(1e-5, 1e5)
                plt.grid()

                plt.subplots(2, 1, 1)
                plt.plot(freqs, np.arg(X), alpha=0.6)
                plt.yscale("log")
                plt.xlim(0.5, 60)
                plt.grid()
                
                
                
                dh_ts = pd.to_datetime(dh)
                dh_str = dh_ts.strftime("%Y-%m-%d_%H-%M")
                plt.suptitle(f"{dev_name} - {dh_str} \nPeak freq. {peak_freq:.3f} Hz")
                plt.tight_layout()
                plt.savefig(
                    f"figures/spectrum/{dev_name}/{dh_str}_sens_{s}.png"
                )
                plt.close()

# =========================================================
# SINE FIT PER DAY-HOUR
# =========================================================
results = []
monitor_freq = 5
if config["analysis"].get("sine_fit"):
    os.makedirs("figures/sine_fit", exist_ok=True)

    for dev_name, dev in devices.items():
        os.makedirs(f"figures/sine_fit/{dev_name}", exist_ok=True)

        period = dev.fs / 5  # 5 Hz target
        n_periods = 10       # fit over 10 periods

        phases = {}
        for dh in Device.get_all_day_hours(dev.df):
            df_h = dev.df[dev.df["day_hour"] == dh]

            for s in dev.sensors:
                n = int(n_periods * period)
                n = min(n, len(df_h))
                if n < 5:
                    continue  # too few points

                t_raw = (df_h["time"].iloc[:n] - df_h["time"].iloc[0]).dt.total_seconds()
                y_raw = df_h[s].iloc[:n]

                # remove NaN and infinite
                mask = np.isfinite(t_raw) & np.isfinite(y_raw)
                t = t_raw[mask]
                y = y_raw[mask]

                if len(t) < 5:
                    continue

                # clip extreme outliers
                q1, q3 = np.percentile(y, [5, 95])
                y = np.clip(y, q1, q3)

                # sinusoidal model
                model = lambda x, A, B, C: A * np.sin(2 * np.pi * monitor_freq * (x + B)) + C

                # initial guess
                A_guess = (np.max(y) - np.min(y)) / 2
                B_guess = 0
                C_guess = np.mean(y)
                p0 = [A_guess, B_guess, C_guess]

                try:
                    p, _ = curve_fit(model, t, y, p0=p0, maxfev=20000)
                except Exception as e:
                    print(e)
                    continue

                # store results
                results.append({
                    "device": dev_name,
                    "sensor": s,
                    "day_hour": dh,
                    "vpp": np.abs(p[0]),
                    "phase": p[1],
                    "offset": p[2],
                })

                # ---- PLOT ----
                plt.figure(figsize=(8, 4))
                plt.plot(t, y, label="data")
                plt.plot(t, model(t, *p), label="fit", linestyle="--")
                plt.xlabel("time [s]")
                plt.ylabel("sensor value")
                plt.title(f"{dev_name} {s} {pd.to_datetime(dh).strftime('%Y-%m-%d %H:%M')} \nVpp: {np.abs(p[0]):.3f}, phase: {p[1]:.3f}, offset: {p[2]:.3f}")
                plt.legend()
                plt.grid()
                dh_str = pd.to_datetime(dh).strftime("%Y-%m-%d_%H-%M")
                plt.savefig(f"figures/sine_fit/{dev_name}/{dh_str}_sens_{s}.png")
                plt.close()

    results = pd.DataFrame(results)
    phase_delays = []
    for _, gr_df in results.groupby('day_hour'): 
        dev1_df = gr_df[gr_df['device'] == models[0]['name']]
        dev2_df = gr_df[gr_df['device'] == models[1]['name']]

        if dev1_df.empty or dev2_df.empty:
            continue

        # assume single sensor per device
        pd1 = dev1_df['phase'].values[0] * 1e3
        pd2 = dev2_df['phase'].values[0] * 1e3

        phase_delays.append(pd1 - pd2)

    # convert to numpy array for plotting
    phase_delays = np.array(phase_delays)

    plt.figure(figsize=(10, 5))
    plt.plot(phase_delays, "o--")
    plt.xlabel("hour index")
    plt.ylabel("phase delay [s]")
    plt.grid()
    plt.tight_layout()
    plt.savefig("figures/relative_phase_sin.png")
    plt.close()




####################################################################
############ PHASE DELAY BETWEEN THE MAP AND THE ACD ###############
# NON CORRETTO!!!!!!!!!!!!!!! bisogna considerare la fase iniziale!!!!!!!!!!!!!!! 

acd_df = devices["acd"].df.copy()
map_df = devices["map"].df.copy()
delays = []

if config.get("analysis").get("sine_fit") is True:
    n_seconds = 2

    for dh in Device.get_all_day_hours(dev.df):
        # select by day_hour
        map_df_h = map_df[map_df["day_hour"] == dh].copy()
        acd_df_h = acd_df[acd_df["day_hour"] == dh].copy()

        if map_df_h.empty or acd_df_h.empty:
            continue

        # enforce single-sensor assumption
        if len(devices["acd"].sensors) > 1 or len(devices["map"].sensors) > 1:
            print("!!! Attention !!!")
            print("Phase delay via FFT does not support multiple sensors yet.")
            continue
        acd_sensor = devices["acd"].sensors[0]
        map_sensor = devices["map"].sensors[0]

        # ---- CLEAN DATA ----
        acd_vals = acd_df_h[acd_sensor].to_numpy()
        map_vals = map_df_h[map_sensor].to_numpy()
        mask = np.isfinite(acd_vals) & np.isfinite(map_vals)
        if np.sum(mask) < 2:
            continue
        acd_vals = acd_vals[mask] - np.mean(acd_vals[mask])
        map_vals = map_vals[mask] - np.mean(map_vals[mask])

        # FFT
        X_acd = rfft(acd_vals)
        freqs_acd = rfftfreq(len(acd_vals), 1 / devices["acd"].fs)

        X_map = rfft(map_vals)
        freqs_map = rfftfreq(len(map_vals), 1 / devices["map"].fs)

        # compute phase at closest frequency to 5 Hz
        phase_acd = np.angle(X_acd[np.argmin(np.abs(freqs_acd - 5))]) / (2 * np.pi * 5.0) * 1e3
        phase_map = np.angle(X_map[np.argmin(np.abs(freqs_map - 5))]) / (2 * np.pi * 5.0) * 1e3

        phase_delay = phase_acd - phase_map
        delays.append(phase_delay)

# plot phase delays
plt.figure(figsize=(10, 5))
plt.plot(delays, "o--")
plt.xlabel("hour index")
plt.ylabel("phase delay [ms]")
plt.grid()
plt.tight_layout()
plt.savefig("figures/relative_phase_fft.png")
plt.close()


