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
            print(f"Reading file {path}/{f}")
            if f.endswith(".csv"):
                filepath = os.path.join(path, f)
                df = pd.read_csv(filepath, delimiter=";")

                df = df[sensors + ["time"]]
                self.df = pd.concat([self.df, df], ignore_index=True)

        self.df["time"] = pd.to_datetime(self.df["time"], format="%Y/%m/%d %H:%M:%S:%f")
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

chunk_time = config["chunk_time"]

devices = {}
models = config.get("models", [])
for idx, model in enumerate(models):
    devices[model["name"]] = Device(
        model["data_folder"], model["sensors"], flip=model["flip_values"]
    )

    devices[model["name"]].df.index = pd.to_datetime(
        devices[model["name"]].df["time"].copy(), utc=True
    )
    devices[model["name"]].df.sort_index()

# =========================================================
# VISUALIZE 2 SECONDS OF DATA PER DAY-HOUR
# =========================================================
if config["analysis"].get("time_visualization"):
    n_seconds = 2
    os.makedirs("figures/2_seconds", exist_ok=True)

    for dev_name, dev in devices.items():
        os.makedirs(f"figures/2_seconds/{dev_name}", exist_ok=True)

        for gr_idx, df_h in dev.df.groupby(pd.Grouper(freq=chunk_time)):

            if df_h.empty:
                continue

            for s in dev.sensors:
                n = int(n_seconds * dev.fs)
                t0 = df_h["time"].iloc[0]

                plt.plot(
                    (df_h["time"].iloc[:n] - t0).dt.total_seconds(),
                    df_h[s].iloc[:n],
                )

                plt.title(f"{dev_name} – {gr_idx}")
                plt.xlabel("time [s]")
                plt.ylabel("values")
                plt.grid()
                dh_ts = pd.to_datetime(gr_idx)
                dh_str = dh_ts.strftime("%Y-%m-%d_%H-%M")
                plt.savefig(f"figures/2_seconds/{dev_name}/{dh_str}_sens_{s}.png")
                plt.close()

"""for dev in devices.values():
    dev.minutes_per_hour = config['analysis']['minutes_per_hour']
    dev.points_per_hour = int(dev.fs * dev.minutes_per_hour * 60)"""
# =========================================================
# OFFSET OVER TIME
# =========================================================
if config["analysis"].get("offset"):
    os.makedirs("figures/offset", exist_ok=True)

    for dev_name, dev in devices.items():
        os.makedirs(f"figures/offset/{dev_name}", exist_ok=True)

        for s in dev.sensors:
            to_plot = [
                [hr_idx, hr_df[s].mean()]
                for hr_idx, hr_df in dev.df[[s, "day_hour"]].groupby(
                    pd.Grouper(freq=chunk_time)
                )
            ]
            to_plot = np.array(to_plot)

            plt.plot(to_plot[:, 0], to_plot[:, 1], "o--")

            plt.xlabel("time")
            plt.ylabel("offset")
            plt.grid()
            plt.tight_layout()
            plt.savefig(f"figures/offset/{dev_name}/sensor_{s}_offset.png")
            plt.close()

# =========================================================
# SPECTRAL ANALYSIS PER DAY-HOUR
# =========================================================
if config["analysis"].get("spectrum"):
    os.makedirs("figures/spectrum", exist_ok=True)

    for dev_name, dev in devices.items():
        os.makedirs(f"figures/spectrum/{dev_name}", exist_ok=True)

        for gr_idx, df_h in dev.df.groupby(pd.Grouper(freq=chunk_time)):

            for s in dev.sensors:
                x = df_h[s].to_numpy()
                x = x - x.mean()

                X = rfft(x)
                freqs = rfftfreq(len(x), 1 / dev.fs)

                peak_freq = freqs[np.argmax(np.abs(X))]

                plt.subplot(2, 1, 1)
                plt.plot(freqs, np.abs(X), alpha=0.6)
                plt.yscale("log")
                plt.xlim(0.5, 60)
                plt.ylim(1e-5, 1e5)
                plt.grid()

                plt.subplot(2, 1, 2)
                plt.plot(freqs, np.angle(X), alpha=0.6)
                # plt.yscale("log")
                plt.xlim(0.5, 60)
                plt.grid()

                plt.suptitle(
                    f"{dev_name} - {gr_idx.strftime("%Y-%m-%d_%H-%M")} \nPeak freq. {peak_freq:.3f} Hz"
                )
                plt.tight_layout()
                plt.savefig(
                    f"figures/spectrum/{dev_name}/{gr_idx.strftime("%Y-%m-%d_%H-%M")}_sens_{s}.png"
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
        n_periods = 10  # fit over 10 periods

        phases = {}
        for gr_idx, df_h in dev.df.groupby(pd.Grouper(freq=chunk_time)):

            for s in dev.sensors:
                os.makedirs(f"figures/sine_fit/{dev_name}/{s}", exist_ok=True)
                n = int(n_periods * period)
                n = min(n, len(df_h))
                if n < 5:
                    continue  # too few points

                t_raw = (
                    df_h["time"].iloc[:n] - df_h["time"].iloc[0]
                ).dt.total_seconds()
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
                model = (
                    lambda x, A, B, C: A * np.sin(2 * np.pi * monitor_freq * (x + B))
                    + C
                )

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
                results.append(
                    {
                        "device": dev_name,
                        "sensor": s,
                        "idx": gr_idx,
                        "vpp": np.abs(p[0]),
                        "phase": p[1],
                        "offset": p[2],
                    }
                )

                # ---- PLOT ----
                plt.figure(figsize=(8, 4))
                plt.plot(t, y, label="data")
                plt.plot(t, model(t, *p), label="fit", linestyle="--")
                plt.xlabel("time [s]")
                plt.ylabel("sensor value")
                plt.title(
                    f"{dev_name} {s} {gr_idx} \nVpp: {np.abs(p[0]):.3f}, phase: {p[1]:.3f}, offset: {p[2]:.3f}"
                )
                plt.legend()
                plt.grid()
                plt.savefig(
                    f"figures/sine_fit/{dev_name}/{s}/{gr_idx.strftime("%Y-%m-%d_%H-%M")}.png"
                )
                plt.close()

    results = pd.DataFrame(results)
    results.index = pd.to_datetime(results["idx"])

    os.makedirs("figures/relative_phases", exist_ok=True)
    for pair in config["analysis"]["relative_phases"]:
        if len(pair) != 2:
            raise Exception("Formato coppie di sensori non valido")
        dev1, s1 = pair[0].split(".")
        dev2, s2 = pair[1].split(".")
        phase_delays = []
        for _, gr_df in results.groupby(pd.Grouper(freq=chunk_time)):
            dev1_df = gr_df[(gr_df["device"] == dev1) & (gr_df["sensor"] == s1)]
            dev2_df = gr_df[(gr_df["device"] == dev2) & (gr_df["sensor"] == s2)]

            # assume single sensor per device
            pd1 = dev1_df["phase"].values[0] * 1e3
            pd2 = dev2_df["phase"].values[0] * 1e3

            phase_delays.append(pd1 - pd2)

        # convert to numpy array for plotting
        phase_delays = np.array(phase_delays)

        plt.figure(figsize=(10, 5))
        plt.plot(phase_delays, "o--")
        plt.xlabel("hour index")
        plt.ylabel("phase delay [ms]")
        plt.grid()
        plt.tight_layout()
        plt.savefig(
            f"figures/relative_phases/{pair[0].replace('.', '-')}_{pair[1].replace('.', '-')}.png"
        )
        plt.close()
