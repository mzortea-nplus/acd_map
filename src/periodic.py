from device import Device
import os
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt


class SineModelFit:
    def __init__(self):
        pass

    @staticmethod
    def model(x, A, B, C, sine_freq_hz):
        return A * np.sin(2 * np.pi * sine_freq_hz * (x + B)) + C

    def preprocess(self, t, y):
        # remove NaN and infinite
        mask = np.isfinite(t) & np.isfinite(y)
        t = t[mask]
        y = y[mask]

        # clip extreme outliers
        q1, q3 = np.percentile(y, [5, 95])
        y = np.clip(y, q1, q3)

        return t, y

    def fit(self, t, y, monitor_freq):
        # initial guess
        A_guess = (np.max(y) - np.min(y)) / 2
        B_guess = 0
        C_guess = np.mean(y)
        p0 = [A_guess, B_guess, C_guess]

        try:
            p, _ = curve_fit(
                lambda x, A, B, C: self.model(x, A, B, C, monitor_freq),
                t,
                y,
                p0=p0,
                maxfev=20000,
            )
            return p
        except Exception as e:
            print(e)
            return None

    def predict(self, t, p, monitor_freq):
        return self.model(t, *p, sine_freq_hz=monitor_freq)


def sine_fit(
    dev: Device,
    chunk_time: str,
    n_seconds: int,
    monitor_freq: int,
    sensor_list: list | None,
):
    os.makedirs(f"figures/sine_fit/{dev.name}", exist_ok=True)

    period = dev.fs / 5  # 5 Hz target
    n_periods = 10  # fit over 10 periods

    if sensor_list is None:
        sensor_list = dev.sensors

    for gr_idx, df_h in dev.df.groupby(pd.Grouper(freq=chunk_time)):

        for s in sensor_list:

            smf = SineModelFit()

            os.makedirs(f"figures/sine_fit/{dev.name}/{s}", exist_ok=True)
            n = int(n_periods * period)
            n = min(n, len(df_h))
            if n < 5:
                continue  # too few points

            t_raw = (df_h["time"].iloc[:n] - df_h["time"].iloc[0]).dt.total_seconds()
            y_raw = df_h[s].iloc[:n]

            t, y = smf.preprocess(t_raw.values, y_raw.values)

            p = smf.fit(t, y, monitor_freq)
            if p is None:
                continue

            # ---- PLOT ----
            plt.figure(figsize=(8, 4))
            plt.plot(t, y, label="data")
            plt.plot(t, smf.predict(t, p, monitor_freq), label="fit", linestyle="--")
            plt.xlabel("time [s]")
            plt.ylabel("sensor value")
            plt.title(
                f"{dev.name} {s} {gr_idx} \nVpp: {np.abs(p[0]):.3f}, phase: {p[1]:.3f}, offset: {p[2]:.3f}"
            )
            plt.legend()
            plt.grid()
            plt.savefig(
                f"figures/sine_fit/{dev.name}/{s}/{gr_idx.strftime('%Y-%m-%d_%H-%M')}.png"
            )
            plt.close()
