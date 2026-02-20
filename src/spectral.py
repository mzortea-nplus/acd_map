from src.device import Device
from matplotlib import pyplot as plt
import pandas as pd
import os
from scipy.fft import rfft, rfftfreq
import numpy as np


def device_spectral_plot(dev: Device, chunk_time: str, sensor_list: list | None):
    os.makedirs(f"figures/spectrum/{dev.name}", exist_ok=True)
    if sensor_list is None:
        sensor_list = dev.sensors
    for gr_idx, df_h in dev.df.groupby(pd.Grouper(freq=chunk_time)):
        for s in sensor_list:
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
                f"{dev.name} - {s} - {gr_idx.strftime("%Y-%m-%d_%H-%M")} \nPeak freq. {peak_freq:.3f} Hz"
            )
            plt.tight_layout()
            plt.savefig(
                f"figures/spectrum/{dev.name}/{gr_idx.strftime("%Y-%m-%d_%H-%M")}_sens_{s}.png"
            )
            plt.close()
