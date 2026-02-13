from src.device import Device
from matplotlib import pyplot as plt
import pandas as pd
import os, sys

FIGURES_DIR = "figures"


def device_plots(
    dev: Device, chunk_time: str, n_seconds: int, sensor_list: list | None
):
    os.makedirs(f"figures/2_seconds/{dev.name}", exist_ok=True)
    if sensor_list is None:
        sensor_list = dev.sensors
    for gr_idx, df_h in dev.df.groupby(pd.Grouper(freq=chunk_time)):
        if df_h.empty:
            continue
        for s in sensor_list:
            n = int(n_seconds * dev.fs)
            t0 = df_h["time"].iloc[0]

            plt.plot(
                (df_h["time"].iloc[:n] - t0).dt.total_seconds(),
                df_h[s].iloc[:n],
            )

            plt.title(f"{dev.name} – {gr_idx}")
            plt.xlabel("time [s]")
            plt.ylabel("values")
            plt.grid()
            dh_ts = pd.to_datetime(gr_idx)
            dh_str = dh_ts.strftime("%Y-%m-%d_%H-%M")
            plt.savefig(f"figures/2_seconds/{dev.name}/{dh_str}_sens_{s}.png")
            plt.close()
