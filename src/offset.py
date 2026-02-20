from src.device import Device
from matplotlib import pyplot as plt
import pandas as pd
import os
import numpy as np


def device_offset_plot(dev: Device, chunk_time: str, sensor_list: list | None):
    os.makedirs(f"figures/offset/{dev.name}", exist_ok=True)
    if sensor_list is None:
        sensor_list = dev.sensors
    for s in sensor_list:
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
        plt.savefig(f"figures/offset/{dev.name}/sensor_{s}_offset.png")
        plt.close()
