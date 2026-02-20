import numpy as np
import os, sys

BASE_PATH = os.path.join(os.path.dirname(__file__), "src")
sys.path.append(BASE_PATH)

from device import Device
from viz import device_plots
from offset import device_offset_plot
from spectral import device_spectral_plot
from periodic import sine_fit, SineModelFit
from delay import phase_delay

import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime, timedelta

for my_delay in [0.003, 0.005, 0.008]:
    dt = datetime(2024, 1, 1) + pd.to_timedelta(np.arange(0, 60, 0.005), unit="s")
    t = (dt - dt[0]).total_seconds()
    samples_per_period = int(
        1 / 5 / 0.005
    )  # 5 Hz, 200 ms period, 40 samples per period

    y1 = (
        2 * np.sin(2 * np.pi * 5 * (t + 0.0))
        + 1.0
        + 0.1 * np.random.normal(size=t.shape)
    )
    y2 = (
        2 * np.sin(2 * np.pi * 5 * (t + my_delay))
        + 1.0
        + 0.1 * np.random.normal(size=t.shape)
    )
    df1 = pd.DataFrame({"time": pd.to_datetime(dt), "sensor1": y1})
    df2 = pd.DataFrame({"time": pd.to_datetime(dt), "sensor2": y2})
    smf1 = SineModelFit()
    smf2 = SineModelFit()
    delay = phase_delay(
        df1[["time", "sensor1"]],
        df2[["time", "sensor2"]],
        chunk_time="10s",
        sine_freq_hz=5,
    )
    plt.plot(t[:samples_per_period], y1[:samples_per_period], label="sensor1")
    plt.plot(t[:samples_per_period], y2[:samples_per_period], label="sensor2")
    plt.legend()
    plt.grid()
    plt.show()

    plt.plot([d["delay"] for d in delay])
    plt.ylabel("delay [ms]")
    plt.xlabel("chunk index")
    plt.show()
