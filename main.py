import os
import sys
import yaml
import pandas as pd
import numpy as np

BASE_PATH = os.path.join(os.path.dirname(__file__), "src")
sys.path.append(BASE_PATH)

from device import Device
from viz import device_plots
from offset import device_offset_plot
from spectral import device_spectral_plot
from periodic import sine_fit, SineModelFit
from delay import phase_delay


# ================= CONFIG =================
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

chunk_time = config["chunk_time"]
os.makedirs("figures", exist_ok=True)

devices = {}
models = config.get("models", [])
for idx, model in enumerate(models):
    devices[model["name"]] = Device(
        model["data_folder"], model["sensors"], model["name"], flip=model["flip_values"]
    )

    devices[model["name"]].df.index = pd.to_datetime(
        devices[model["name"]].df["time"].copy(), utc=True
    )
    devices[model["name"]].df.sort_index()

    dts = devices[model["name"]].df["time"].diff().dt.total_seconds()
    dt = dts.mode().iloc[0]
    mask = dts.isna() | (dts != dt)
    if any(mask[1:]):
        print("Found irregular sampling")
        print(devices[model["name"]].df["time"][mask])
        raise Exception("Error: time is not contiguous.")

# =========================================================
# VISUALIZE 2 SECONDS OF DATA PER DAY-HOUR
# =========================================================
if config["analysis"].get("time_visualization"):
    n_seconds = 2
    os.makedirs("figures/2_seconds", exist_ok=True)
    print("Visualization")
    for dev_name, dev in devices.items():
        device_plots(
            dev=dev,
            chunk_time=config["chunk_time"],
            n_seconds=5,
            sensor_list=dev.sensors,
        )
# =========================================================


# =========================================================
# OFFSET OVER TIME
# =========================================================
if config["analysis"].get("offset"):
    os.makedirs("figures/offset", exist_ok=True)
    print("Offset")
    for dev_name, dev in devices.items():
        device_offset_plot(
            dev=dev, chunk_time=config["chunk_time"], sensor_list=dev.sensors
        )
# =========================================================


# =========================================================
# SPECTRAL ANALYSIS PER DAY-HOUR
# =========================================================
if config["analysis"].get("spectrum"):
    os.makedirs("figures/spectrum", exist_ok=True)
    print("Spectrum")
    for dev_name, dev in devices.items():
        device_spectral_plot(
            dev=dev, chunk_time=config["chunk_time"], sensor_list=dev.sensors
        )
# =========================================================


# =========================================================
# SINE FIT PER DAY-HOUR
# =========================================================
monitor_freq = 5
if config["analysis"].get("sine_fit"):
    os.makedirs("figures/sine_fit", exist_ok=True)
    print("Sine fit")
    for dev_name, dev in devices.items():
        sine_fit(
            dev=dev,
            chunk_time=config["chunk_time"],
            n_seconds=2,
            monitor_freq=5,
            sensor_list=dev.sensors,
        )

    os.makedirs("figures/phase_delay", exist_ok=True)
    print("Phase delay")

all_delays = []
for pair in config["analysis"]["relative_phases"]:
    if len(pair) != 2:
        raise Exception("Formato coppie di sensori non valido")
    dev1_str, s1 = pair[0].split(".")
    dev2_str, s2 = pair[1].split(".")
    df1 = devices[dev1_str].df
    df2 = devices[dev2_str].df
    smf1 = SineModelFit()
    smf2 = SineModelFit()

    delay = phase_delay(
        df1[["time", s1]],
        df2[["time", s2]],
        chunk_time=config["chunk_time"],
        sine_freq_hz=float(config["sine_freq"]),
    )

    all_delays += [
        {
            "device1": dev1_str,
            "sensor1": s1,
            "device2": dev2_str,
            "sensor2": s2,
            "start_time": d["start_time"],
            "delay": d["delay"],
        }
        for d in delay
    ]

df_delays = pd.DataFrame(all_delays)
df_delays.to_csv("figures/phase_delay/delays.csv", index=False)
# print(df_delays)
print("Done")


# =========================================================
