import re
import stat
from unittest import result
import duckdb
import pandas as pd
from matplotlib import pyplot as plt
import sys
import os
from scipy.fft import rfft, rfftfreq
import numpy as np
import yaml
from datetime import datetime as dt
from scipy.optimize import curve_fit


class Device:
    def __init__(self, df, sensors):
        self.df = df
        self.fs = self.compute_sampling_frequency(df["ts"])
        self.sensors = sensors

    @staticmethod
    def compute_sampling_frequency(time_df):
        dt = time_df.diff().dt.total_seconds().iloc[1]
        return 1 / dt

    @staticmethod
    def get_all_hours(df):
        return df["hour"].unique()

    @staticmethod
    def get_all_days(df):
        return df["day"].unique()

    @staticmethod
    def get_all_months(df):
        return df["month"].unique()


# Read sensors from config.yml
with open("config.yml", "r") as f:
    config = yaml.safe_load(f)

db_path = sys.argv[1]
os.makedirs("figures", exist_ok=True)
os.makedirs("figures/spectral_analysis", exist_ok=True)

models = config.get("models", [])

conn = duckdb.connect(db_path, read_only=True)  # connect to database


devices = {}
for m in models:
    dev_name = m["name"]
    df = conn.execute(
        f"""
        SELECT * FROM {dev_name} WHERE date_part('minute', ts) <= {config.get('analysis').get('minutes_per_hour')}
        """
    ).df()

    devices[dev_name] = Device(df, m.get("sensors", []))

    print(f"Loaded dataframe for model {dev_name}")
    print(f"Estimated sampling frequency: {devices[dev_name].fs}")
    print(f"Data example:")
    print(df.head(5))
    print()


####################################################################
###### VISUALIZE 2 SECONDS OF DATA FOR EACH HOUR FOR EACH DAY ######
####################################################################
if config.get("analysis").get("time_visualization") is True:
    n_seconds = 2
    os.makedirs("figures/2_seconds", exist_ok=True)
    for dev_name in devices.keys():
        dev = devices[dev_name]
        os.makedirs(f"figures/2_seconds/{dev_name}", exist_ok=True)
        for day in Device.get_all_days(dev.df):
            dev_df_day = dev.df[dev.df["day"] == day].copy()
            for hour in Device.get_all_hours(dev_df_day):
                dev_df_hour = dev_df_day[dev_df_day["hour"] == hour].copy()

                if dev_df_hour.shape[0] is None:
                    continue

                for s in dev.sensors:
                    plt.plot(
                        (
                            dev_df_hour["ts"].iloc[: int(n_seconds * dev.fs)]
                            - dev_df_hour["ts"].iloc[0]
                        ).dt.total_seconds(),
                        dev_df_hour[s].iloc[: int(n_seconds * dev.fs)],
                        label=s,
                    )

                    plt.xlabel("time [s]")
                    plt.ylabel("values")
                    plt.legend()
                    plt.title(f"Device: {dev_name}, Day: {day}, Hour: {hour}")
                    plt.grid()
                    plt.savefig(
                        f"figures/2_seconds/{dev_name}/day_{day}_hour_{hour}_sens_{s}.png"
                    )
                    plt.close()
####################################################################


####################################################################
#### VISUALIZE OFFSET BETWEEN THE MAP AND THE ACD ####
####################################################################
if config.get("analysis").get("offset") is True:
    os.makedirs("figures/offset", exist_ok=True)
    for dev_name in devices.keys():
        dev = devices[dev_name]
        os.makedirs(f"figures/offset/{dev_name}", exist_ok=True)
        avg_vals = []
        time_list = []
        for day in Device.get_all_days(dev.df):
            dev_df_day = dev.df[dev.df["day"] == day].copy()
            for hour in Device.get_all_hours(dev_df_day):
                dev_df_hour = dev_df_day[dev_df_day["hour"] == hour].copy()

                if dev_df_hour.shape[0] is None:
                    continue

                for s in dev.sensors:
                    avg_vals.append(np.average(dev_df_hour[s].values))
                    time_list.append(dev_df_hour["ts"].iloc[0].strftime("%m/%d\n%H:%H"))
        plt.plot(time_list, avg_vals, "o--")
        plt.xlabel("time")
        plt.ylabel("offset")
        plt.title(f"{dev_name} offset over time")
        plt.grid()
        factor = len(time_list) // 10 + 1
        plt.xticks(time_list[::factor] + [time_list[-1]], rotation=30, ha="right")
        plt.tight_layout()
        plt.savefig(f"figures/offset/{dev_name}/offset_5min_per_hour.png")
        plt.close()
# ####################################################################


####################################################################
########### SPECTRAL ANALYSIS OF THE MAP AND THE ACD ###############
####################################################################
if config.get("analysis").get("spectrum") is True:
    os.makedirs("figures/offset", exist_ok=True)
    for dev_name in devices.keys():
        dev = devices[dev_name]
        os.makedirs(f"figures/spectrum/{dev_name}", exist_ok=True)
        avg_vals = []
        time_list = []
        for day in Device.get_all_days(dev.df):
            dev_df_day = dev.df[dev.df["day"] == day].copy()
            for hour in Device.get_all_hours(dev_df_day):
                dev_df_hour = dev_df_day[dev_df_day["hour"] == hour].copy()
                for s in dev.sensors:
                    X_acd = rfft(dev_df_hour[s].to_numpy() - dev_df_hour[s].mean())
                    freqs_acd = rfftfreq(dev_df_hour.shape[0], 1 / dev.fs)
                    plt.figure(figsize=(10, 5))
                    plt.plot(
                        freqs_acd[1 : len(dev_df_hour[s]) // 2],
                        np.abs(X_acd)[1 : len(dev_df_hour[s]) // 2],
                        label=f"acd",
                        color="blue",
                        alpha=1.0,
                    )
                    plt.xlabel("Frequency [Hz]")
                    plt.ylabel("Amplitude")
                    plt.title(f"Day: {day}, Hour: {hour}")
                    plt.yscale("log")
                    plt.xlim([0.5, 60])
                    plt.xticks(list(range(0, 61, 5)))
                    plt.grid()
                    plt.tight_layout()
                    plt.savefig(
                        f"figures/spectrum/{dev_name}/day_{day}_hour_{hour}_sens_{s}.png"
                    )
                    plt.close()

####################################################################

####################################################################
################### FIT TO SINUSOIDAL FUNCTION #####################
####################################################################
results = []
if config.get("analysis").get("sine_fit") is True:
    n_seconds = 2
    os.makedirs("figures/sine_fit", exist_ok=True)
    for dev_name in devices.keys():
        dev = devices[dev_name]
        os.makedirs(f"figures/sine_fit/{dev_name}", exist_ok=True)
        for day in Device.get_all_days(dev.df):
            dev_df_day = dev.df[dev.df["day"] == day].copy()
            for hour in Device.get_all_hours(dev_df_day):
                dev_df_hour = dev_df_day[dev_df_day["hour"] == hour].copy()

                if dev_df_hour.shape[0] is None:
                    continue

                for s in dev.sensors:
                    time_data = (
                        dev_df_hour["ts"].iloc[: int(n_seconds * dev.fs)]
                        - dev_df_hour["ts"].iloc[0]
                    ).dt.total_seconds()
                    sens_data = dev_df_hour[s].iloc[: int(n_seconds * dev.fs)]

                    fit_model = (
                        lambda x, A, B, C: A * np.sin(2 * np.pi * 5 * (x + B)) + C
                    )
                    params, covariance = curve_fit(
                        fit_model,
                        time_data,
                        sens_data,
                        p0=[0.1, 0.0, 0.0],
                        bounds=([0, 0, -np.inf], [np.inf, np.inf, np.inf]),
                    )
                    plt.plot(time_data, sens_data, label="data")

                    plt.plot(
                        time_data,
                        fit_model(time_data, params[0], params[1], params[2]),
                        label="fit",
                    )

                    plt.xlabel("time [s]")
                    plt.ylabel("values")
                    plt.legend()
                    plt.title(f"Device: {dev_name}, Day: {day}, Hour: {hour}")
                    plt.grid()
                    plt.savefig(
                        f"figures/sine_fit/{dev_name}/day_{day}_hour_{hour}_sens_{s}.png"
                    )
                    plt.close()

                    results.append(
                        {
                            "sensor": s,
                            "device": dev_name,
                            "hour": hour,
                            "vpp": params[0],
                            "phase": params[1],
                            "offset": params[2],
                        }
                    )
results = pd.DataFrame(results)
results.to_csv("results.csv", index=False)

relative_phase = (
    results[results["device"] == "acd"]["phase"].values
    - results[results["device"] == "map"]["phase"].values
)
plt.plot(range(len(relative_phase)), relative_phase, "o--")
plt.show()

####################################################################


# ####################################################################
# ############ PHASE DELAY BETWEEN THE MAP AND THE ACD ###############
# ####################################################################
# peak_phases_acd = []
# peak_phases_map = []
# for day in days:
#     acd_day = df_acd[df_acd["day"] == day].copy()
#     map_day = df_map[df_map["day"] == day].copy()
#     for hour in hours:
#         dev_df_hour = acd_day[acd_day["hour"] == hour].copy()
#         map_hour = map_day[map_day["hour"] == hour].copy()

#         if len(dev_df_hour) == 0 or len(map_hour) == 0:
#             continue

#         X_acd = rfft(
#             dev_df_hour["sensor_vals"].to_numpy() - dev_df_hour["sensor_vals"].mean()
#         )
#         freqs_acd = rfftfreq(dev_df_hour.shape[0], 1 / fs_acd)
#         X_map = rfft(
#             map_hour["sensor_vals"].to_numpy() - map_hour["sensor_vals"].mean()
#         )
#         freqs_map = rfftfreq(map_hour.shape[0], 1 / fs_map)

#         angles = np.angle(X_acd)
#         phase_at_peak = angles[
#             np.argmin(np.abs(freqs_acd - 5))
#         ]  # compute phase at closest point to 5Hz
#         peak_phases_acd.append(phase_at_peak)

#         angles = np.angle(X_map)
#         phase_at_peak = angles[
#             np.argmin(np.abs(freqs_map - 5))
#         ]  # compute phase at closest point to 5Hz
#         peak_phases_map.append(phase_at_peak)
# peak_phases_map = np.asarray(peak_phases_map)
# peak_phases_acd = np.asarray(peak_phases_acd)
# # plt.subplot(3, 1, 1)
# # plt.plot(peak_phases_acd / (np.pi * 2 * fs_acd) * 1e3, linestyle="--", marker="o")
# # plt.grid()
# # plt.xlabel("Time [h]")
# # plt.ylabel("Phase [ms]")
# # plt.title("ACD phase")
# #
# # plt.subplot(3, 1, 2)
# # plt.plot(peak_phases_map / (np.pi * 2 * fs_map) * 1e3, linestyle="--", marker="o")
# # plt.grid()
# # plt.xlabel("Time [h]")
# # plt.ylabel("Phase [ms]")
# # plt.title("MAP phase")
# #
# # plt.subplot(3, 1, 3)
# plt.plot(
#     peak_phases_map / (2 * np.pi * 5.0) * 1e3
#     - peak_phases_acd / (2 * np.pi * 5.0) * 1e3,
#     linestyle="--",
#     marker="o",
# )
# plt.grid()
# plt.xlabel("Time [h]")
# plt.ylabel("Phase [ms]")
# plt.title("ACD-MAP phase delay")
# plt.tight_layout()
# plt.savefig("figures/phases.png")

# ####################################################################

conn.close()
