import numpy as np
import pandas as pd
import os


class Device:
    def __init__(self, path, sensors, name, flip=False):
        self.name = name
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
