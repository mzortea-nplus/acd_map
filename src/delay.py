import numpy as np
import pandas as pd
from scipy.signal import correlate
from matplotlib import pyplot as plt
from periodic import SineModelFit

debug = False


def sine_model(x, A, B, C, sine_freq_hz):
    return A * np.sin(2 * np.pi * sine_freq_hz * (x + B)) + C


def phase_delay(
    df1: pd.DataFrame, df2: pd.DataFrame, chunk_time: str, sine_freq_hz: float
):
    # compute delay in seconds
    # merge df on time
    # ensure there's a single 'time' column - prefer existing column over index
    def _ensure_time_col(df: pd.DataFrame) -> pd.DataFrame:
        if "time" in df.columns:
            # keep the existing 'time' column and drop the index
            return df.reset_index(drop=True)
        else:
            # promote the index to a column (will be named by index.name or 'index')
            df = df.reset_index()
            # remove any accidental duplicate column names
            return df.loc[:, ~df.columns.duplicated()]

    df1 = _ensure_time_col(df1)
    df2 = _ensure_time_col(df2)

    if len(df1.columns) > 2 or len(df2.columns) > 2:
        raise Exception(
            "DataFrame deve avere solo due colonne: time e valore del sensore"
        )
    s1 = df1.columns.difference(["time"])[0]
    s2 = df2.columns.difference(["time"])[0]
    merged = pd.merge_asof(
        df1.sort_values("time"),
        df2.sort_values("time"),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta(seconds=0.01),
    )
    merged = merged.dropna()
    delays = []
    for gr_idx, gr_df in merged.groupby(pd.Grouper(key="time", freq=chunk_time)):
        if debug:
            print(f"Processing chunk {gr_idx}")

        fs = 1 / gr_df["time"].diff().dt.total_seconds().mode().iloc[0]
        gr_df = gr_df.iloc[
            : int(fs / sine_freq_hz) * 5, :
        ]  # keep only 10 periods for fitting
        y1 = gr_df[s1].values
        y2 = gr_df[s2].values

        smf1 = SineModelFit()
        smf2 = SineModelFit()

        # build relative time vector in seconds for this chunk
        t_raw = (gr_df["time"] - gr_df["time"].iloc[0]).dt.total_seconds().values

        t, y1 = smf1.preprocess(t_raw, y1)
        _, y2 = smf2.preprocess(t_raw, y2)
        p1 = smf1.fit(t, y1, monitor_freq=sine_freq_hz)
        p2 = smf2.fit(t, y2, monitor_freq=sine_freq_hz)

        t = np.linspace(
            0, t[-1], len(t) * 100
        )  # increase resolution for better delay estimation
        delta_t = t[1] - t[0]
        predicted1 = smf1.predict(t, p1, monitor_freq=sine_freq_hz)
        predicted2 = smf2.predict(t, p2, monitor_freq=sine_freq_hz)
        zero_crossings1 = np.where(
            np.diff(np.sign(predicted1 - np.mean(predicted1))) > 0
        )[0]
        zero_crossings2 = np.where(
            np.diff(np.sign(predicted2 - np.mean(predicted2))) > 0
        )[0]
        delay_idx = zero_crossings2[0] - zero_crossings1[0]
        delay_milliseconds = delay_idx * delta_t * 1000  # convert to ms
        if delay_milliseconds > 0.5 / sine_freq_hz * 1000:
            delay_milliseconds -= 1000 / sine_freq_hz  # adjust for wrap-around
        elif delay_milliseconds < -0.5 / sine_freq_hz * 1000:
            delay_milliseconds += 1000 / sine_freq_hz  # adjust for wrap-around
        if debug:
            print(f"Delay for chunk {gr_idx}: {delay_milliseconds} ms")
        delays.append({"start_time": gr_df["time"].min(), "delay": delay_milliseconds})
    plt.plot(
        [d["start_time"] for d in delays], [d["delay"] for d in delays], marker="o"
    )
    plt.xlabel("Time")
    plt.grid()
    plt.ylabel("Delay (ms)")
    plt.title(f"Delay over time {s1} - {s2}")
    plt.tight_layout()
    plt.savefig(f"figures/phase_delay/{s1}_{s2}.png")
    plt.close()

    print("Max delay (ms):", max(d["delay"] for d in delays))
    print("Min delay (ms):", min(d["delay"] for d in delays))
    print(
        "Delta delay (ms):",
        max(d["delay"] for d in delays) - min(d["delay"] for d in delays),
    )
    print("Average delay (ms):", np.mean([d["delay"] for d in delays]))
    return delays
