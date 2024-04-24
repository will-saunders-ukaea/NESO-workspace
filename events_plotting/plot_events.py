"""
Helper script for plotting ProfileRegions.
"""

import json
import os
import glob
import numpy as np
import glob
import math
import sys
import pandas as pd
import plotly.express as px


class ColourMapper:
    def __init__(self):
        self.num_colours = 0
        self.names = {}

    def get(self, name):
        if name in self.names:
            return self.names[name]
        else:
            n = self.num_colours
            self.names[name] = n
            self.num_colours += 1
            return n


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("""Error: Expected at least one argument. Arguments are:
    1) A directory containing JSON files containing regions to plot.
    2) (optional) A start time to ignore events before.
    3) (optional) A end time to ignore events after. """)
        exit(-1)

    files = glob.glob(os.path.join(sys.argv[1], "*.json"))
    cutoff_start = -sys.float_info.max
    cutoff_end = sys.float_info.max

    if len(sys.argv) > 2:
        cutoff_start = float(sys.argv[2])
    if len(sys.argv) > 3:
        cutoff_end = float(sys.argv[3])

    dd = {
        "rank": [],
        "name": [],
        "time_start": [],
        "time_end": [],
        "time_elapsed_plot": [],
        "time_elapsed": [],
        "colour": [],
    }

    colour_mapper = ColourMapper()

    print(f"Found {len(files)} source files.")

    for fx in files:
        data = json.loads(open(fx).read())
        rank = data["rank"]
        for rx in data["regions"]:
            time_start = rx[2]
            time_end = rx[3]
            if (time_start >= cutoff_start) and (time_end <= cutoff_end):
                dd["rank"].append(rank)
                name = rx[0] + ":" + rx[1]
                dd["name"].append(name)
                dd["time_start"].append(time_start)
                dd["time_end"].append(time_end)
                dd["time_elapsed_plot"].append(time_end - time_start)
                dd["time_elapsed"].append(time_end - time_start)
                dd["colour"].append(
                    px.colors.qualitative.Dark24[colour_mapper.get(name)]
                )

    df = pd.DataFrame.from_dict(dd)
    print(df)

    labels = {
        "time_elapsed_plot": "Time",
        "time_start": "Start Time",
        "time_end": "End Time",
        "time_elapsed": "Time Elapsed",
    }

    fig = px.bar(
        df,
        x="time_elapsed_plot",
        base="time_start",
        y="rank",
        orientation="h",
        hover_data={
            "name": True,
            "time_start": True,
            "time_end": True,
            "time_elapsed_plot": False,
            "time_elapsed": True,
            "colour": False,
        },
        color="name",
        barmode="overlay",
        # barmode="group",
        # barmode="relative",
        color_discrete_sequence=px.colors.qualitative.Dark24,
        hover_name="name",
        labels=labels,
    )

    fig.show()
