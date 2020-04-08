"""
Author:         David Walshe
Date:           09/04/2020   
"""


import matplotlib.pyplot as plt


def plot_model_build_time(stages, times):
    import math
    fig, ax = plt.subplots()
    ax.bar(stages, times)
    plt.xticks(stages, stages)
    max_time = math.ceil(max(times))
    tick_scale = math.ceil(max_time / 20)
    max_time += tick_scale
    plt.yticks([i for i in range(0, max_time, tick_scale)],
               [i if max_time < 60 else f"{int(i / 60)}:{i % 60}" for idx, i in
                enumerate(range(0, max_time, tick_scale))])
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    total_time = sum(times)
    if max_time > 60:
        total_time = f"{round(total_time / 60)}m {round(total_time % 60)}s"
        plt.ylabel("Minutes")
    else:
        plt.ylabel("Seconds")
    plt.xlabel("Stages")

    textstr = f"Total Time: {total_time}"

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    # place a text box in upper left in axes coords
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

    plt.show()
