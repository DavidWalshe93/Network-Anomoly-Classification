"""
Author:         David Walshe
Date:           05/04/2020   
"""

from datetime import datetime as dt
from collections import OrderedDict


class Timer:

    def __init__(self):
        print("Timer Started")
        self.initial_start_time = dt.now()
        self.section_start_time = dt.now()
        self.delta = None
        self.time_log = OrderedDict()

    def time_stage(self, stage):
        end_time = dt.now()
        self.delta = end_time - self.section_start_time
        self._print_time(stage=stage)

        self.time_log.update({stage: self.delta.total_seconds()})

        self._reset()

    def time_script(self):
        end_time = dt.now()
        self.delta = end_time - self.initial_start_time
        self._print_time(stage="END")

        self._reset()

    def _reset(self):
        self.section_start_time = dt.now()

    def _print_time(self, stage):
        minutes = round(self.delta.total_seconds() / 60)
        seconds = round((self.delta.total_seconds() % 60))
        microseconds = round(((self.delta.total_seconds() % 60) % 1) * 1000)

        print(f"{minutes:02d}:{seconds:02d}.{microseconds:03d} - Stage: {stage}")

    @property
    def plot_data(self):
        return list(self.time_log.keys()), list(self.time_log.values())
