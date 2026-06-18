import numpy as np
from typing import Optional, List


class BaseControlChart:
    def __init__(self, data, datetimes=None):
        self.data = np.asarray(data)
        self.datetimes = np.asarray(datetimes) if datetimes is not None else None
        self.n_subgroups = len(data)
        self.subgroup_size = data.shape[1] if data.ndim > 1 else 1

        self.stat_main = None
        self.stat_disp = None
        self.cl_main = None
        self.ucl_main = None
        self.lcl_main = None
        self.cl_disp = None
        self.ucl_disp = None
        self.lcl_disp = None
        self.main_label = "Основная статистика"
        self.disp_label = "Статистика разброса"

    def fit(self, baseline_mask=None, method='classic'):
        raise NotImplementedError

    def _get_mask(self, start=None, end=None, last_n=30):
        if self.datetimes is None:
            if last_n is not None:
                return np.arange(len(self.stat_main)) >= (len(self.stat_main) - last_n)
            return np.ones(len(self.stat_main), dtype=bool)

        dates = np.asarray(self.datetimes)
        if start is not None:
            return dates >= np.datetime64(start)
        elif end is not None:
            return dates <= np.datetime64(end)
        else:
            return np.arange(len(dates)) >= (len(dates) - last_n)

    def plot(self, start=None, end=None, last_n=30):
        from spc_lib.visualization import plot_control_chart
        return plot_control_chart(self, start, end, last_n)

    def plot_rules(self, start=None, end=None, last_n=30, rules=None, n_cols=1):
        from spc_lib.visualization import plot_rules_violations
        return plot_rules_violations(self, start, end, last_n, rules, n_cols)
