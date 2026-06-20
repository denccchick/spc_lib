# spc_lib/core/base_chart.py

import numpy as np
from typing import Optional, List


class BaseControlChart:
    def __init__(self, data, datetimes=None, target=None, usl=None, lsl=None):
        self.data = np.asarray(data)
        self.datetimes = np.asarray(datetimes) if datetimes is not None else None
        self.target = target
        self.usl = usl
        self.lsl = lsl
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
        self.sigma_est = None

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

    def plot(self, start=None, end=None, last_n=30, show_spec=False):
        from spc_lib.visualization import plot_control_chart
        return plot_control_chart(self, start, end, last_n, show_spec)

    def plot_rules(self, start=None, end=None, last_n=30, rules=None, n_cols=1, show_spec=False):
        from spc_lib.visualization import plot_rules_violations
        return plot_rules_violations(self, start, end, last_n, rules, n_cols, show_spec)

    def capability(self, usl=None, lsl=None):
            """
            Расчет показателей пригодности процесса (Cp, Cpk, Cpl, Cpu).
            """
            usl = usl if usl is not None else self.usl
            lsl = lsl if lsl is not None else self.lsl

            if usl is None and lsl is None:
                raise ValueError("Необходимо задать хотя бы одну границу спецификации (usl или lsl)")

            if self.stat_main is None or self.cl_main is None:
                raise ValueError("Карта не обучена (сначала вызовите .fit())")

            # Центр процесса
            mu = self.cl_main

            # Оценка сигмы (внутригрупповой)
            if self.sigma_est is not None:
                sigma = self.sigma_est
            elif self.ucl_main is not None and self.lcl_main is not None:
                if self.subgroup_size > 1:
                    sigma = (self.ucl_main - self.lcl_main) * np.sqrt(self.subgroup_size) / 6
                else:
                    sigma = (self.ucl_main - self.lcl_main) / 6
            else:
                # Fallback (общая дисперсия, даст Pp вместо Cp)
                stats_data = np.asarray(self.stat_main)
                stats_data = stats_data[~np.isnan(stats_data)]
                sigma = np.std(stats_data, ddof=1)

            if sigma <= 0:
                return {'cp': np.nan, 'cpk': np.nan, 'cpl': np.nan, 'cpu': np.nan}

            # Инициализируем NaN
            cp = cpk = cpl = cpu = np.nan

            # Считаем то, что возможно
            if usl is not None and lsl is not None:
                cp = (usl - lsl) / (6 * sigma)

            if usl is not None:
                cpu = (usl - mu) / (3 * sigma)

            if lsl is not None:
                cpl = (mu - lsl) / (3 * sigma)

            # Cpk - это минимум из существующих односторонних индексов
            if usl is not None and lsl is not None:
                cpk = min(cpu, cpl)
            elif usl is not None:
                cpk = cpu
            else:
                cpk = cpl

            return {
                'cp': cp,
                'cpk': cpk,
                'cpl': cpl,
                'cpu': cpu
            }
