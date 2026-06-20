import numpy as np
from spc_lib.core.base_chart import BaseControlChart

class PChart(BaseControlChart):
    """
    p-карта (контрольная карта доли дефектных изделий).
    """
    def __init__(self, data, n_fixed=None, datetimes=None, target=None, usl=None, lsl=None):
        super().__init__(data, datetimes, target, usl, lsl)
        self.n_fixed = n_fixed
        self.main_label = "p-карта: Доля дефектных единиц"
        self.disp_label = None

        self.p_values = None
        self.n_values = None
        self.p_bar = None
        self.n_bar = None

    def fit(self, baseline_mask=None, method='classic', use_average_n=False):
        """
        use_average_n: bool
            Если True, для переменного размера выборки будут вычислены
            константные (прямые) границы на основе среднего размера выборки (n_bar).
        """
        if baseline_mask is None:
            baseline_mask = np.ones(self.n_subgroups, dtype=bool)

        if self.n_fixed is not None:
            if self.data.ndim == 2 and self.data.shape[1] == 1:
                self.p_values = self.data.flatten()
            else:
                self.p_values = np.asarray(self.data).flatten()
            self.n_values = np.full(len(self.p_values), self.n_fixed)
        else:
            if self.data.ndim == 2 and self.data.shape[1] == 2:
                self.p_values = self.data[:, 0] / self.data[:, 1]
                self.n_values = self.data[:, 1]
            else:
                raise ValueError("При переменном объеме выборки data должно быть 2D массивом: [x, n]")

        base_p = self.p_values[baseline_mask]
        base_n = self.n_values[baseline_mask]

        if self.target is not None:
            self.p_bar = self.target
        else:
            total_defects = np.sum(base_p * base_n)
            total_n = np.sum(base_n)
            self.p_bar = total_defects / total_n if total_n > 0 else np.mean(base_p)

        self.n_bar = np.mean(base_n)

        if method == 'classic':
            self.cl_main = self.p_bar

            # --- ИЗМЕНЕНИЕ ЗДЕСЬ ---
            # Если n_fixed задан ИЛИ пользователь попросил усредненные границы
            if self.n_fixed is not None or use_average_n:
                n_eval = self.n_fixed if self.n_fixed is not None else self.n_bar

                sigma_p = np.sqrt(self.p_bar * (1 - self.p_bar) / n_eval)
                self.ucl_main = self.p_bar + 3 * sigma_p
                self.lcl_main = max(0, self.p_bar - 3 * sigma_p)
            else:
                # Переменные (ступенчатые) границы
                self.ucl_main = np.zeros(len(self.p_values))
                self.lcl_main = np.zeros(len(self.p_values))

                for i in range(len(self.p_values)):
                    n_i = self.n_values[i]
                    sigma_p_i = np.sqrt(self.p_bar * (1 - self.p_bar) / n_i)
                    self.ucl_main[i] = self.p_bar + 3 * sigma_p_i
                    self.lcl_main[i] = max(0, self.p_bar - 3 * sigma_p_i)

            self.sigma_est = np.sqrt(self.p_bar * (1 - self.p_bar))

        elif method == 'percentiles':
            self.cl_main = np.median(base_p)
            self.ucl_main = np.percentile(base_p, 99.865)
            self.lcl_main = np.percentile(base_p, 0.135)
            self.sigma_est = np.std(base_p, ddof=1)

        else:
            raise ValueError(f"Неизвестный метод: {method}")

        self.stat_main = self.p_values
        self.cl_disp = None
        self.ucl_disp = None
        self.lcl_disp = None

        return self

class CChart(BaseControlChart):
    """
    c-карта (контрольная карта числа дефектов).

    Используется для мониторинга количества дефектов на единицу продукции
    при постоянном объеме выборки.
    """

    def __init__(self, data, datetimes=None, target=None, usl=None, lsl=None):
        super().__init__(data, datetimes, target, usl, lsl)
        self.main_label = "c-карта: Число дефектов"
        self.disp_label = None

        self.c_values = None
        self.c_bar = None

    def fit(self, baseline_mask=None, method='classic'):
        if baseline_mask is None:
            baseline_mask = np.ones(self.n_subgroups, dtype=bool)

        if self.data.ndim == 2 and self.data.shape[1] == 1:
            self.c_values = self.data.flatten()
        else:
            self.c_values = np.asarray(self.data).flatten()

        base_c = self.c_values[baseline_mask]

        if self.target is not None:
            self.c_bar = self.target
        else:
            self.c_bar = np.mean(base_c)

        if method == 'classic':
            self.cl_main = self.c_bar
            sigma_c = np.sqrt(self.c_bar)
            self.ucl_main = self.c_bar + 3 * sigma_c
            # LCL не может быть меньше 0
            self.lcl_main = max(0, self.c_bar - 3 * sigma_c)
            self.sigma_est = sigma_c

        elif method == 'percentiles':
            self.cl_main = np.median(base_c)
            self.ucl_main = np.percentile(base_c, 99.865)
            self.lcl_main = np.percentile(base_c, 0.135)
            self.sigma_est = np.std(base_c, ddof=1)

        else:
            raise ValueError(f"Неизвестный метод: {method}")

        self.stat_main = self.c_values
        self.cl_disp = None
        self.ucl_disp = None
        self.lcl_disp = None

        return self
