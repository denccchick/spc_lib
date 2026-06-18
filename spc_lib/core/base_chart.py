import numpy as np
from abc import ABC, abstractmethod


class BaseControlChart(ABC):

    def __init__(self, data, datetimes=None):

        self.data = np.asarray(data)

        if self.data.ndim != 2:
            raise ValueError(
                "Данные должны быть двумерным массивом: "
                "строки - моменты времени, столбцы - измерения."
            )

        self.n_subgroups, self.subgroup_size = self.data.shape

        if datetimes is None:
            self.datetimes = np.arange(1, self.n_subgroups + 1)
        else:
            self.datetimes = np.asarray(datetimes)[:self.n_subgroups]

        # Основной график
        self.stat_main = None
        self.cl_main = None
        self.ucl_main = None
        self.lcl_main = None

        # Второй график
        self.stat_disp = None
        self.cl_disp = None
        self.ucl_disp = None
        self.lcl_disp = None

        # Подписи
        self.main_label = None
        self.disp_label = None

    @abstractmethod
    def fit(self, baseline_mask=None, method='classic'):
        pass

    def plot(
            self,
            start=None,
            end=None,
            last_n=30):

        from spc_lib.visualization.plotly_engine import plot_control_chart

        return plot_control_chart(
            self,
            start=start,
            end=end,
            last_n=last_n
        )
