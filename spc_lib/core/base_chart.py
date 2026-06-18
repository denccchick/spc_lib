import numpy as np
from abc import ABC, abstractmethod


class BaseControlChart(ABC):
    def __init__(self, data, datetimes=None):
        """
        Parameters
        ----------
        data : array-like
            Двумерный массив (n_subgroups × subgroup_size).

        datetimes : array-like, optional
            Метки времени для оси X.
        """

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

        # X-bar
        self.stat_main = None
        self.cl_main = None
        self.ucl_main = None
        self.lcl_main = None

        # R или s
        self.stat_disp = None
        self.cl_disp = None
        self.ucl_disp = None
        self.lcl_disp = None

    @abstractmethod
    def fit(self, baseline_mask=None, method="classic"):
        pass

    def plot(self, **kwargs):
        """
        Построение контрольной карты.
        """
        from spc_lib.visualization import plot_control_chart

        return plot_control_chart(self, **kwargs)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(n_subgroups={self.n_subgroups}, "
            f"subgroup_size={self.subgroup_size})"
        )
