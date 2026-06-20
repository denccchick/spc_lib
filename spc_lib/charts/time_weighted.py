import numpy as np
import warnings
from spc_lib.core.base_chart import BaseControlChart


class CUSUMChart(BaseControlChart):
    """
    Кумулятивная сумма (CUSUM) - контрольная карта для обнаружения
    малых сдвигов процесса.

    Параметры:
    ----------
    data : array-like
        Данные подгрупп (2D массив) или индивидуальные значения (1D)
    datetimes : array-like, optional
        Временные метки
    target : float, optional
        Целевое значение процесса
    usl, lsl : float, optional
        Верхняя/нижняя границы спецификации
    h : float, default=5
        Параметр принятия решения (расстояние от нулевой линии до границы)
    k : float, default=0.5
        Параметр ссылочного значения (обычно 0.5 для стандартного CUSUM)
    std_est : float, optional
        Оценка стандартного отклонения. Если None, рассчитывается из данных.
    """

    def __init__(self, data, datetimes=None, target=None, usl=None, lsl=None,
                 h=5.0, k=0.5, std_est=None):
        super().__init__(data, datetimes, target, usl, lsl)
        self.h = h
        self.k = k
        self.std_est = std_est
        self.main_label = "CUSUM: Кумулятивная сумма (верхняя/нижняя)"
        self.disp_label = None

        self.cusum_upper = None
        self.cusum_lower = None
        self.sigma_est = None

    def fit(self, baseline_mask=None, method='classic'):
        """
        Методы:
        - 'classic': стандартный CUSUM с параметрами h и k
        - 'percentiles': эмпирические пределы на основе процентилей
        - 'made': робастный CUSUM на основе MAD
        """
        if baseline_mask is None:
            baseline_mask = np.ones(self.n_subgroups, dtype=bool)

        if self.data.ndim == 2:
            if self.subgroup_size == 1:
                x = self.data.flatten()
                n = 1
            else:
                x = np.mean(self.data, axis=1)
                n = self.subgroup_size
        else:
            warnings.warn(
                "Передан 1D массив. Рекомендуется использовать 2D формат: data.reshape(-1, 1)",
                UserWarning
            )
            x = self.data
            n = 1

        base_x = x[baseline_mask]

        if self.std_est is None:
            if n > 1 and method == 'classic':
                r = np.ptp(self.data[baseline_mask], axis=1)
                d2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326,
                      6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
                sigma_est = np.mean(r) / d2.get(n, 1.128)
            else:
                sigma_est = np.std(base_x, ddof=1)
        else:
            sigma_est = self.std_est

        if self.target is not None:
            target = self.target
        elif method == 'classic':
            target = np.mean(base_x)
        else:
            target = np.median(base_x)

        # Нормировка
        if sigma_est > 0:
            z = (x - target) / sigma_est
        else:
            z = x - target
            warnings.warn("Сигма равна 0, используется ненормированная статистика", UserWarning)

        self.cusum_upper = np.zeros(len(x))
        self.cusum_lower = np.zeros(len(x))

        if method == 'classic':
            for i in range(1, len(x)):
                self.cusum_upper[i] = max(0, self.cusum_upper[i-1] + z[i] - self.k)
                self.cusum_lower[i] = max(0, self.cusum_lower[i-1] - z[i] - self.k)

            self.cl_main = 0
            self.ucl_main = self.h
            self.lcl_main = self.h

        elif method == 'percentiles':
            for i in range(1, len(x)):
                self.cusum_upper[i] = max(0, self.cusum_upper[i-1] + z[i])
                self.cusum_lower[i] = max(0, self.cusum_lower[i-1] - z[i])

            base_cusum_upper = self.cusum_upper[baseline_mask]
            base_cusum_lower = self.cusum_lower[baseline_mask]

            self.cl_main = 0
            self.ucl_main = np.percentile(base_cusum_upper, 99.865)
            self.lcl_main = np.percentile(base_cusum_lower, 99.865)

        elif method == 'made':
            mad = np.median(np.abs(base_x - np.median(base_x)))
            robust_sigma = 1.4826 * mad if mad > 0 else 1e-10

            z_robust = (x - target) / robust_sigma if robust_sigma > 0 else x - target

            for i in range(1, len(x)):
                self.cusum_upper[i] = max(0, self.cusum_upper[i-1] + z_robust[i] - self.k)
                self.cusum_lower[i] = max(0, self.cusum_lower[i-1] - z_robust[i] - self.k)

            self.cl_main = 0
            self.ucl_main = self.h
            self.lcl_main = self.h
            sigma_est = robust_sigma

        else:
            raise ValueError(f"Неизвестный метод: {method}")

        self.target = target
        self.sigma_est = sigma_est
        self.stat_main = x

        return self

    def get_cusum_stats(self):
        """Возвращает верхнюю и нижнюю CUSUM статистики"""
        return self.cusum_upper, self.cusum_lower


class EWMAChart(BaseControlChart):
    """
    Экспоненциально взвешенное скользящее среднее (EWMA)
    - контрольная карта для обнаружения малых сдвигов процесса.

    Параметры:
    ----------
    data : array-like
        Данные подгрупп (2D массив) или индивидуальные значения (1D)
    datetimes : array-like, optional
        Временные метки
    target : float, optional
        Целевое значение процесса
    usl, lsl : float, optional
        Верхняя/нижняя границы спецификации
    lambda_ : float, default=0.2
        Весовой коэффициент (0 < lambda <= 1)
    L : float, default=3
        Коэффициент ширины контрольных границ
    std_est : float, optional
        Оценка стандартного отклонения. Если None, рассчитывается из данных.
    """

    def __init__(self, data, datetimes=None, target=None, usl=None, lsl=None,
                 lambda_=0.2, L=3, std_est=None):
        super().__init__(data, datetimes, target, usl, lsl)
        self.lambda_ = lambda_
        self.L = L
        self.std_est = std_est
        self.main_label = "EWMA: Экспоненциально взвешенное скользящее среднее"
        self.disp_label = None

        self.ewma_values = None
        self.ewma_sigma = None
        self.sigma_est = None

    def fit(self, baseline_mask=None, method='classic'):
        """
        Методы:
        - 'classic': стандартная EWMA с фиксированными пределами
        - 'percentiles': эмпирические пределы на основе процентилей
        - 'made': робастная EWMA на основе MAD
        """
        if baseline_mask is None:
            baseline_mask = np.ones(self.n_subgroups, dtype=bool)

        # Извлечение данных
        if self.data.ndim == 2:
            if self.subgroup_size == 1:
                x = self.data.flatten()
                n = 1
            else:
                x = np.mean(self.data, axis=1)
                n = self.subgroup_size
        else:
            warnings.warn(
                "Передан 1D массив. Рекомендуется использовать 2D формат: data.reshape(-1, 1)",
                UserWarning
            )
            x = self.data
            n = 1

        base_x = x[baseline_mask]

        # Оценка стандартного отклонения
        if self.std_est is None:
            if n > 1 and method == 'classic':
                r = np.ptp(self.data[baseline_mask], axis=1)
                d2 = {2: 1.128, 3: 1.693, 4: 2.059, 5: 2.326,
                      6: 2.534, 7: 2.704, 8: 2.847, 9: 2.970, 10: 3.078}
                sigma_est = np.mean(r) / d2.get(n, 1.128)
            else:
                sigma_est = np.std(base_x, ddof=1)
        else:
            sigma_est = self.std_est

        # Целевое значение
        if self.target is not None:
            target = self.target
        elif method == 'classic':
            target = np.mean(base_x)
        else:
            target = np.median(base_x)

        lambda_ = self.lambda_
        L = self.L

        # Расчет EWMA
        self.ewma_values = np.zeros(len(x))
        self.ewma_values[0] = target

        for i in range(1, len(x)):
            self.ewma_values[i] = lambda_ * x[i] + (1 - lambda_) * self.ewma_values[i-1]

        # Стандартное отклонение EWMA: σ * sqrt(λ/(2-λ) * (1 - (1-λ)^(2i)))
        weights = np.zeros(len(x))
        for i in range(1, len(x) + 1):
            weights[i-1] = np.sqrt(lambda_ / (2 - lambda_) * (1 - (1 - lambda_)**(2*i)))

        self.ewma_sigma = sigma_est * weights

        # Расчет границ в зависимости от метода
        if method == 'classic':
            self.cl_main = target
            self.ucl_main = target + L * self.ewma_sigma
            self.lcl_main = target - L * self.ewma_sigma

        elif method == 'percentiles':
            base_ewma = self.ewma_values[baseline_mask]
            self.cl_main = np.median(base_ewma)
            self.ucl_main = np.percentile(base_ewma, 99.865)
            self.lcl_main = np.percentile(base_ewma, 0.135)

        elif method == 'made':
            mad = np.median(np.abs(base_x - np.median(base_x)))
            robust_sigma = 1.4826 * mad if mad > 0 else 1e-10

            self.ewma_sigma = robust_sigma * weights
            self.cl_main = target
            self.ucl_main = target + L * self.ewma_sigma
            self.lcl_main = target - L * self.ewma_sigma
            sigma_est = robust_sigma

        else:
            raise ValueError(f"Неизвестный метод: {method}")

        self.target = target
        self.sigma_est = sigma_est
        self.stat_main = self.ewma_values

        return self

    def get_ewma_values(self):
        """Возвращает EWMA значения и их стандартные отклонения"""
        return self.ewma_values, self.ewma_sigma
