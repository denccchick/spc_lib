import numpy as np
from spc_lib.core.base_chart import BaseControlChart

# Константы Монтгомери (n: A2, A3, D3, D4, B3, B4)
SPC_CONSTANTS = {
    2: (1.880, 2.659, 0.0,   3.267, 0.0,   3.267),
    3: (1.023, 1.954, 0.0,   2.574, 0.0,   2.568),
    4: (0.729, 1.628, 0.0,   2.282, 0.0,   2.266),
    5: (0.577, 1.427, 0.0,   2.114, 0.0,   2.089),
    6: (0.483, 1.287, 0.0,   2.004, 0.030, 1.970),
    7: (0.419, 1.182, 0.076, 1.924, 0.118, 1.882),
    8: (0.373, 1.099, 0.136, 1.864, 0.185, 1.815),
    9: (0.337, 1.032, 0.184, 1.816, 0.239, 1.761),
    10: (0.308, 0.975, 0.223, 1.777, 0.284, 1.716)
}

class XBarRChart(BaseControlChart):
    def fit(self, baseline_mask=None, method='classic'):
        if baseline_mask is None:
            baseline_mask = np.ones(self.n_subgroups, dtype=bool)

        n = self.subgroup_size

        self.stat_main = np.mean(self.data, axis=1)
        self.stat_disp = np.ptp(self.data, axis=1)

        base_xbar = self.stat_main[baseline_mask]
        base_r = self.stat_disp[baseline_mask]

        if method == 'classic':
            if n not in SPC_CONSTANTS:
                raise ValueError("Для классического метода n должно быть от 2 до 10")
            A2, _, D3, D4, _, _ = SPC_CONSTANTS[n]

            self.cl_main = np.mean(base_xbar)
            self.cl_disp = np.mean(base_r)

            self.ucl_main = self.cl_main + A2 * self.cl_disp
            self.lcl_main = self.cl_main - A2 * self.cl_disp

            self.ucl_disp = D4 * self.cl_disp
            self.lcl_disp = D3 * self.cl_disp

        elif method == 'percentiles':
            self.cl_main = np.median(base_xbar)
            self.ucl_main = np.percentile(base_xbar, 99.865)
            self.lcl_main = np.percentile(base_xbar, 0.135)

            self.cl_disp = np.median(base_r)
            self.ucl_disp = np.percentile(base_r, 99.865)
            self.lcl_disp = np.percentile(base_r, 0.135)

        elif method == 'made':
            self.cl_main = np.median(base_xbar)
            mad_xbar = np.median(np.abs(base_xbar - self.cl_main))
            robust_std_x = 1.4826 * mad_xbar

            self.ucl_main = self.cl_main + 3 * robust_std_x
            self.lcl_main = self.cl_main - 3 * robust_std_x

            self.cl_disp = np.median(base_r)
            mad_r = np.median(np.abs(base_r - self.cl_disp))
            robust_std_r = 1.4826 * mad_r

            self.ucl_disp = self.cl_disp + 3 * robust_std_r
            self.lcl_disp = self.cl_disp - 3 * robust_std_r

        elif method == 'algo_a':
            mu_star = np.median(base_xbar)
            s_star = 1.483 * np.median(np.abs(base_xbar - mu_star))

            for _ in range(3):
                delta = 1.5 * s_star
                x_star = np.clip(base_xbar, mu_star - delta, mu_star + delta)
                mu_star = np.mean(x_star)
                s_star = 1.134 * np.std(x_star, ddof=1)

            self.cl_main = mu_star
            self.ucl_main = mu_star + 3 * (s_star / np.sqrt(n))
            self.lcl_main = mu_star - 3 * (s_star / np.sqrt(n))

            self.cl_disp = np.median(base_r)
            self.ucl_disp = np.percentile(base_r, 99.865)
            self.lcl_disp = np.percentile(base_r, 0.135)
        else:
            raise ValueError(f"Неизвестный метод: {method}")

        return self

class XBarSChart(BaseControlChart):
    def fit(self, baseline_mask=None, method='classic'):
        if baseline_mask is None:
            baseline_mask = np.ones(self.n_subgroups, dtype=bool)

        n = self.subgroup_size

        self.stat_main = np.mean(self.data, axis=1)
        self.stat_disp = np.std(self.data, axis=1, ddof=1)

        base_xbar = self.stat_main[baseline_mask]
        base_s = self.stat_disp[baseline_mask]

        if method == 'classic':
            if n not in SPC_CONSTANTS:
                raise ValueError("Для классического метода n должно быть от 2 до 10")
            _, A3, _, _, B3, B4 = SPC_CONSTANTS[n]

            self.cl_main = np.mean(base_xbar)
            self.cl_disp = np.mean(base_s)

            self.ucl_main = self.cl_main + A3 * self.cl_disp
            self.lcl_main = self.cl_main - A3 * self.cl_disp

            self.ucl_disp = B4 * self.cl_disp
            self.lcl_disp = B3 * self.cl_disp

        elif method == 'percentiles':
            self.cl_main = np.median(base_xbar)
            self.ucl_main = np.percentile(base_xbar, 99.865)
            self.lcl_main = np.percentile(base_xbar, 0.135)

            self.cl_disp = np.median(base_s)
            self.ucl_disp = np.percentile(base_s, 99.865)
            self.lcl_disp = np.percentile(base_s, 0.135)
        else:
            raise ValueError(f"Метод {method} пока поддерживается только для Xbar-R карты")

        return self

class IMRChart(BaseControlChart):

    def fit(self, baseline_mask=None, method='classic'):

        if baseline_mask is None:
            baseline_mask = np.ones(self.n_subgroups, dtype=bool)

        # Если в строке несколько измерений —
        # усредняем их и получаем индивидуальные наблюдения
        x = np.mean(self.data, axis=1)

        # Moving Range
        mr = np.abs(np.diff(x))

        self.stat_main = x
        self.stat_disp = np.concatenate(([np.nan], mr))

        base_x = x[baseline_mask]

        # для MR первая точка отсутствует
        mr_mask = baseline_mask[1:] & baseline_mask[:-1]
        base_mr = mr[mr_mask]

        if method == 'classic':

            d2 = 1.128

            self.cl_main = np.mean(base_x)

            mr_bar = np.mean(base_mr)

            sigma_hat = mr_bar / d2

            self.ucl_main = self.cl_main + 3 * sigma_hat
            self.lcl_main = self.cl_main - 3 * sigma_hat

            self.cl_disp = mr_bar
            self.ucl_disp = 3.267 * mr_bar
            self.lcl_disp = 0

        elif method == 'percentiles':

            self.cl_main = np.median(base_x)
            self.ucl_main = np.percentile(base_x, 99.865)
            self.lcl_main = np.percentile(base_x, 0.135)

            self.cl_disp = np.median(base_mr)
            self.ucl_disp = np.percentile(base_mr, 99.865)
            self.lcl_disp = np.percentile(base_mr, 0.135)

        elif method == 'made':

            self.cl_main = np.median(base_x)

            mad_x = np.median(
                np.abs(base_x - self.cl_main)
            )

            sigma_x = 1.4826 * mad_x

            self.ucl_main = self.cl_main + 3 * sigma_x
            self.lcl_main = self.cl_main - 3 * sigma_x

            self.cl_disp = np.median(base_mr)

            mad_mr = np.median(
                np.abs(base_mr - self.cl_disp)
            )

            sigma_mr = 1.4826 * mad_mr

            self.ucl_disp = self.cl_disp + 3 * sigma_mr
            self.lcl_disp = max(
                0,
                self.cl_disp - 3 * sigma_mr
            )

        elif method == 'algo_a':

            mu_star = np.median(base_x)

            s_star = (
                1.483 *
                np.median(np.abs(base_x - mu_star))
            )

            for _ in range(3):

                delta = 1.5 * s_star

                x_star = np.clip(
                    base_x,
                    mu_star - delta,
                    mu_star + delta
                )

                mu_star = np.mean(x_star)

                s_star = (
                    1.134 *
                    np.std(x_star, ddof=1)
                )

            self.cl_main = mu_star

            self.ucl_main = (
                mu_star +
                3 * s_star
            )

            self.lcl_main = (
                mu_star -
                3 * s_star
            )

            self.cl_disp = np.median(base_mr)
            self.ucl_disp = np.percentile(base_mr, 99.865)
            self.lcl_disp = np.percentile(base_mr, 0.135)

        else:
            raise ValueError(
                f"Неизвестный метод: {method}"
            )

        return self
