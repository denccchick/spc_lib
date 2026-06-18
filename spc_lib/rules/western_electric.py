import numpy as np
from typing import List, Dict, Optional, Union
from datetime import datetime


class _WesternElectricRules:
    """Приватный класс. Не использовать напрямую."""

    def __init__(self, data: np.ndarray, center: float, sigma: float):
        self.data = np.asarray(data)
        self.center = center
        self.sigma = sigma if sigma != 0 else 1e-10
        self.n = len(data)
        self.z_scores = (self.data - self.center) / self.sigma
        self.signs = np.sign(self.z_scores)

    def _rule_1(self) -> List[int]:
        return np.where(np.abs(self.z_scores) > 3)[0].tolist()

    def _rule_2(self) -> List[int]:
        violations = []
        for i in range(self.n - 8):
            if np.all(self.signs[i:i+9] == 1) or np.all(self.signs[i:i+9] == -1):
                violations.extend(range(i, i+9))
        return list(set(violations))

    def _rule_3(self) -> List[int]:
        violations = []
        for i in range(self.n - 5):
            if np.all(np.diff(self.data[i:i+6]) > 0) or np.all(np.diff(self.data[i:i+6]) < 0):
                violations.extend(range(i, i+6))
        return list(set(violations))

    def _rule_4(self) -> List[int]:
        violations = []
        for i in range(self.n - 13):
            signs_diff = np.sign(np.diff(self.data[i:i+14]))
            if np.all(signs_diff[:-1] * signs_diff[1:] == -1):
                violations.extend(range(i, i+14))
        return list(set(violations))

    def _rule_5(self) -> List[int]:
        violations = []
        for i in range(self.n - 2):
            in_zone_a = np.abs(self.z_scores[i:i+3]) >= 2
            if np.sum(in_zone_a) >= 2:
                signs_subset = self.signs[i:i+3][in_zone_a]
                if np.all(signs_subset == 1) or np.all(signs_subset == -1):
                    violations.extend(range(i, i+3))
        return list(set(violations))

    def _rule_6(self) -> List[int]:
        violations = []
        for i in range(self.n - 4):
            in_zone_b = np.abs(self.z_scores[i:i+5]) >= 1
            if np.sum(in_zone_b) >= 4:
                signs_subset = self.signs[i:i+5][in_zone_b]
                if np.all(signs_subset == 1) or np.all(signs_subset == -1):
                    violations.extend(range(i, i+5))
        return list(set(violations))

    def _rule_7(self) -> List[int]:
        violations = []
        for i in range(self.n - 14):
            if np.all(np.abs(self.z_scores[i:i+15]) <= 1):
                violations.extend(range(i, i+15))
        return list(set(violations))

    def _rule_8(self) -> List[int]:
        violations = []
        for i in range(self.n - 7):
            if np.all(np.abs(self.z_scores[i:i+8]) > 1):
                violations.extend(range(i, i+8))
        return list(set(violations))

    def check(self, rules: Optional[List[int]] = None) -> Dict[int, List[int]]:
        rules = rules or list(range(1, 9))
        methods = {
            1: self._rule_1, 2: self._rule_2, 3: self._rule_3, 4: self._rule_4,
            5: self._rule_5, 6: self._rule_6, 7: self._rule_7, 8: self._rule_8
        }
        return {r: methods[r]() for r in rules if r in methods}


def detect_violations(
    data: np.ndarray,
    center: float,
    sigma: float,
    last_n: Optional[int] = 30,
    date_from: Optional[Union[str, datetime]] = None,
    date_to: Optional[Union[str, datetime]] = None,
    dates: Optional[np.ndarray] = None,
    rules: Optional[List[int]] = None
) -> Dict[int, List[int]]:
    """
    Проверка правил Western Electric на одномерном массиве статистик.

    Parameters
    ----------
    data : np.ndarray
        Одномерный массив (stat_main или stat_disp)
    center, sigma : float
        Центр и сигма процесса
    last_n : int, default=30
        Последние N точек. None - все данные
    date_from, date_to : str или datetime
        Период фильтрации
    dates : np.ndarray
        Даты (обязательны при фильтрации по датам)
    rules : List[int]
        Номера правил (1-8). По умолчанию все.

    Returns
    -------
    Dict[int, List[int]]
        {номер_правила: [индексы нарушений]}
    """
    # Срез данных
    if date_from is not None or date_to is not None:
        if dates is None:
            raise ValueError("dates required for date filtering")

        dates = np.asarray(dates)
        mask = np.ones(len(data), dtype=bool)
        if date_from is not None:
            mask &= dates >= np.datetime64(date_from)
        if date_to is not None:
            mask &= dates <= np.datetime64(date_to)

        slice_data = data[mask]
        start_idx = np.where(mask)[0][0] if np.any(mask) else 0

    elif last_n is not None and last_n > 0:
        start_idx = max(0, len(data) - last_n)
        slice_data = data[start_idx:]
    else:
        start_idx = 0
        slice_data = data

    if len(slice_data) == 0:
        return {r: [] for r in (rules or list(range(1, 9)))}

    we = _WesternElectricRules(slice_data, center, sigma)
    violations = we.check(rules)

    return {r: [start_idx + i for i in indices] for r, indices in violations.items()}
