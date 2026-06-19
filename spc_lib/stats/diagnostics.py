import numpy as np
from scipy import stats
from statsmodels.stats.stattools import durbin_watson

def diagnose(data, alpha=0.05, iqr_multiplier=1.5, min_samples=15):
    """
    Диагностика распределения данных перед построением контрольных карт.

    Parameters
    ----------
    data : array-like
        Вектор (1D) или матрица (2D). Если матрица — усредняется по строкам.
    alpha : float, default=0.05
        Уровень значимости для статистических тестов.
    iqr_multiplier : float, default=1.5
        Множитель для IQR (1.5 - классический предел для выбросов).
    min_samples : int, default=15
        Минимальное количество наблюдений для проведения тестов.

    Returns
    -------
    dict
        Результаты проверок: нормальность, автокорреляция, выбросы.
    """
    # 1. Валидация
    if not hasattr(data, '__len__') and not isinstance(data, (list, tuple, np.ndarray)):
        raise TypeError(f"Данные должны быть массивом, получен {type(data).__name__}")

    data = np.asarray(data, dtype=float)

    if len(data) == 0:
        raise ValueError("Нет данных для анализа (пустой массив)")

    # Если 2D — усредняем по строкам (axis=1), так как контрольная карта средних работает с X-bar
    if data.ndim == 2:
        data = np.mean(data, axis=1)

    # Очистка от бесконечностей
    if np.any(np.isinf(data)):
        n_inf = np.isinf(data).sum()
        raise ValueError(f"Обнаружены бесконечные значения (inf): {n_inf} шт.")

    # Очистка от NaN
    n_nan = np.isnan(data).sum()
    if n_nan > 0:
        print(f"Предупреждение: удалено {n_nan} пропущенных значений (NaN)")
        data = data[~np.isnan(data)]

    n = len(data)

    if n == 0:
        raise ValueError("После удаления пропусков (NaN) данных не осталось")

    # Проверка на минимальный размер выборки
    if n < min_samples:
        msg = f'Недостаточно данных (n={n} < {min_samples})'
        return {'normality': msg, 'autocorrelation': msg, 'outliers': msg}

    # Проверка на нулевую дисперсию
    if np.all(data == data[0]):
        msg = 'Нет вариативности (все значения одинаковы)'
        return {'normality': msg, 'autocorrelation': msg, 'outliers': msg}


    # 2. Тесты
    normality = _check_normality(data, alpha)
    autocorr = _check_autocorrelation(data)
    outliers = _check_outliers(data, iqr_multiplier)

    return {
        'normality': normality,
        'autocorrelation': autocorr,
        'outliers': outliers
    }


def _check_normality(data, alpha):
    """Проверка нормальности (Шапиро-Уилк или Колмогоров-Смирнов для больших данных)"""
    n = len(data)

    if n > 5000:
        # Для больших выборок Шапиро-Уилк может быть неточным, используем K-S тест
        data_std = (data - np.mean(data)) / np.std(data, ddof=1)
        _, p_value = stats.kstest(data_std, 'norm')
    else:
        _, p_value = stats.shapiro(data)

    return 'Нормальное' if p_value >= alpha else 'Не нормальное'


def _check_autocorrelation(data):
    """Проверка автокорреляции (Критерий Дарбина-Уотсона из statsmodels)"""
    # Дарбин-Уотсон тестирует остатки. Для простого ряда остатки = данные минус среднее.
    residuals = data - np.mean(data)
    dw_stat = durbin_watson(residuals)

    if 1.5 <= dw_stat <= 2.5:
        return 'Нет автокорреляции'
    elif dw_stat < 1.5:
        return 'Положительная автокорреляция'
    else:
        return 'Отрицательная автокорреляция'


def _check_outliers(data, multiplier):
    """Поиск выбросов методом межквартильного размаха (IQR)"""
    n = len(data)

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    if iqr == 0:
        return 'IQR = 0 (невозможно определить выбросы классическим методом)'

    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    # Считаем количество точек за пределами усов
    outliers_count = np.sum((data < lower) | (data > upper))
    percentage = (outliers_count / n) * 100

    if percentage < 1.0:
        return 'Выбросов почти нет (<1%)'
    elif percentage < 5.0:
        return 'Выбросов нормальное количество (1-5%)'
    elif percentage < 10.0:
        return 'Много выбросов (5-10%)'
    else:
        return 'Очень много выбросов (>10%, тяжелые хвосты)'
