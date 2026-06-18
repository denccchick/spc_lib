import numpy as np
from scipy import stats


def diagnose(data, alpha=0.05, iqr_multiplier=1.5, min_samples=15):
    """
    Диагностика распределения данных.

    Parameters
    ----------
    data : array-like
        Вектор (1D) или матрица (2D). Если матрица — усредняется по строкам.
    alpha : float, default=0.05
        Уровень значимости для статистических тестов
    iqr_multiplier : float, default=1.5
        Множитель для IQR (1.5 - классический)
    min_samples : int, default=15
        Минимальное количество наблюдений для проведения тестов

    Returns
    -------
    dict
        Результаты проверок: нормальность, автокорреляция, выбросы
    """
    # Валидация входных данных

    # 1. Проверка типа данных
    if not hasattr(data, '__len__') and not isinstance(data, (list, tuple, np.ndarray)):
        raise TypeError(f"Данные должны быть массивом, получен {type(data).__name__}")

    # 2. Приводим к numpy array
    data = np.asarray(data)

    # 3. Проверка на пустой массив
    if len(data) == 0:
        raise ValueError("Нет данных для анализа (пустой массив)")

    # 4. Если 2D — усредняем по строкам (axis=1)
    if data.ndim == 2:
        data = np.mean(data, axis=1)

    # 5. Проверка на бесконечности
    if np.any(np.isinf(data)):
        n_inf = np.isinf(data).sum()
        raise ValueError(f"Обнаружены бесконечные значения (inf): {n_inf} шт.")

    # 6. Удаление NaN с предупреждением
    n_nan = np.isnan(data).sum()
    if n_nan > 0:
        print(f"Предупреждение: удалено {n_nan} пропущенных значений (NaN)")
        data = data[~np.isnan(data)]

    # 7. Проверка на пустой массив после удаления NaN
    if len(data) == 0:
        raise ValueError("После удаления пропусков (NaN) данных не осталось")

    n = len(data)

    # 8. Если данных мало — возвращаем "недостаточно данных"
    if n < min_samples:
        return {
            'normality': f'Недостаточно данных (n={n} < {min_samples})',
            'autocorrelation': f'Недостаточно данных (n={n} < {min_samples})',
            'outliers': f'Недостаточно данных (n={n} < {min_samples})'
        }

    # 9. Проверка на все одинаковые значения (нулевая дисперсия)
    if np.all(data == data[0]):
        return {
            'normality': 'Нет вариативности (все значения одинаковы)',
            'autocorrelation': 'Нет вариативности (все значения одинаковы)',
            'outliers': 'Нет вариативности (все значения одинаковы)'
        }

    # Проверки
    normality = _check_normality(data, alpha)
    autocorr = _check_autocorrelation(data)
    outliers = _check_outliers(data, iqr_multiplier)

    return {
        'normality': normality,
        'autocorrelation': autocorr,
        'outliers': outliers
    }


def _check_normality(data, alpha):
    """Проверка нормальности (тест Шапиро-Уилк)"""
    n = len(data)

    # Для больших выборок используем Колмогорова-Смирнова
    if n > 5000:
        from scipy.stats import kstest, norm
        data_std = (data - np.mean(data)) / np.std(data, ddof=1)
        statistic, p_value = kstest(data_std, 'norm')
    else:
        statistic, p_value = stats.shapiro(data)

    return 'Нормальное' if p_value >= alpha else 'Не нормальное'


def _check_autocorrelation(data):
    """Проверка автокорреляции (тест Дарбина-Уотсона)"""
    diff = np.diff(data)
    numerator = np.sum(diff ** 2)
    denominator = np.sum(data ** 2)

    if denominator == 0:
        statistic = 0.0
    else:
        statistic = numerator / denominator

    if 1.5 <= statistic <= 2.5:
        return 'Нет автокорреляции'
    elif statistic < 1.5:
        return 'Положительная автокорреляция'
    else:
        return 'Отрицательная автокорреляция'


def _check_outliers(data, multiplier):
    """Поиск выбросов методом IQR"""
    n = len(data)

    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1

    if iqr == 0:
        return 'IQR = 0'

    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr

    outliers = data[(data < lower) | (data > upper)]
    count = len(outliers)
    percentage = (count / n) * 100

    if percentage < 1.0:
        return 'Выбросов почти нет'
    elif percentage < 5.0:
        return 'Выбросов нормальное количество'
    elif percentage < 10.0:
        return 'Много выбросов'
    else:
        return 'Очень много выбросов (тяжёлые хвосты)'
