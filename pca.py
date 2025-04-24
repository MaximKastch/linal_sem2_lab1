import math
import random
from typing import List, Optional, Tuple
from matplotlib.figure import Figure

import typing as tp



class Matrix:
    def __init__(
            self,
            rows: int,
            cols: int,
            matrix_data: tp.List[tp.List[float]],
    ) -> None:
        self.rows = rows
        self.cols = cols

        if len(matrix_data) != rows:
            print("длина", len(matrix_data))
            raise ValueError("Несоответствие количества строк")

        for row in matrix_data:
            if len(row) != cols:
                raise ValueError("Неравномерная длина строк")

        self.matrix_data = [row.copy() for row in matrix_data]

    def __add__(self, other: 'Matrix') -> 'Matrix':
        if not isinstance(other, Matrix):
            raise TypeError("Можно складывать только с матрицей")

        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Разный размер матриц")

        result = [
            [self.matrix_data[i][j] + other.matrix_data[i][j] for j in range(self.cols)]  # <-- Закрывающая скобка добавлена
            for i in range(self.rows)
        ]
        return Matrix(self.rows, self.cols, result)

    def __mul__(self, other: tp.Union['Matrix', float, int]) -> 'Matrix':
        if isinstance(other, (int, float)):
            return Matrix(
                self.rows,
                self.cols,
                [[elem * other for elem in row] for row in self.matrix_data]
            )

        if isinstance(other, Matrix):
            if self.cols != other.rows:
                raise ValueError("Неподходящие размеры для умножения")

            result = [
                [
                    sum(a * b for a, b in zip(row, col))
                    for col in zip(*other.matrix_data)
                ]
                for row in self.matrix_data
            ]
            return Matrix(self.rows, other.cols, result)

        raise TypeError("Неподдерживаемый тип операции")

    def __rmul__(self, other: tp.Union[float, int]) -> 'Matrix':
        return self.__mul__(other)

    def transpose(self) -> 'Matrix':
        return Matrix(
            self.cols,
            self.rows,
            [list(col) for col in zip(*self.matrix_data)]
        )

    def __str__(self) -> str:
        return '\n'.join(' '.join(map(str, row)) for row in self.matrix_data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return False
        return self.matrix_data == other.matrix_data

    def get_raw_determinant(self) -> float:
        if self.rows != self.cols:
            raise ValueError("The matrix is not square")
        matrix = [row.copy() for row in self.matrix_data]  # Создаем копию данных
        det = 1.0
        n = self.rows

        for i in range(n):
            max_row = i
            for k in range(i, n):
                if abs(matrix[k][i]) > abs(matrix[max_row][i]):
                    max_row = k

            if abs(matrix[max_row][i]) < 1e-10:  # Учет погрешности
                return 0.0

            if max_row != i:
                matrix[i], matrix[max_row] = matrix[max_row], matrix[i]
                det *= -1

            pivot = matrix[i][i]
            det *= pivot  # Умножение определителя на диагональный элемент

            for j in range(i + 1, n):
                factor = matrix[j][i] / pivot
                for k in range(i, n):
                    matrix[j][k] -= factor * matrix[i][k]

        return det


def gauss_solver(A: Matrix, b: Matrix, epsilon: float = 1e-10) -> List[Matrix]:
    """
    Вход:
    A: матрица коэффициентов (n×n). Используется класс Matrix из предыдущей
    ,→ лабораторной работы
    b: вектор правых частей (n×1)
    epsilon: число для учёта погрешностей при работе с float числами
    Выход:
    list[Matrix]: список базисных векторов решения системы
    Raises:
    ValueError: если система несовместна
    """
    n = A.rows
    augmented = []
    for i in range(n):
        row = [float(x) for x in A.matrix_data[i]] + [float(b.matrix_data[i][0])]
        augmented.append(row)

    # Прямой ход с выбором ведущего элемента
    for i in range(n):
        max_row = i
        for k in range(i, n):
            if abs(augmented[k][i]) > abs(augmented[max_row][i]):
                max_row = k

        if abs(augmented[max_row][i]) < epsilon:  # Проверка с допуском
            continue

        augmented[i], augmented[max_row] = augmented[max_row], augmented[i]


    # Проверка на совместность
    for row in augmented:
        if all(abs(x) < epsilon for x in row[:-1]) and abs(row[-1]) >= epsilon:
            raise ValueError("Система несовместна")

    # Обратный ход
    solution = [0.0] * n
    lead_cols = []
    for i in reversed(range(n)):
        lead = -1
        for j in range(n):
            if abs(augmented[i][j]) >= epsilon:  # Проверка с допуском
                lead = j
                break
        if lead == -1:
            continue

        lead_cols.append(lead)
        solution[lead] = augmented[i][-1]
        for j in range(lead + 1, n):
            solution[lead] -= augmented[i][j] * solution[j]
        if abs(augmented[i][lead]) >= epsilon:
            solution[lead] /= augmented[i][lead]
        else:
            solution[lead] = 0.0

    # Построение базиса ядра
    free_vars = [j for j in range(n) if j not in lead_cols]
    solutions = []
    if any(abs(x) >= epsilon for x in solution):
        solutions.append(Matrix(n, 1, [[x] for x in solution]))

    for var in free_vars:
        vec = [0.0] * n
        vec[var] = 1.0
        for i in reversed(range(n)):
            lead = -1
            for j in range(n):
                if abs(augmented[i][j]) >= epsilon:  # Проверка с допуском
                    lead = j
                    break
            if lead == -1 or lead in free_vars:
                continue

            sum_val = sum(augmented[i][k] * vec[k] for k in range(lead + 1, n))
            if abs(augmented[i][lead]) >= epsilon:
                vec[lead] = -sum_val / augmented[i][lead]

        if any(abs(x) >= epsilon for x in vec):  # Фильтрация нулевых векторов
            solutions.append(Matrix(n, 1, [[x] for x in vec]))

    return solutions



def center_data(X: 'Matrix') -> 'Matrix':
    """
    Вход: матрица данных X (n×m)
    Выход: центрированная матрица X_centered (n×m)
    """
    mean_vec = list()
    for col in range(X.cols):
        curr_sum = 0.0
        for row in range(X.rows):
            curr_sum += X.matrix_data[row][col]
        mean_vec.append(curr_sum / X.rows)
    meaned_data = []
    for row in range(X.rows):
        meaned_data.append([mean_vec[col] for col in range(X.cols)])

    X_centered = [
        [X.matrix_data[row][col] - mean_vec[col] for col in range(X.cols)] for row in range(X.rows)
                    ]
    centered_data = Matrix(X.rows, X.cols, X_centered)
    return centered_data

def covariance_matrix(X_centered: 'Matrix') -> 'Matrix':
    """
    Вход: центрированная матрица X_centered (n×m)
    Выход: матрица ковариаций C (m×m)
    """
    C = (X_centered.transpose()) * (X_centered)
    C = C * (1/(X_centered.rows-1))
    return C


def characteristic_polynomial(C: Matrix, lambda_: float) -> float:
    """
    Вход:
    C (Matrix): квадратная матрица (n×n)
    lambda_ (float): значение переменной лямбда
    Выход: float: значение характеристического многослена в точке лямбда
    """
    m = C.rows
    new_data = [
        [C.matrix_data[i][j] - (lambda_ if i == j else 0.0)
         for j in range(m)
         ] for i in range(m)]
    return Matrix(m, m, new_data).get_raw_determinant()







# Вычисляет границы интервалов собственных значений по теореме Гершгорина.
# Возвращает общий интервал (lower, upper), покрывающий все возможные вещественные собственные значения.
def gershgorin_bounds(matrix: Matrix) -> tuple:
    """
    Вход:
    matrix (Matrix): квадратная матрица (n×n)
    Выход: tuple: границы интервалов собственных значений в формате (lower, upper)
    """
    n = matrix.rows
    lower = float('inf')
    upper = -float('inf')

    for i in range(n):
        radius = sum(abs(matrix.matrix_data[i][j]) for j in range(n) if j != i)
        center = matrix.matrix_data[i][i]
        lower = min(lower, center - radius)
        upper = max(upper, center + radius)

    return (lower, upper)


# Вычисляет определитель матрицы (matrix - lambda * I).
def determinant_at_lambda(matrix: Matrix, lambda_val: float) -> float:
    """
    Вход:
    matrix (Matrix): квадратная матрица (n×n)
    lambda_val (float): значение лямбда

    Выход:
        float: определитель матрицы (matrix - λI)
    """
    n = matrix.rows
    # Создаем матрицу (matrix - λI) с явным указанием matrix_data
    matrix_data = [
        [
            matrix.matrix_data[i][j] if i != j
            else matrix.matrix_data[i][i] - lambda_val
            for j in range(n)
        ]
        for i in range(n)
    ]
    C_minus_lambda = Matrix(n, n, matrix_data)
    return C_minus_lambda.get_raw_determinant()


def find_eigenvalues(C: Matrix, tol: float = 1e-6) -> List[float]:
    """
    Вход:
    C: матрица ковариаций (m×m)
    eigenvalues: список собственных значений
    Выход: список собственных векторов (каждый вектор - объект Matrix)
"""
    eigenvalues = []

    # Определение границ интервалов по Гершгорину
    lower_bound, upper_bound = gershgorin_bounds(C)
    search_step = (upper_bound - lower_bound) / 10000

    # Поиск интервалов смены знака определителя
    current_lambda = lower_bound
    prev_det = determinant_at_lambda(C, current_lambda)

    while current_lambda <= upper_bound:
        current_lambda += search_step
        current_det = determinant_at_lambda(C, current_lambda)

        if prev_det * current_det < 0 or abs(current_det) < tol:
            # Уточнение корня методом бисекции
            a = current_lambda - search_step
            b = current_lambda

            while abs(b - a) > tol:
                mid = (a + b) / 2
                mid_det = determinant_at_lambda(C, mid)

                if mid_det * determinant_at_lambda(C, a) < 0:
                    b = mid
                else:
                    a = mid

            eigenvalue = (a + b) / 2
            eigenvalues.append(eigenvalue)

        prev_det = current_det

    eigenvalues.sort()

    return eigenvalues


def find_eigenvectors(C: Matrix, eigenvalues: List[float]) -> List[Matrix]:
    """
    Вход:
    C: матрица ковариаций (m×m)
    eigenvalues: список собственных значений
    Выход: список собственных векторов (каждый вектор - объект Matrix)
    """
    eigenvectors = []
    n = C.rows
    epsilon = 1e-8  # Единый допуск

    for lambda_val in eigenvalues:
        # Создание матрицы (C - λI) с явным преобразованием в float
        matrix_data = [
            [
                float(C.matrix_data[i][j] - (lambda_val if i == j else 0.0))
                for j in range(n)
            ]
            for i in range(n)
        ]
        A = Matrix(n, n, matrix_data)
        b = Matrix(n, 1, [[0.0] for _ in range(n)])  # 0.0 вместо 0

        try:
            solutions = gauss_solver(A, b, epsilon)
            # Фильтрация векторов по норме
            for vec in solutions:
                norm = sum(x**2 for row in vec.matrix_data for x in row)**0.5
                if norm > epsilon:
                    eigenvectors.append(vec)
        except Exception as e:
            print(f"Ошибка для λ={lambda_val:.6f}: {str(e)}")

    return eigenvectors


from typing import List


def explained_variance_ratio(eigenvalues: List[float], k: int) -> float:
    """
    Вычисляет долю объяснённой дисперсии для первых k компонент.
    Вход:
        eigenvalues: список собственных значений
        k: число компонент
    Выход:
        доля объяснённой дисперсии в диапазоне [0, 1]
    """
    if not eigenvalues:
        raise ValueError("Список собственных значений пустой")
    if k <= 0 or k > len(eigenvalues):
        raise ValueError(f"неправильный диапазон [1, {len(eigenvalues)}]")

    sum_k = sum(eigenvalues[:k])

    total = sum(eigenvalues)

    if total == 0:
        return 0.0

    return sum_k / total


def pca(X: Matrix, k: int) -> Tuple[Matrix, float]:
    """
    Вход:
    X: матрица данных (n×m)
    k: число главных компонент
    Выход:
    X_proj: проекция данных (n×k)
    : доля объяснённой дисперсии
    """

    X_centered = center_data(X)

    C = covariance_matrix(X_centered)


    eigenvalues = find_eigenvalues(C, tol=1e-10)
    eigenvectors = find_eigenvectors(C, eigenvalues)

    # Сортировка собственных значений и векторов по убыванию
    sorted_pairs = sorted(zip(eigenvalues, eigenvectors), key=lambda x: -x[0])
    eigenvalues_sorted, eigenvectors_sorted = zip(*sorted_pairs) if sorted_pairs else ([], [])


    # Преобразование векторов в списки
    projection_data = [
        [vec.matrix_data[i][0] for i in range(vec.rows)]
        for vec in eigenvectors_sorted[:k]
    ]

    # Создание матрицы проекции (m x k)
    projection_matrix = Matrix(k, X.cols, projection_data).transpose()

    # Умножение центрированных данных на матрицу проекции
    X_proj = X_centered * projection_matrix

    # Шаг 5: Вычисление доли объяснённой дисперсии
    explained_var = explained_variance_ratio(eigenvalues_sorted, k)

    return X_proj, explained_var



def plot_pca_projection(X_proj: Matrix) -> Figure:
    """
    Вход: проекция данных X_proj (n×2)
    Выход: объект Figure из Matplotlib
    """
    # Извлечение данных из матрицы проекции
    pc1 = [row[0] for row in X_proj.matrix_data]
    pc2 = [row[1] for row in X_proj.matrix_data]

    # Создание фигуры и осей
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)

    # Построение scatter plot
    ax.scatter(pc1, pc2, alpha=0.5, c='blue', edgecolors='w', s=40)

    # Настройка оформления
    ax.set_title("Проекция данных на главные компоненты", pad=15)
    ax.set_xlabel("Первая главная компонента (PC1)")
    ax.set_ylabel("Вторая главная компонента (PC2)")
    ax.grid(True, linestyle='--', alpha=0.7)

    return fig




def reconstruction_error(X_orig: 'Matrix', X_recon: 'Matrix') -> float:
    """
    X_orig: исходные данные (n×m)
    X_recon: восстановленные данные (n×m)
    Выход: значение MSE
    """
    # Проверка совпадения размеров матриц
    if X_orig.rows != X_recon.rows or X_orig.cols != X_recon.cols:
        raise ValueError("Матрицы имеют разные размеры")

    total_error = 0.0
    # Суммирование квадратов разностей элементов
    for i in range(X_orig.rows):
        for j in range(X_orig.cols):
            diff = X_orig.matrix_data[i][j] - X_recon.matrix_data[i][j]
            total_error += diff ** 2

    # Вычисление MSE
    mse = total_error / (X_orig.rows * X_orig.cols)
    return mse


def auto_select_k(eigenvalues: List[float], threshold: float = 0.95) -> int:
    """
    Вход:
    eigenvalues (List[float]): список собственных значений
    threshold (float, optional): порог объясненной дисперсии (0.95 по умолчанию)
    Выход:
    int: минимальное количество компонент k, покрывающих долю дисперсии >= threshold
    raises:
    ValueError: если список eigenvalues пуст или threshold вне диапазона (0, 1]
    """
    if not eigenvalues:
        raise ValueError("Список собственных значений пуст")
    if not (0 < threshold <= 1):
        raise ValueError("Порог должен быть в диапазоне (0, 1]")

    # Сортировка по убыванию
    sorted_eigen = sorted(eigenvalues, reverse=True)

    total_variance = sum(sorted_eigen)
    cumulative = 0.0

    for k, value in enumerate(sorted_eigen, 1):
        cumulative += value
        ratio = cumulative / total_variance
        if ratio > threshold:
            return k

    return len(sorted_eigen)


def handle_missing_values(X: Matrix) -> Matrix:
    """
    Вход: матрица данных X (n×m) с возможными NaN
    Выход: матрица данных X_filled (n×m) без NaN
    """
    col_means = []
    for col in range(X.cols):
        sum_val = 0.0
        count = 0
        for row in range(X.rows):
            val = X.matrix_data[row][col]
            if not math.isnan(val):
                sum_val += val
                count += 1

        # Если все значения NaN - используем 0
        if count == 0:
            col_means.append(0.0)
        else:
            col_means.append(sum_val / count)

    # Замена NaN на средние/0
    new_data = []
    for row in X.matrix_data:
        new_row = [
            col_means[col] if math.isnan(val) else val
            for col, val in enumerate(row)
        ]
        new_data.append(new_row)

    return Matrix(X.rows, X.cols, new_data)


def add_noise_and_compare(X: Matrix, noise_level: float = 0.1) -> dict:
    """
    Вход:
    X: матрица данных (n×m)
    noise_level: уровень шума (доля от стандартного отклонения)
    Выход: результаты PCA до и после добавления шума.
    В этом задании можете проявить творческие способности, поэтому выходные данные не
    ,→ типизированы.
    """
    # 1. Выполняем PCA для исходных данных
    X_centered = center_data(X)
    C = covariance_matrix(X_centered)
    eigenvalues = find_eigenvalues(C, 1e-6)
    eigenvectors = find_eigenvectors(C, eigenvalues)

    # 2. Генерируем шум
    noisy_data = []
    for row in X.matrix_data:
        noisy_row = [
            val + random.gauss(0, noise_level * math.sqrt(abs(val)))
            for val in row
        ]
        noisy_data.append(noisy_row)

    X_noisy = Matrix(X.rows, X.cols, noisy_data)

    # 3. Выполняем PCA для зашумленных данных
    X_noisy_centered = center_data(X_noisy)
    C_noisy = covariance_matrix(X_noisy_centered)
    eigenvalues_noisy = find_eigenvalues(C_noisy, 1e-6)

    # 4. Сравниваем результаты
    return {
        "original_variances": eigenvalues,
        "noisy_variances": eigenvalues_noisy,
        "components_diff": sum(abs(a - b) for a, b in zip(eigenvalues, eigenvalues_noisy)),
        "explained_variance_diff": explained_variance_ratio(eigenvalues, 2) -
                                   explained_variance_ratio(eigenvalues_noisy, 2)
    }


def evaluate_model(X: Matrix) -> float:
    """
    Вход:
    X (Matrix): матрица данных с целевой переменной в последнем столбце
    Выход:
    float: точность классификации по метрике 1-NN (значение от 0 до 1)
    """
    # Разделение на признаки и целевую переменную
    y = [row[-1] for row in X.matrix_data]

    # Простейший 1-NN классификатор
    correct = 0
    for i in range(len(X.matrix_data)):
        min_dist = float('inf')
        pred = -1
        for j in range(len(X.matrix_data)):
            if i == j: continue
            dist = sum((a - b) ** 2 for a, b in zip(X.matrix_data[i], X.matrix_data[j]))
            if dist < min_dist:
                min_dist = dist
                pred = y[j]
        if pred == y[i]:
            correct += 1
    return correct / len(y)


def apply_pca_to_dataset(dataset_path: str, k: int) -> Tuple[Matrix, float]:
    """
    Вход:
    dataset_name: название датасета
    k: число главных компонент
    Выход: кортеж (проекция данных, качество модели)
    """
    with open(dataset_path, 'r', encoding='utf-8') as f:
        numeric_data = []
        for line in f:
            try:
                values = line.strip().split(',')
                numeric_row = []
                for v in values:
                    try:
                        numeric_row.append(float(v))
                    except:
                        numeric_row.append(math.nan)
                numeric_data.append(numeric_row)
            except:
                continue

    # Обработка пропусков и пустых колонок
    max_len = max(len(row) for row in numeric_data) if numeric_data else 0
    numeric_data = [row + [math.nan] * (max_len - len(row)) for row in numeric_data]

    columns = list(zip(*numeric_data))
    filtered_columns = [col for col in columns if not all(math.isnan(v) for v in col)]

    if not filtered_columns:
        raise ValueError("Все колонки пустые")

    if len(filtered_columns) < 2:
        raise ValueError("Недостаточно столбцов")

    filtered_data = list(zip(*filtered_columns))
    filtered_data = [list(row) for row in filtered_data]

    # Создание матрицы с выделением целевой переменной
    X = Matrix(len(filtered_data), len(filtered_columns) - 1,
               [row[:-1] for row in filtered_data])
    y = [row[-1] for row in filtered_data]

    # Обработка пропусков
    X_clean = handle_missing_values(X)

    # Оценка исходных данных
    orig_score = evaluate_model(Matrix(X_clean.rows, X_clean.cols + 1,
                                       [X_clean.matrix_data[i] + [y[i]]
                                        for i in range(X_clean.rows)]))

    X_proj, _ = pca(X_clean, k)

    # Оценка после PCA
    pca_score = evaluate_model(Matrix(X_proj.rows, X_proj.cols + 1,
                                      [X_proj.matrix_data[i] + [y[i]]
                                       for i in range(X_proj.rows)]))

    print(f"Качество до PCA: {orig_score:.2f}, после PCA: {pca_score:.2f}")

    return X_proj, pca_score





