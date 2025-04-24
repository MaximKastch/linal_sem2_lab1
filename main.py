import unittest
import math

import numpy as np
import pandas as pd
import matplotlib as plt

from pca import Matrix, center_data, covariance_matrix, pca, handle_missing_values, find_eigenvalues, find_eigenvectors, plot_pca_projection, auto_select_k, apply_pca_to_dataset
from matplotlib import figure


class TestPCA(unittest.TestCase):
    def test_eigen_properties(self):
        """Тест основных свойств собственных векторов и значений"""
        try:
            # Загрузка и предобработка данных
            df = pd.read_csv("Math-Students.csv", encoding='unicode-escape')
            df_num = df.select_dtypes(include=['number']).dropna(axis=1, how='all')

            # Проверка минимальных требований к данным
            if df_num.shape[1] < 2:
                raise ValueError("Недостаточно числовых столбцов для анализа")

            # Конвертация в Matrix
            data = df_num.values.tolist()
            X = Matrix(rows=len(data), cols=len(data[0]), matrix_data=data)

            # Центрирование данных
            X_centered = center_data(X)
            C = covariance_matrix(X_centered)

            # Получение собственных значений и векторов
            eigenvalues = find_eigenvalues(C, tol=1e-10)
            eigenvectors = find_eigenvectors(C, eigenvalues)

            # Основные проверки
            with self.subTest("Проверка количества компонент"):
                self.assertEqual(len(eigenvalues), C.rows,
                                 "Количество собственных значений не совпадает с размерностью матрицы")

            with self.subTest("Проверка положительной определенности"):
                for val in eigenvalues:
                    self.assertGreaterEqual(val, -1e-6,
                                            f"Отрицательное собственное значение: {val}")


            # Вывод информации при успехе
            print("\nСобственные значения успешно прошли проверки:")
            for i, val in enumerate(sorted(eigenvalues, reverse=True), 1):
                print(f"λ_{i}: {val:.5f}")

        except FileNotFoundError:
            self.skipTest("Файл данных не найден")
        except ValueError as e:
            if "недостаточно" in str(e).lower():
                self.skipTest(f"Пропуск теста: {str(e)}")
            else:
                self.fail(f"Ошибка обработки данных: {str(e)}")
        except Exception as e:
            self.fail(f"Неожиданная ошибка: {str(e)}")

    def test_centering(self):
        print("\nЗапуск теста центрирования данных...")
        X = Matrix(2, 2, [[1.0, 2.0], [3.0, 4.0]])
        centered = center_data(X)

        means = [
            sum(col) / centered.rows
            for col in zip(*centered.matrix_data)
        ]

        for mean in means:
            self.assertAlmostEqual(mean, 0.0, delta=1e-6)

        print("Тест центрирования: данные успешно центрированы")

    def test_covariance_matrix_symmetry(self):
        print("\nЗапуск теста симметрии ковариационной матрицы...")
        X = Matrix(3, 2, [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        X_centered = center_data(X)
        C = covariance_matrix(X_centered)

        self.assertEqual(C, C.transpose())
        print("ковариационная матрица симметрична")

    def test_pca_reconstruction(self):
        print("\nЗапуск теста реконструкции PCA...")
        original = Matrix(3, 3, [[4, 1, 0], [1, 5, 2], [0, 2, 6]])
        X_proj, _ = pca(original, 2)

        self.assertEqual(X_proj.cols, 2)
        self.assertEqual(X_proj.rows, 3)
        print("Тест реконструкции: проекция создана")

    def test_missing_values_handling(self):
        with self.subTest("Проверка замены NaN"):
            X = Matrix(2, 2, [[math.nan, 2.0], [3.0, math.nan]])
            cleaned = handle_missing_values(X)
            expected = [
                [3.0, 2.0],
                [3.0, 2.0]
            ]
            for i in range(cleaned.rows):
                for j in range(cleaned.cols):
                    self.assertAlmostEqual(
                        cleaned.matrix_data[i][j],
                        expected[i][j],
                        delta=1e-6,
                        msg=f"Ошибка в элементе ({i},{j})"
                    )

    def test_pca_quality_on_dataset(self):
        print("\nЗапуск теста для pca на реальных данных из датасета...")
        try:
            X_proj, pca_score = apply_pca_to_dataset("Math-Students.csv", k=2)
            self.assertGreaterEqual(pca_score, 0.3)
            print(f"тест: качество {pca_score:.2f} соответствует ожиданиям")
        except FileNotFoundError:
            self.skipTest("Файл не найден")
        except ValueError as e:
            if "недостаточно столбцов" in str(e):
                self.skipTest("В данных недостаточно столбцов для PCA")
            else:
                self.fail(f"Ошибка: {str(e)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)




