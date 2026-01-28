import numpy as np
import h5py
import scipy.io
from pathlib import Path


def debug_transposition_issue(mat_file_path):
    """Анализирует и демонстрирует проблему транспонирования"""

    # 1. Открываем через h5py напрямую
    print("1. Анализ через h5py:")
    with h5py.File(mat_file_path, 'r') as f:
        for key in f.keys():
            if isinstance(f[key], h5py.Dataset):
                dataset = f[key]
                data_raw = dataset[()]
                print(f"   {key}: raw shape = {data_raw.shape}, dtype = {data_raw.dtype}")

                # Показываем порядок хранения
                print(f"      C-contiguous: {data_raw.flags['C_CONTIGUOUS']}")
                print(f"      F-contiguous: {data_raw.flags['F_CONTIGUOUS']}")

                # Показываем первые значения
                if data_raw.ndim == 2:
                    print(f"      Первые значения (raw):")
                    print(f"      {data_raw[0, 0]:.6f}, {data_raw[0, 1]:.6f}, {data_raw[0, 2]:.6f}")
                    print(f"      {data_raw[1, 0]:.6f}, {data_raw[1, 1]:.6f}, {data_raw[1, 2]:.6f}")

                    # Транспонируем
                    data_transposed = data_raw.T
                    print(f"\n      Первые значения (транспонированный):")
                    print(
                        f"      {data_transposed[0, 0]:.6f}, {data_transposed[0, 1]:.6f}, {data_transposed[0, 2]:.6f}")
                    print(
                        f"      {data_transposed[1, 0]:.6f}, {data_transposed[1, 1]:.6f}, {data_transposed[1, 2]:.6f}")

    # 2. Открываем через scipy (если возможно)
    print("\n2. Анализ через scipy.io:")
    try:
        mat_data = scipy.io.loadmat(mat_file_path)
        for key in mat_data.keys():
            if not key.startswith('__'):
                data = mat_data[key]
                if isinstance(data, np.ndarray):
                    print(f"   {key}: shape = {data.shape}, dtype = {data.dtype}")
                    print(f"      C-contiguous: {data.flags['C_CONTIGUOUS']}")
                    print(f"      F-contiguous: {data.flags['F_CONTIGUOUS']}")

                    if data.ndim == 2:
                        print(f"      Первые значения (scipy):")
                        print(f"      {data[0, 0]:.6f}, {data[0, 1]:.6f}, {data[0, 2]:.6f}")
                        print(f"      {data[1, 0]:.6f}, {data[1, 1]:.6f}, {data[1, 2]:.6f}")
    except:
        print("   Не удалось загрузить через scipy (вероятно v7.3)")

    # 3. Создаем тестовые данные для проверки
    print("\n3. Тест на понимание порядка хранения:")

    # Создаем тестовый массив как бы из MATLAB
    test_matlab_like = np.array([[1, 2, 3],
                                 [4, 5, 6]], order='F')  # Column-major как в MATLAB

    test_numpy_like = np.array([[1, 2, 3],
                                [4, 5, 6]], order='C')  # Row-major как в NumPy

    print(f"   MATLAB-style (F): {test_matlab_like.flags['F_CONTIGUOUS']}")
    print(f"   NumPy-style (C): {test_numpy_like.flags['C_CONTIGUOUS']}")

    # Показываем как они хранятся в памяти
    print(f"\n   Плоское представление MATLAB-style: {test_matlab_like.flatten()}")
    print(f"   Плоское представление NumPy-style: {test_numpy_like.flatten()}")

# Использование:
debug_transposition_issue('/home/linkoln-xd/python_projects/fordew/bm/1/1_1_1-pr_Page_141-PZU0001.mat')