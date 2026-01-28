import os
import h5py
import numpy as np
import scipy.io
import warnings
from pathlib import Path
import traceback
import json
from typing import Dict, Any, Optional, Tuple
import hashlib


class MatToNpyConverterDebug:
    def __init__(self, input_root='bm', output_root='bm_npy'):
        self.input_root = Path(input_root)
        self.output_root = Path(output_root)
        self.conversion_log = []

    def compute_array_hash(self, array: np.ndarray) -> str:
        """Вычисляет хеш массива для проверки целостности"""
        # Преобразуем в байты и вычисляем хеш
        if array.dtype == np.float64 or array.dtype == np.float32:
            # Для вещественных чисел используем округление для стабильности
            array_rounded = np.round(array, decimals=6)
            return hashlib.md5(array_rounded.tobytes()).hexdigest()[:8]
        else:
            return hashlib.md5(array.tobytes()).hexdigest()[:8]

    def compare_matrices(self, mat_data: np.ndarray, npy_data: np.ndarray, filename: str) -> Dict[str, Any]:
        """Сравнивает два массива и выявляет различия"""
        comparison = {
            'filename': filename,
            'shapes_equal': mat_data.shape == npy_data.shape,
            'dtypes_equal': mat_data.dtype == npy_data.dtype,
            'values_equal': False,
            'transposed_equal': False,
            'differences': {}
        }

        # 1. Проверяем совпадение форм
        comparison['differences']['shape'] = {
            'matlab': mat_data.shape,
            'numpy': npy_data.shape
        }

        # 2. Проверяем совпадение типов данных
        comparison['differences']['dtype'] = {
            'matlab': str(mat_data.dtype),
            'numpy': str(npy_data.dtype)
        }

        # 3. Если формы совпадают, проверяем значения
        if mat_data.shape == npy_data.shape:
            try:
                # Используем относительную погрешность для float
                if np.issubdtype(mat_data.dtype, np.floating):
                    abs_diff = np.abs(mat_data - npy_data)
                    rel_diff = abs_diff / (np.abs(mat_data) + 1e-10)
                    max_rel_diff = np.max(rel_diff)

                    comparison['differences']['max_relative_difference'] = float(max_rel_diff)
                    comparison['values_equal'] = max_rel_diff < 1e-10
                else:
                    comparison['values_equal'] = np.array_equal(mat_data, npy_data)

            except Exception as e:
                comparison['differences']['comparison_error'] = str(e)

        # 4. Проверяем, не является ли один массив транспонированным другого
        if mat_data.ndim == 2 and npy_data.ndim == 2:
            # Для 2D массивов проверяем транспонирование
            if mat_data.shape == npy_data.T.shape:
                # Проверяем значения транспонированного
                mat_transposed = mat_data.T
                try:
                    if np.issubdtype(mat_data.dtype, np.floating):
                        comparison['transposed_equal'] = np.allclose(mat_transposed, npy_data, rtol=1e-10)
                    else:
                        comparison['transposed_equal'] = np.array_equal(mat_transposed, npy_data)
                except:
                    pass

        # 5. Сравниваем хеши
        comparison['differences']['hash'] = {
            'matlab': self.compute_array_hash(mat_data),
            'numpy': self.compute_array_hash(npy_data)
        }

        return comparison

    def inspect_mat_file_deep(self, file_path: Path) -> Dict[str, Any]:
        """Детальный анализ .mat файла"""
        info = {
            'file': str(file_path),
            'version': None,
            'variables': [],
            'issues': []
        }

        try:
            # Определяем версию
            version = self.detect_mat_version(file_path)
            info['version'] = version

            if version == 'v7.3':
                with h5py.File(file_path, 'r') as f:
                    for key in f.keys():
                        var_info = {
                            'name': key,
                            'type': type(f[key]).__name__,
                            'shape': None,
                            'dtype': None,
                            'is_reference': False
                        }

                        if isinstance(f[key], h5py.Dataset):
                            dataset = f[key]
                            var_info['shape'] = dataset.shape
                            var_info['dtype'] = str(dataset.dtype)

                            # Проверяем, является ли это ссылкой
                            if dataset.dtype == 'object':
                                var_info['is_reference'] = True
                                try:
                                    # Пробуем получить данные
                                    data = dataset[()]
                                    if isinstance(data, h5py.Reference):
                                        ref_obj = f[data]
                                        var_info['referenced_type'] = type(ref_obj).__name__
                                        var_info['referenced_shape'] = ref_obj.shape if hasattr(ref_obj,
                                                                                                'shape') else None
                                except:
                                    pass

                        info['variables'].append(var_info)

                        # Проверяем потенциальные проблемы
                        if var_info.get('is_reference', False):
                            info['issues'].append(f"Variable '{key}' is a reference (may need special handling)")

            else:
                # Старый формат
                mat_data = scipy.io.loadmat(file_path)
                for key in mat_data.keys():
                    if not (key.startswith('__') and key.endswith('__')):
                        var = mat_data[key]
                        var_info = {
                            'name': key,
                            'type': type(var).__name__,
                            'shape': var.shape if hasattr(var, 'shape') else None,
                            'dtype': str(var.dtype) if hasattr(var, 'dtype') else None
                        }
                        info['variables'].append(var_info)

        except Exception as e:
            info['issues'].append(f"Error inspecting file: {e}")

        return info

    def load_mat_correctly(self, file_path: Path, variable_name: Optional[str] = None) -> Tuple[Any, Dict]:
        """
        Загружает .mat файл с учетом всех особенностей MATLAB

        Возвращает:
        - данные
        - метаинформация о загрузке
        """
        meta = {
            'file': str(file_path),
            'version': None,
            'loaded_variables': [],
            'warnings': [],
            'transpose_applied': False
        }

        version = self.detect_mat_version(file_path)
        meta['version'] = version

        if version == 'v7.3':
            # Загружаем через h5py
            data_dict = {}

            with h5py.File(file_path, 'r') as f:
                for key in f.keys():
                    if isinstance(f[key], h5py.Dataset):
                        dataset = f[key]

                        # Получаем данные
                        raw_data = dataset[()]

                        # Обрабатываем разные типы данных
                        if dataset.dtype == 'object':
                            # Object dtype - может быть ссылкой
                            if isinstance(raw_data, h5py.Reference):
                                # Это прямая ссылка
                                ref_obj = f[raw_data]
                                if isinstance(ref_obj, h5py.Dataset):
                                    data = ref_obj[()]
                                else:
                                    data = raw_data
                                    meta['warnings'].append(f"Variable '{key}' is a reference to non-dataset")
                            elif isinstance(raw_data, np.ndarray) and raw_data.dtype == np.object:
                                # Массив объектов
                                try:
                                    # Пробуем разыменовать все элементы
                                    dereferenced = []
                                    for item in raw_data.flat:
                                        if isinstance(item, h5py.Reference):
                                            dereferenced.append(f[item][()])
                                        else:
                                            dereferenced.append(item)
                                    data = np.array(dereferenced).reshape(raw_data.shape)
                                except:
                                    data = raw_data
                                    meta['warnings'].append(f"Could not dereference object array for '{key}'")
                            else:
                                data = raw_data
                        else:
                            data = raw_data

                        # Для массивов применяем транспонирование
                        if isinstance(data, np.ndarray) and data.ndim >= 2:
                            # Сохраняем оригинальную форму
                            meta['loaded_variables'].append({
                                'name': key,
                                'original_shape': data.shape,
                                'dtype': str(data.dtype)
                            })

                            # Транспонируем (column-major → row-major)
                            data = data.T
                            meta['transpose_applied'] = True

                        data_dict[key] = data

                # Выбираем данные для возврата
                if variable_name and variable_name in data_dict:
                    data = data_dict[variable_name]
                    meta['selected_variable'] = variable_name
                elif len(data_dict) == 1:
                    data = list(data_dict.values())[0]
                    meta['selected_variable'] = list(data_dict.keys())[0]
                else:
                    data = data_dict
                    meta['selected_variable'] = 'dict'

        else:
            # Старый формат
            mat_data = scipy.io.loadmat(file_path, squeeze_me=False, mat_dtype=True)

            # Убираем служебные переменные
            data_dict = {}
            for key in mat_data.keys():
                if not (key.startswith('__') and key.endswith('__')):
                    data = mat_data[key]

                    # Для массивов сохраняем как есть (scipy уже делает преобразования)
                    if isinstance(data, np.ndarray):
                        meta['loaded_variables'].append({
                            'name': key,
                            'shape': data.shape,
                            'dtype': str(data.dtype)
                        })

                    data_dict[key] = data

            # Выбираем данные для возврата
            if variable_name and variable_name in data_dict:
                data = data_dict[variable_name]
                meta['selected_variable'] = variable_name
            elif len(data_dict) == 1:
                data = list(data_dict.values())[0]
                meta['selected_variable'] = list(data_dict.keys())[0]
            else:
                data = data_dict
                meta['selected_variable'] = 'dict'

        return data, meta

    def convert_and_verify(self, mat_file_path: Path, npy_file_path: Optional[Path] = None) -> Dict[str, Any]:
        """
        Конвертирует файл и сразу проверяет результат

        Возвращает детальный отчет о конвертации
        """
        result = {
            'input_file': str(mat_file_path),
            'output_file': None,
            'success': False,
            'verification': {},
            'warnings': [],
            'errors': []
        }

        try:
            # 1. Сначала анализируем исходный файл
            file_info = self.inspect_mat_file_deep(mat_file_path)

            # 2. Загружаем данные из .mat файла
            mat_data, load_meta = self.load_mat_correctly(mat_file_path)
            result['load_meta'] = load_meta

            # 3. Создаем путь для выходного файла
            if npy_file_path is None:
                rel_path = mat_file_path.relative_to(self.input_root)
                npy_file_path = self.output_root / rel_path.with_suffix('.npy')

            npy_file_path.parent.mkdir(parents=True, exist_ok=True)
            result['output_file'] = str(npy_file_path)

            # 4. Сохраняем в .npy
            np.save(npy_file_path, mat_data, allow_pickle=True)

            # 5. Загружаем обратно для проверки
            npy_data = np.load(npy_file_path, allow_pickle=True)

            # 6. Сравниваем данные
            if isinstance(mat_data, np.ndarray) and isinstance(npy_data, np.ndarray):
                comparison = self.compare_matrices(mat_data, npy_data, mat_file_path.name)
                result['verification'] = comparison

                if not comparison['values_equal']:
                    if comparison['transposed_equal']:
                        result['warnings'].append("Arrays differ but one is transposed of the other")
                    else:
                        result['errors'].append("Arrays are different (not just transposed)")
                else:
                    result['success'] = True

            elif isinstance(mat_data, dict) and isinstance(npy_data, dict):
                # Для словарей проверяем ключи
                mat_keys = set(mat_data.keys())
                npy_keys = set(npy_data.keys())

                if mat_keys == npy_keys:
                    result['success'] = True
                    result['verification'] = {
                        'type': 'dict',
                        'keys_match': True,
                        'keys': list(mat_keys)
                    }
                else:
                    result['errors'].append(f"Dictionary keys differ: {mat_keys - npy_keys} vs {npy_keys - mat_keys}")
            else:
                # Простая проверка типов
                if type(mat_data) == type(npy_data):
                    result['success'] = True
                else:
                    result['errors'].append(f"Types differ: {type(mat_data)} vs {type(npy_data)}")

        except Exception as e:
            result['errors'].append(str(e))
            result['traceback'] = traceback.format_exc()

        return result

    def detect_mat_version(self, file_path: Path) -> str:
        """Определяет версию .mat файла"""
        try:
            with h5py.File(file_path, 'r') as f:
                return 'v7.3'
        except (OSError, IOError):
            try:
                scipy.io.loadmat(file_path, mat_dtype=True)
                return 'v7.0_or_older'
            except:
                return 'unknown'

    def batch_convert_with_verification(self) -> Dict[str, Any]:
        """
        Пакетная конвертация с проверкой каждого файла
        """
        print(f"🔍 Начинаем конвертацию с проверкой целостности данных...")

        # Находим все файлы
        mat_files = []
        for root, dirs, files in os.walk(self.input_root):
            for file in files:
                if file.lower().endswith('.mat'):
                    mat_files.append(Path(root) / file)

        if not mat_files:
            print("⚠️  Файлы .mat не найдены")
            return {}

        print(f"📊 Найдено {len(mat_files)} файлов для конвертации")

        # Создаем структуру директорий
        for mat_file in mat_files:
            rel_path = mat_file.relative_to(self.input_root)
            output_dir = self.output_root / rel_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)

        # Конвертируем и проверяем
        results = {}
        issues_count = 0

        for i, mat_file in enumerate(mat_files, 1):
            print(f"\n[{i}/{len(mat_files)}] Анализируем: {mat_file.relative_to(self.input_root)}")

            # Конвертируем с проверкой
            result = self.convert_and_verify(mat_file)

            # Сохраняем результат
            results[str(mat_file)] = result

            # Выводим краткий отчет
            if result['success']:
                print(f"   ✅ Успешно")
                if 'verification' in result and 'differences' in result['verification']:
                    shape_info = result['verification']['differences'].get('shape', {})
                    if 'matlab' in shape_info and 'numpy' in shape_info:
                        print(f"   📐 Форма: {shape_info['matlab']} → {shape_info['numpy']}")
            else:
                print(f"   ⚠️  Проблемы:")
                for error in result.get('errors', []):
                    print(f"      - {error}")
                issues_count += 1

            # Предупреждения
            for warning in result.get('warnings', []):
                print(f"   ⚠️  {warning}")

        # Сохраняем детальный отчет
        self.save_detailed_report(results)

        print(f"\n{'=' * 60}")
        print(f"📊 ИТОГОВЫЙ ОТЧЕТ:")
        print(f"   Всего файлов: {len(mat_files)}")
        print(f"   Успешно: {len([r for r in results.values() if r['success']])}")
        print(f"   С проблемами: {issues_count}")
        print(f"   Детальный отчет сохранен в: {self.output_root / 'conversion_report.json'}")

        return results

    def save_detailed_report(self, results: Dict[str, Any]):
        """Сохраняет детальный отчет о конвертации"""
        report_file = self.output_root / 'conversion_report.json'

        # Собираем статистику
        stats = {
            'total_files': len(results),
            'successful': sum(1 for r in results.values() if r['success']),
            'with_warnings': sum(1 for r in results.values() if r.get('warnings')),
            'with_errors': sum(1 for r in results.values() if r.get('errors')),
            'file_types': {},
            'common_issues': {}
        }

        # Анализируем проблемы
        all_issues = []
        for file_result in results.values():
            all_issues.extend(file_result.get('errors', []))

        from collections import Counter
        issue_counts = Counter(all_issues)
        stats['common_issues'] = dict(issue_counts.most_common(10))

        # Сохраняем полный отчет
        full_report = {
            'summary': stats,
            'conversion_details': results,
            'timestamp': str(np.datetime64('now'))
        }

        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(full_report, f, indent=2, ensure_ascii=False, default=str)


def test_single_file():
    """Тестирование на одном файле для отладки"""
    converter = MatToNpyConverterDebug()

    # Укажите путь к вашему файлу
    test_file = Path("test.mat")  # Замените на ваш файл

    if not test_file.exists():
        print(f"Файл {test_file} не найден")
        return

    # 1. Анализируем файл
    print("🔍 Анализируем структуру файла...")
    file_info = converter.inspect_mat_file_deep(test_file)

    print(f"\nВерсия: {file_info['version']}")
    print(f"Переменные в файле:")
    for var in file_info['variables']:
        print(f"  {var['name']}: {var['type']}, shape: {var['shape']}, dtype: {var['dtype']}")
        if var.get('is_reference'):
            print(f"    ⚠️  Это ссылка!")

    # 2. Загружаем и проверяем
    print(f"\n📥 Загружаем данные...")
    mat_data, meta = converter.load_mat_correctly(test_file)

    print(f"\nМетаинформация о загрузке:")
    print(f"  Выбрана переменная: {meta.get('selected_variable')}")
    print(f"  Транспонирование применено: {meta.get('transpose_applied')}")

    if isinstance(mat_data, np.ndarray):
        print(f"  Данные: shape={mat_data.shape}, dtype={mat_data.dtype}")
        print(f"  Пример данных (первые 3x3):")
        if mat_data.ndim >= 2:
            print(mat_data[:3, :3])
        elif mat_data.ndim == 1:
            print(mat_data[:10])

    # 3. Сохраняем и проверяем
    print(f"\n💾 Сохраняем и проверяем...")
    result = converter.convert_and_verify(test_file, Path("test_output.npy"))

    print(f"\nРезультат проверки:")
    if result['success']:
        print("  ✅ Данные совпадают")
    else:
        print("  ❌ Данные отличаются")

    if 'verification' in result:
        verif = result['verification']
        if 'differences' in verif:
            diffs = verif['differences']
            print(f"\nРазличия:")
            for key, value in diffs.items():
                print(f"  {key}: {value}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Конвертация .mat в .npy с проверкой целостности')
    parser.add_argument('--input', '-i', default='bm', help='Входная директория')
    parser.add_argument('--output', '-o', default='bm_npy', help='Выходная директория')
    parser.add_argument('--test', action='store_true', help='Протестировать на одном файле')
    parser.add_argument('--test-file', help='Файл для тестирования')

    args = parser.parse_args()

    if args.test:
        if args.test_file:
            test_file = Path(args.test_file)
        else:
            test_file = Path("test.mat")  # Или любой другой файл для теста

        # Создаем временный конвертер
        converter = MatToNpyConverterDebug()

        if test_file.exists():
            test_single_file()
        else:
            print(f"Файл {test_file} не найден. Создайте тестовый файл или укажите путь через --test-file")
    else:
        converter = MatToNpyConverterDebug(args.input, args.output)
        results = converter.batch_convert_with_verification()
