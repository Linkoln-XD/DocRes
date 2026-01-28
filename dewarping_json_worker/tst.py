#!/usr/bin/env python3
"""
GUI версия для создания train.json (требует tkinter)
"""

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
except ImportError:
    print("Для GUI версии требуется tkinter")
    print("Установите: sudo apt-get install python3-tk")
    exit(1)

import json
from pathlib import Path


class TrainJsonCreator:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Создание train.json")
        self.root.geometry("600x500")

        # Переменные
        self.data_dir = tk.StringVar()
        self.output_file = tk.StringVar()
        self.use_relative = tk.BooleanVar(value=True)
        self.verbose = tk.BooleanVar(value=True)

        self.setup_ui()

    def setup_ui(self):
        # Заголовок
        title = tk.Label(self.root, text="Создание train.json",
                         font=("Arial", 16, "bold"))
        title.pack(pady=10)

        # Фрейм для директории с данными
        frame_data = tk.LabelFrame(self.root, text="Директория с данными", padx=10, pady=10)
        frame_data.pack(fill="x", padx=20, pady=5)

        tk.Label(frame_data, text="Папка, содержащая img/, uv/, bm_npy/:").pack(anchor="w")

        entry_data = tk.Entry(frame_data, textvariable=self.data_dir, width=50)
        entry_data.pack(side="left", fill="x", expand=True, padx=(0, 10))

        btn_browse_data = tk.Button(frame_data, text="Обзор...", command=self.browse_data_dir)
        btn_browse_data.pack(side="right")

        # Фрейм для выходного файла
        frame_output = tk.LabelFrame(self.root, text="Выходной файл", padx=10, pady=10)
        frame_output.pack(fill="x", padx=20, pady=5)

        tk.Label(frame_output, text="Куда сохранить train.json:").pack(anchor="w")

        entry_output = tk.Entry(frame_output, textvariable=self.output_file, width=50)
        entry_output.pack(side="left", fill="x", expand=True, padx=(0, 10))

        btn_browse_output = tk.Button(frame_output, text="Обзор...", command=self.browse_output_file)
        btn_browse_output.pack(side="right")

        # Настройки
        frame_settings = tk.LabelFrame(self.root, text="Настройки", padx=10, pady=10)
        frame_settings.pack(fill="x", padx=20, pady=5)

        tk.Checkbutton(frame_settings, text="Использовать относительные пути",
                       variable=self.use_relative).pack(anchor="w")
        tk.Checkbutton(frame_settings, text="Подробный вывод",
                       variable=self.verbose).pack(anchor="w")

        # Кнопка создания
        btn_create = tk.Button(self.root, text="Создать train.json",
                               command=self.create_json, bg="#4CAF50", fg="white",
                               font=("Arial", 12, "bold"))
        btn_create.pack(pady=20)

        # Консоль вывода
        frame_console = tk.LabelFrame(self.root, text="Вывод", padx=10, pady=10)
        frame_console.pack(fill="both", expand=True, padx=20, pady=5)

        self.console = tk.Text(frame_console, height=10, width=60)
        scrollbar = tk.Scrollbar(frame_console, command=self.console.yview)
        self.console.configure(yscrollcommand=scrollbar.set)

        self.console.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def browse_data_dir(self):
        directory = filedialog.askdirectory(title="Выберите директорию с данными")
        if directory:
            self.data_dir.set(directory)

            # Автоматически предлагаем имя для выходного файла
            if not self.output_file.get():
                output_path = Path(directory) / "train.json"
                self.output_file.set(str(output_path))

    def browse_output_file(self):
        filename = filedialog.asksaveasfilename(
            title="Сохранить train.json как",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.output_file.set(filename)

    def log(self, message):
        """Добавляет сообщение в консоль"""
        self.console.insert(tk.END, message + "\n")
        self.console.see(tk.END)
        self.root.update()

    def create_json(self):
        # Очищаем консоль
        self.console.delete(1.0, tk.END)

        data_dir = Path(self.data_dir.get())
        output_file = Path(self.output_file.get())

        if not data_dir.exists():
            messagebox.showerror("Ошибка", f"Директория не найдена:\n{data_dir}")
            return

        # Проверяем наличие папок
        required_folders = ["img", "uv", "bm_npy"]
        missing = []

        for folder in required_folders:
            if not (data_dir / folder).exists():
                missing.append(folder)

        if missing:
            messagebox.showerror(
                "Ошибка",
                f"В директории {data_dir} отсутствуют папки:\n" +
                "\n".join([f"• {f}/" for f in missing])
            )
            return

        self.log(f"📁 Директория с данными: {data_dir}")
        self.log(f"💾 Выходной файл: {output_file}")
        self.log("-" * 50)

        try:
            # Находим файлы
            img_dir = data_dir / "img"
            uv_dir = data_dir / "uv"
            bm_dir = data_dir / "bm_npy"

            self.log("🔍 Поиск файлов...")

            img_files = {f.stem: f for f in img_dir.rglob('*') if f.is_file()}
            uv_files = {f.stem: f for f in uv_dir.rglob('*') if f.is_file()}
            bm_files = {f.stem: f for f in bm_dir.rglob('*') if f.is_file()}

            self.log(f"📊 Найдено:")
            self.log(f"  img/: {len(img_files)} файлов")
            self.log(f"  uv/: {len(uv_files)} файлов")
            self.log(f"  bm_npy/: {len(bm_files)} файлов")

            # Находим общие имена
            common_names = set(img_files.keys()) & set(uv_files.keys()) & set(bm_files.keys())

            if not common_names:
                self.log("❌ Нет общих имен файлов!")
                messagebox.showwarning("Внимание", "Нет общих имен файлов между папками!")
                return

            self.log(f"✅ Найдено {len(common_names)} пар соответствующих файлов")

            # Создаем JSON данные
            data = []
            for name in sorted(common_names):
                if self.use_relative.get():
                    entry = {
                        "in_path": str(img_files[name].relative_to(data_dir)),
                        "mask_path": str(uv_files[name].relative_to(data_dir)),
                        "gt_path": str(bm_files[name].relative_to(data_dir))
                    }
                else:
                    entry = {
                        "in_path": str(img_files[name].resolve()),
                        "mask_path": str(uv_files[name].resolve()),
                        "gt_path": str(bm_files[name].resolve())
                    }
                data.append(entry)

            # Создаем директорию для сохранения
            output_file.parent.mkdir(parents=True, exist_ok=True)

            # Сохраняем JSON
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            self.log(f"\n🎉 Файл успешно создан!")
            self.log(f"📝 Всего записей: {len(data)}")

            if self.verbose.get():
                self.log(f"\n📋 Пример первой записи:")
                self.log(json.dumps(data[0], indent=2, ensure_ascii=False))

            messagebox.showinfo("Успех", f"Файл создан:\n{output_file}\n\nВсего записей: {len(data)}")

        except Exception as e:
            self.log(f"❌ Ошибка: {str(e)}")
            messagebox.showerror("Ошибка", f"Произошла ошибка:\n{str(e)}")

    def run(self):
        self.root.mainloop()


if __name__ == "__main__":
    app = TrainJsonCreator()
    app.run()