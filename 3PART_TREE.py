# ============================================
# ГЛАВА 3. АНАЛИЗ ДАННЫХ С ИЗОБРАЖЕНИЯМИ
# ============================================
print("\n" + "=" * 60 + "\nГЛАВА 3. АНАЛИЗ ДАННЫХ С ИЗОБРАЖЕНИЯМИ\n" + "=" * 60)

import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# ============================================
# 1. ЗАГРУЗКА И ПЕРВИЧНОЕ ЗНАКОМСТВО
# ============================================
print("\n" + "=" * 50 + "\n1. ЗАГРУЗКА И ПЕРВИЧНОЕ ЗНАКОМСТВО\n" + "=" * 50)

DATASET_PATH = "C:/Users/bobv6/Desktop/Kyrsovaia/3part_TOMAT_data"
SPLIT = "train"

COLORS = [
    (255, 0, 0),   # b_fully_ripened - синий
    (0, 255, 0),   # b_half_ripened - зеленый
    (0, 0, 255),   # b_green - красный
    (255, 255, 0), # l_fully_ripened - голубой
    (255, 0, 255), # l_half_ripened - пурпурный
    (0, 255, 255)  # l_green - желтый
]

# Пути к папкам
image_dir = Path(DATASET_PATH) / SPLIT / "images"
label_dir = Path(DATASET_PATH) / SPLIT / "labels"

# Находим все изображения
image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
print(f"Найдено изображений: {len(image_files)}\n\n" +
      "-"*50 + "\nВИЗУАЛИЗАЦИЯ ПЕРВЫХ 3 ФОТО\n" + "-"*50)

# Показываем первые 3
for img_file in image_files[:3]:
    print(f"\nОбработка: {img_file.name}")

    # Загрузка изображения
    img = cv2.imread(str(img_file))
    if img is None:
        print(f"  ОШИБКА: не удалось загрузить {img_file}")
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Загрузка аннотации
    label_file = label_dir / f"{img_file.stem}.txt"
    print(f"  Аннотация: {label_file}")

    if label_file.exists():
        with open(label_file, "r") as f:
            annotations = f.readlines()
        print(f"  Найдено объектов: {len(annotations)}")

        # Рисуем прямоугольники
        for ann in annotations:
            parts = ann.strip().split()
            if len(parts) == 5:
                class_id, x_center, y_center, width, height = map(float, parts)

                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)

                color = COLORS[int(class_id) % len(COLORS)]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 5)
                cv2.putText(img, str(int(class_id)), (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Показываем результат
        plt.figure(figsize=(10, 8))
        plt.imshow(img)
        plt.title(f"Файл: {img_file.name}")
        plt.axis("off")
        plt.show()
    else:
        print(f"  НЕТ ФАЙЛА АННОТАЦИИ")

# ============================================
# 2. Визуализация распределения объектов по классам (train и test)
# ============================================
print("\n" + "=" * 50 +
      "\n2. ВИЗУАЛИЗАЦИЯ РАСПРЕДЕЛЕНИЯ ОБЪЕКТОВ ПО КЛАССАМ (train и test)\n"
      + "=" * 50)

# Данные из таблицы
classes = ['b_fully_ripened', 'b_half_ripened', 'b_green',
           'l_fully_ripened', 'l_half_ripened', 'l_green']

# Сокращённые названия для подписей на графике
short_labels = ['b_fully', 'b_half', 'b_green', 'l_fully', 'l_half', 'l_green']

# Цвета для столбцов
colors = ['#FF6B6B', '#FFB347', '#66BB6A', '#FFD93D', '#6C5B7B', '#4ECDC4']

# --------------------------------------------
# Распределение объектов по классам в тестовой выборке (train)
# --------------------------------------------
counts = [348, 520, 1467, 982, 797, 3667]

# Построение диаграммы
plt.figure(figsize=(10, 6))
bars = plt.bar(short_labels, counts, color=colors, edgecolor='black', linewidth=1.2)

# Добавление значений на столбцы
for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
             str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')

# Настройка оформления
plt.title('Распределение объектов по классам в обучающей выборке (train)',
          fontsize=14, fontweight='bold')
plt.xlabel('Класс', fontsize=12)
plt.ylabel('Количество объектов', fontsize=12)
plt.xticks(rotation=45, ha='right')

# Добавление сетки
plt.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

# Вывод статистики
print("\n" + "-"*50 + "\nСТАТИСТИКА ПО КЛАССАМ (train):\n" + "-"*50)

for i, (cls, count) in enumerate(zip(classes, counts)):
    percent = (count / sum(counts)) * 100
    print(f"{cls:20} : {count:4} объектов ({percent:.1f}%)")
print(f"\nВсего объектов в train: {sum(counts)}")

# --------------------------------------------
# Распределение объектов по классам в тестовой выборке (test)
# --------------------------------------------
counts1 = [72, 116, 387, 269, 223, 929]

# Построение диаграммы
plt.figure(figsize=(10, 6))
bars = plt.bar(short_labels, counts1, color=colors, edgecolor='black', linewidth=1.2)

# Добавление значений
for bar, count in zip(bars, counts1):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
             str(count), ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.title('Распределение объектов по классам в тестовой выборке (test)',
          fontsize=14, fontweight='bold')
plt.xlabel('Класс', fontsize=12)
plt.ylabel('Количество объектов', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

# Вывод статистики
print("\n" + "-"*50 + "\nСТАТИСТИКА ПО КЛАССАМ (test):\n" + "-"*50 + "\n")

for i, (cls, count) in enumerate(zip(classes, counts1)):
    percent = (count / sum(counts1)) * 100
    print(f"{cls:20} : {count:4} объектов ({percent:.1f}%)")
print(f"\nВсего объектов в test: {sum(counts1)}\n" + "-"*50 + f"\nВсего объектов: {sum(counts1) + sum(counts)}")

# ============================================
# 3. Примеры визуализации разметки
# ============================================
print("\n" + "=" * 50 + "\n3. Примеры визуализации разметки\n" + "=" * 50)
# Папки с картинками и аннотациями
images_dir = Path(DATASET_PATH) / "train" / "images"
labels_dir = Path(DATASET_PATH) / "train" / "labels"

# Находим все картинки
images = list(images_dir.glob("*.jpg"))

# Показываем первые 3 картинки
plt.figure(figsize=(15, 10))

for i in range(3):
    # Загружаем картинку
    img = cv2.imread(str(images[i]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    # Загружаем аннотацию
    label_file = labels_dir / (images[i].stem + ".txt")
    if label_file.exists():
        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    # Читаем координаты
                    cls = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    w_box = float(parts[3])
                    h_box = float(parts[4])

                    # Переводим в пиксели
                    x1 = int((x - w_box/2) * w)
                    y1 = int((y - h_box/2) * h)
                    x2 = int((x + w_box/2) * w)
                    y2 = int((y + h_box/2) * h)

                    # Рисуем прямоугольник
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 15)

                    # Название класса (напишите нужные названия)
                    class_names = {
                        0: "b_fully_ripened",
                        1: "b_half_ripened",
                        2: "b_green",
                        3: "l_fully_ripened",
                        4: "l_half_ripened",
                        5: "l_green"
                    }

    # Показываем
    plt.subplot(1, 3, i+1)
    plt.imshow(img)
    plt.title(images[i].name)
    plt.axis("off")

plt.suptitle("Визуализация аннотаций (bounding box)", fontsize=14)
plt.show()

print("\nРАЗМЕТКА БЫЛА ОТОБРАЖЕНА")

# ============================================
# 4. Анализ размера и разрешения изображений из train
# ============================================
print("\n" + "=" * 50 +
      "\n4. Анализ размера и разрешения изображений из train\n"
      + "=" * 50)

# 1. Списки для сбора размеров
all_widths = []
all_heights = []
all_areas = []

# 2. Проходим по всем файлам разметки
for label_file in label_dir.glob("*.txt"):
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                w = float(parts[3])  # ширина
                h = float(parts[4])  # высота

                all_widths.append(w)
                all_heights.append(h)
                all_areas.append(w * h)

# 3. Выводим результаты
print(f"\nВсего объектов: {len(all_widths)}"
      f"\nШИРИНА: \nМинимум: {np.min(all_widths):.4f} \nМаксимум: {np.max(all_widths):.4f} "
      f"\nСреднее: {np.mean(all_widths):.4f} \nМедиана: {np.median(all_widths):.4f}\n"
      f"\nВЫСОТА: \nМинимум: {np.min(all_heights):.4f} \nМаксимум: {np.max(all_heights):.4f} "
      f"\nСреднее: {np.mean(all_heights):.4f} \nМедиана: {np.median(all_heights):.4f}\n"
      f"\nПЛОЩАДЬ: \nМинимум: {np.min(all_areas):.4f} \nМаксимум: {np.max(all_areas):.4f} "
      f"\nСреднее: {np.mean(all_areas):.4f} \nМедиана: {np.median(all_areas):.4f}")

# 5. Пересчёт в пиксели (для картинки 3024x4032)
print("\n" + "-" * 50 + "\nПЕРЕСЧЁТ В ПИКСЕЛИ (изображение 3024x4032):\n" + "-" * 50)

w_mean_px = np.mean(all_widths) * 3024
h_mean_px = np.mean(all_heights) * 4032
w_min_px = np.min(all_widths) * 3024
h_min_px = np.min(all_heights) * 4032
w_max_px = np.max(all_widths) * 3024
h_max_px = np.max(all_heights) * 4032

print(f"\nСредний размер: {w_mean_px:.0f} x {h_mean_px:.0f} пикселей"
      f"\nМинимальный: {w_min_px:.0f} x {h_min_px:.0f} пикселей"
      f"\nМаксимальный: {w_max_px:.0f} x {h_max_px:.0f} пикселей")

# ============================================
# 5. Анализ пространственного расположения объектов из train
# ============================================
print("\n" + "=" * 50 +
      "\n5. Анализ пространственного расположения объектов из train\n"
      + "=" * 50)

# Списки для координат центров
centers_x = []
centers_y = []

# Собираем координаты центров всех объектов
for label_file in label_dir.glob("*.txt"):
    with open(label_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                x_center = float(parts[1])  # X центра (норм. 0-1)
                y_center = float(parts[2])  # Y центра (норм. 0-1)
                centers_x.append(x_center)
                centers_y.append(y_center)

# Переводим в numpy
centers_x = np.array(centers_x)
centers_y = np.array(centers_y)

# ------------------------------------------
# СТАТИСТИКА ПО НАЛОЖЕНИЮ РАЗМЕТКИ
# ------------------------------------------
print(f"Всего объектов: {len(centers_x)}")

# Делим изображение на зоны: лево/центр/право и верх/центр/низ
def zone(x, y):
    #Определяет зону по X и Y
    hz = "лево" if x < 0.33 else ("центр" if x < 0.66 else "право")
    vz = "верх" if y < 0.33 else ("центр" if y < 0.66 else "низ")
    return f"{vz}-{hz}"

# Подсчёт по зонам
zones = {}
for x, y in zip(centers_x, centers_y):
    z = zone(x, y)
    zones[z] = zones.get(z, 0) + 1

# Вывод зон (отсортированы по убыванию)
print(f"\nРаспределение объектов по зонам изображения: \n{'Зона':<15} {'Количество':<12} {'Доля %':<10}\n" + "-" * 50)

# Определяем порядок зон для красивого вывода
zone_order = [
    "верх-лево", "верх-центр", "верх-право",
    "центр-лево", "центр-центр", "центр-право",
    "низ-лево", "низ-центр", "низ-право"
]

total = len(centers_x)
for z in zone_order:
    count = zones.get(z, 0)
    percent = (count / total) * 100
    bar = "|" * int(percent / 2)
    print(f"{z:<15} {count:<12} {percent:<12.1f} {bar}")

# Статистика по осям отдельно
print("\nРаспределение по горизонтали (X):")
left = sum(1 for x in centers_x if x < 0.33)
mid_x = sum(1 for x in centers_x if 0.33 <= x < 0.66)
right = sum(1 for x in centers_x if x >= 0.66)
print(f"Левая треть: {left} ({left/total*100:.1f}%) "
      f"\nЦентральная: {mid_x} ({mid_x/total*100:.1f}%) "
      f"\nПравая треть: {right} ({right/total*100:.1f}%)"
      f"\n\nРаспределение по вертикали (Y):")

top = sum(1 for y in centers_y if y < 0.33)
mid_y = sum(1 for y in centers_y if 0.33 <= y < 0.66)
bottom = sum(1 for y in centers_y if y >= 0.66)
print(f"Верхняя треть: {top} ({top/total*100:.1f}%)"
      f"\nЦентральная: {mid_y} ({mid_y/total*100:.1f}%) "
      f"\nНижняя треть: {bottom} ({bottom/total*100:.1f}%)")

# ============================================
# 6. ТЕПЛОВАЯ КАРТА (HEATMAP)
# ============================================
print("\n" + "=" * 50 +
      "\n6. ТЕПЛОВАЯ КАРТА (HEATMAP)\n"
      + "=" * 50)

plt.figure(figsize=(8, 10))

# Строим 2D гистограмму
heatmap, xedges, yedges = np.histogram2d(
    centers_y, centers_x,  # Y первый — строки изображения
    bins=30,
    range=[[0, 1], [0, 1]]
)

# Рисуем
plt.imshow(
    heatmap,
    origin='upper',
    extent=[0, 1, 1, 0],  # X слева направо, Y сверху вниз
    cmap='hot',
    aspect='equal'
)

plt.colorbar(label='Количество объектов в ячейке')

# Подписи
plt.xlabel('X (нормализованная координата)')
plt.ylabel('Y (нормализованная координата)')
plt.title('Тепловая карта расположения помидоров на изображениях\n',
          fontsize=12, fontweight='bold')

# Добавляем сетку для зон
for pos in [0.33, 0.66]:
    plt.axvline(x=pos, color='cyan', linestyle='--', alpha=0.5, linewidth=0.8)
    plt.axhline(y=pos, color='cyan', linestyle='--', alpha=0.5, linewidth=0.8)

# Подписи зон
zone_labels = [
    ("верх-лево", 0.165, 0.165),
    ("верх-центр", 0.50, 0.165),
    ("верх-право", 0.835, 0.165),
    ("центр-лево", 0.165, 0.50),
    ("центр-центр", 0.50, 0.50),
    ("центр-право", 0.835, 0.50),
    ("низ-лево", 0.165, 0.835),
    ("низ-центр", 0.50, 0.835),
    ("низ-право", 0.835, 0.835),
]

for text, x, y in zone_labels:
    plt.text(x, y, text, ha='center', va='center', fontsize=7,
             color='cyan', bbox=dict(boxstyle='round,pad=0.2', facecolor='black', alpha=0.5))

plt.tight_layout()
plt.show()

print("\nТЕПЛОВАЯ КАРТА БЫЛА ОТОБРАЖЕНА")

# ============================================
# ВЫВОДЫ ПО 3 ГЛАВЕ
# ============================================
print("\n" + "=" * 50 +
      "\nВЫВОДЫ ПО 3 ГЛАВЕ\n"
      + "=" * 50)

# Находим самую заполненную зону
max_zone = max(zones, key=zones.get)
print(f"""
1. Объём данных:
   - 804 изображения (643 train / 161 test)
   - 9 777 размеченных объектов (7 781 train / 1 996 test)
   - 6 классов (размер + стадия созревания)

2. Распределение классов:
   - Сильный дисбаланс: l_green — 47.1%, b_fully_ripened — 4.5%
   - Разница более чем в 10 раз. Требуется аугментация редких классов

3. Размеры объектов (в пикселях):
   - Средний: {np.mean(all_widths)*3024:.0f} x {np.mean(all_heights)*4032:.0f}
   - Минимальный: {np.min(all_widths)*3024:.0f} x {np.min(all_heights)*4032:.0f}
   - Максимальный: {np.max(all_widths)*3024:.0f} x {np.max(all_heights)*4032:.0f}
   - Рекомендуемое входное разрешение: не менее 1024x1024

4. Пространственное расположение:
   - Центр-центр: {zones.get('центр-центр', 0)} объектов ({zones.get('центр-центр', 0)/total*100:.1f}%)
   - По горизонтали: центр {sum(1 for x in centers_x if 0.33 <= x < 0.66)/total*100:.0f}%
   - По вертикали: центр {sum(1 for y in centers_y if 0.33 <= y < 0.66)/total*100:.0f}%
   - Объекты смещены в центр и вниз кадра

5. Качество разметки:
   - Все видимые помидоры обведены
   - Классы соответствуют визуальному состоянию
   - Пропущенных объектов не обнаружено

6. Пригодность: данные полностью пригодны для детекции помидоров.
""")

print("\n" + "=" * 50 + "\nАНАЛИЗ ЗАВЕРШЁН\n" + "=" * 50)

# ============================================
#--------------------END----------------------
# ============================================