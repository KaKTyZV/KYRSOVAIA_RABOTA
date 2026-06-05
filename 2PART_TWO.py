# ============================================
# ГЛАВА 2. АНАЛИЗ ДАННЫХ С ВРЕМЕННЫМИ РЯДАМИ
# ============================================
print("\n" + "=" * 60 + "\nГЛАВА 2. АНАЛИЗ ДАННЫХ С ВРЕМЕННЫМИ РЯДАМИ\n" + "=" * 60)

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# 1. Загрузка и первичное знакомство (SKAB)
# ============================================

# Путь к папке с данными
path = r"C:\Users\bobv6\Desktop\Kyrsovaia\2part_SKAB_data"

# Находим все CSV файлы
all_files = glob.glob(f"{path}/**/*.csv", recursive=True)

print("\n"+"="*50 +
      "\n1. ЗАГРУЗКА ДАННЫХ И ПЕРВИЧНОЕ ЗНАКОМСТВО\n"+
      "="*50 + f"\n\nНайдено файлов: {len(all_files)}")

# Загружаем все файлы с правильным разделителем (;)
df_list = []
for f in all_files:
    name = f.split("\\")[-1].replace(".csv", "")
    temp = pd.read_csv(f, sep=";")
    temp["эксперимент"] = name
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)

print(f"Всего строк: {len(df)} \nВсего столбцов: {df.shape[1]} "
      f"\nСтолбцы: {df.columns.tolist()}")

# Проверяем, есть ли столбец datetime
if 'datetime' in df.columns:
    df["datetime"] = pd.to_datetime(df["datetime"])
    df = df.set_index("datetime")
    print(f"Первые 5 строк: \n{df.head()}\n")
else:
    print(f"Первые 5 строк (без индекса времени): \n{df.head()}"
          f"\nСтолбцы: {df.columns.tolist()}")

# Каналы
data_cols = ['Accelerometer1RMS', 'Accelerometer2RMS', 'Current',
             'Pressure', 'Temperature', 'Thermocouple', 'Voltage',
             'Volume Flow RateRMS']

print(f"""  
--------------------------------------------------------------------------------------
ИТОГИ
--------------------------------------------------------------------------------------
- Данные загружены без ошибок из 35 CSV-файлов

- Структура соответствует описанию: 46 860 строк, 8 измерительных каналов

- Ряд многомерный (8 каналов)

- Временная метка присутствует в формате YYYY-MM-DD hh:mm:ss, преобразована в datetime

- Все каналы имеют числовой тип float64
""")

# ============================================
# 2. Визуализация исходных данных (SKAB)
# ============================================
# Берём эксперимент 1
df_exp = df[df['эксперимент'] == '1'].copy()

print("=" * 50+
      "\n2. ВИЗУАЛИЗАЦИЯ ДАННЫХ\n"
      + "=" * 50 + f"\n\nЭксперимент: 1 "
      f"\nДлина ряда: {len(df_exp)} точек "
      f"\nАномалий: {int(df_exp['anomaly'].sum())} "
      f"\nМетки смены режима: {int(df_exp['changepoint'].sum())} "
      f"\nПериод: {df_exp.index.min()} - {df_exp.index.max()}")

# График 1: все каналы
fig, axes = plt.subplots(len(data_cols), 1, figsize=(16, 12), sharex=True)

for i, col in enumerate(data_cols):
    ax = axes[i]
    ax.plot(df_exp.index, df_exp[col], linewidth=1.2, color='steelblue')

    anomaly_mask = df_exp['anomaly'] == 1
    if anomaly_mask.any():
        ax.scatter(df_exp.index[anomaly_mask],
                   df_exp.loc[anomaly_mask, col],
                   color='red', s=15, alpha=0.8)

    ax.set_ylabel(col, fontsize=9)
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Дата и время', fontsize=10)
plt.xticks(rotation=45, fontsize=8)
fig.suptitle('Визуализация всех каналов - эксперимент 1',
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# График 2: Temperature с аномалиями
fig, ax = plt.subplots(figsize=(16, 6))
ax.plot(df_exp.index, df_exp['Temperature'], linewidth=1.5, color='steelblue')

anomaly_mask = df_exp['anomaly'] == 1
ax.scatter(df_exp.index[anomaly_mask],
           df_exp.loc[anomaly_mask, 'Temperature'],
           color='red', s=20, alpha=0.8, label=f'Аномалии ({int(df_exp["anomaly"].sum())} точек)')

ax.set_title('Канал Temperature с аномалиями - эксперимент 1',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Дата и время', fontsize=11)
ax.set_ylabel('Temperature (°C)', fontsize=11)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11)
plt.xticks(rotation=45, fontsize=9)
plt.tight_layout()
plt.show()

print(f"""  
--------------------------------------------------------------------------------------
ИТОГИ
--------------------------------------------------------------------------------------
- Размеченные аномалии визуально совпадают с резкими скачками

- Присутствуют точечные аномалии и коллективные(при смене режима)

- Разрывов и пропусков на графиках не обнаружено

- Каналы взаимосвязаны (ток, напряжение, давление меняются синхронно)
- Видны повторяющиеся циклы работы насоса
""")
# ============================================
# 3. Статистический анализ (SKAB)
# ============================================
print("=" * 50+
      "\n3. СТАТИСТИЧЕСКИЙ АНАЛИЗ\n"+
      "=" * 50)

# 1. ПОЛНАЯ ТАБЛИЦА
print("\n1. Описательные статистики измерительных каналов\n"+
      f"{'Канал':<25} {'count':<8} {'mean':<10} {'std':<10} "
      f"{'min':<10} {'Q1 (25%)':<10} {'Q2 (50%)':<10} {'Q3 (75%)':<10} {'max':<10}\n"+
      "-" * 50)

for col in data_cols:
    stats = df[col].describe()
    print(f"{col:<25} {int(stats['count']):<8} {stats['mean']:<10.4f} {stats['std']:<10.4f} "
          f"{stats['min']:<10.4f} {stats['25%']:<10.4f} {stats['50%']:<10.4f} "
          f"{stats['75%']:<10.4f} {stats['max']:<10.4f}")

# 2. ЧАСТОТА ДИСКРЕТИЗАЦИИ
print("\n" + "-" * 50 + "\n2. Частота дискретизации:")

df_one = df[df['эксперимент'] == '1'].copy()
time_diffs = df_one.index.to_series().diff().dropna()
most_common = time_diffs.mode()[0]
freq = 1 / most_common.total_seconds()

print(f"\nЭксперимент 1: {len(df_one)} точек"
      f"\nИнтервал: {most_common}"
      f"\nЧастота: {freq:.1f} Гц"
      f"\nУникальных интервалов: {len(time_diffs.unique())}")

print(f"""
--------------------------------------------------------------------------------------
ИТОГИ
--------------------------------------------------------------------------------------
- Все значения в пределах физической нормы, кроме целевых аномалий (Pressure < 0 — кавитация)

- Сильная асимметрия: Accelerometer1RMS, Accelerometer2RMS, Current, Pressure
- Причина: насос большую часть времени выключен (значения ~0)
- Распределение Voltage близко к симметричному

- Частота дискретизации: 1 Гц, интервалы с небольшими отклонениями

- Коэффициент вариации: от 6% (Voltage) до 345% (Current).

- Каналов с нулевым стандартным отклонением нет
""")
# ============================================
# 4. Анализ пропусков и выбросов (SKAB)
# ============================================
# Заменяем NaN в anomaly и changepoint на 0 (нет аномалии)
df['anomaly'] = df['anomaly'].fillna(0).astype(int)
df['changepoint'] = df['changepoint'].fillna(0).astype(int)

print("=" * 50+
      "\n4. АНАЛИЗ ПРОПУСКОВ И ВЫБРОСОВ\n"+
      "=" * 50)

# 1. ПРОПУСКИ (только измерительные каналы)
print("\n1. ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ (измерительные каналы):\n"+
      f"{'Канал':<25} {'Пропусков':<12} {'Доля %':<10}\n"+
      "-" * 50)

for col in data_cols:
    missing = df[col].isnull().sum()
    percent = missing / len(df) * 100
    print(f"{col:<25} {missing:<12} {percent:<10.3f}")

total_missing = df[data_cols].isnull().sum().sum()
print(
    f"\nВсего пропусков в измерительных каналах: {total_missing} из "
    f"{len(df) * len(data_cols)} ({total_missing / (len(df) * len(data_cols)) * 100:.3f}%)\n"+
    "\n" + "-" * 50 +"\n2. ВЫБРОСЫ (правило трёх сигм):\n"+
    f"{'Канал':<25} {'Среднее':<10} {'±3σ границы':<35} {'Выбросов':<10} {'Доля %':<8}\n"+
    "-" * 50)

for col in data_cols:
    mean_v = df[col].mean()
    std_v = df[col].std()
    lower = mean_v - 3 * std_v
    upper = mean_v + 3 * std_v

    outliers = df[(df[col] < lower) | (df[col] > upper)]
    count = len(outliers)
    percent = count / len(df) * 100

    bounds = f"[{lower:.3f}, {upper:.3f}]"
    print(f"{col:<25} {mean_v:<10.3f} {bounds:<35} {count:<10} {percent:<8.2f}")

# 3. BOXPLOT
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, col in enumerate(data_cols):
    ax = axes[i]
    bp = ax.boxplot(df[col].dropna(), vert=True, patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.6),
                    flierprops=dict(marker='o', markerfacecolor='red',
                                    markersize=3, alpha=0.5))
    ax.set_title(col, fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle('Boxplot-диаграммы каналов (выбросы — красные точки)',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

print(f"""
--------------------------------------------------------------------------------------
ИТОГИ
--------------------------------------------------------------------------------------
- Пропуски в измерительных каналах: 0 из 374 880 (данные полные)
- Выбросы: Accelerometer1RMS — 1253 (2.67%), Accelerometer2RMS — 1227 (2.62%)
Current, Pressure, Voltage: < 1% — редкие выбросы
Temperature, Thermocouple, Volume Flow Rate: 0% — стабильные показатели

- Выбросы целевые аномалии (переходные режимы насоса), не удаляем
- Для ML-моделей потребуется масштабирование (StandardScaler)
""")

# ============================================
# 5. Анализ диапазонов значений (SKAB)
# ============================================
print("=" * 50 + "\n5. АНАЛИЗ ДИАПАЗОНОВ ЗНАЧЕНИЙ\n" + "=" * 50 +
      "\n\nДИАПАЗОНЫ ЗНАЧЕНИЙ ПО КАНАЛАМ:\n" +
      f"{'Канал':<25} {'Мин':<12} {'Макс':<12} {'Размах':<12} {'Ед.изм.'}\n" + "-" * 73)

units = {
    'Accelerometer1RMS': 'g',
    'Accelerometer2RMS': 'g',
    'Current': 'А',
    'Pressure': 'бар',
    'Temperature': '°C',
    'Thermocouple': '°C',
    'Voltage': 'В',
    'Volume Flow RateRMS': 'л/мин'
}

ranges = {}

for col in data_cols:
    min_v = df[col].min()
    max_v = df[col].max()
    range_v = max_v - min_v
    ranges[col] = range_v
    unit = units.get(col, '')
    print(f"{col:<25} {min_v:<12.4f} {max_v:<12.4f} {range_v:<12.4f} {unit}")

# 2. BOXPLOT ВСЕХ КАНАЛОВ НА ОДНОМ ГРАФИКЕ
fig, ax = plt.subplots(figsize=(14, 6))

# Нормализуем для сравнения (Min-Max scaling к [0,1])
df_norm = df[data_cols].copy()
for col in data_cols:
    min_v = df[col].min()
    max_v = df[col].max()
    if max_v > min_v:
        df_norm[col] = (df[col] - min_v) / (max_v - min_v)

bp = ax.boxplot([df_norm[col].dropna() for col in data_cols],
                vert=True, patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.6),
                flierprops=dict(marker='o', markerfacecolor='red', markersize=3, alpha=0.5))

ax.set_xticklabels(data_cols, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Нормализованные значения [0, 1]', fontsize=10)
ax.set_title('Сравнение диапазонов каналов (нормализованные)',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"""
--------------------------------------------------------------------------------------
ИТОГИ
--------------------------------------------------------------------------------------
- Размах от 0.722 (Accelerometer1RMS) до 254.74 (Voltage)
- Отношение: 352:1 — масштабы критически различаются
- Каналы измеряются в разных единицах (g, А, бар, °C, В, л/мин)

- Требуется стандартизация, так как:
    * Алгоритмы поиска аномалий чувствительны к масштабу
    * Разница в 352 раза приведёт к доминированию Voltage над другими каналами
- Наиболее подходит StandardScaler (приведение к μ=0, σ=1), так как:
    * Данные имеют выбросы, но это целевые аномалии (сохраняем их)
    * MinMaxScaler сожмёт выбросы и сделает их незаметными
""")

# ============================================
# 6. Корреляционный анализ (SKAB)
# ============================================
print("=" * 50 + "\n6. КОРРЕЛЯЦИОННЫЙ АНАЛИЗ\n" + "=" * 50)

# 1. МАТРИЦА КОРРЕЛЯЦИЙ
corr_matrix = df[data_cols].corr()

print("\n" + "-" * 50 + f"\n1. МАТРИЦА КОРРЕЛЯЦИЙ (Пирсон):\n{corr_matrix.round(3)}")

# 2. ТЕПЛОВАЯ КАРТА
fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            center=0, vmin=-1, vmax=1, square=True,
            linewidths=0.5, cbar_kws={'label': 'Коэффициент корреляции'},
            ax=ax)

ax.set_title('Тепловая карта корреляций измерительных каналов',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# 3. ПОИСК СИЛЬНЫХ СВЯЗЕЙ
print("\n" + "-" * 50 + "\n2. СИЛЬНЫЕ КОРРЕЛЯЦИИ (|r| > 0.7):\n")

strong_pairs = []

for i in range(len(data_cols)):
    for j in range(i+1, len(data_cols)):
        r = corr_matrix.iloc[i, j]
        if abs(r) > 0.7:
            strong_pairs.append((data_cols[i], data_cols[j], r))
            print(f"   {data_cols[i]} - {data_cols[j]}: r = {r:.3f}")

if not strong_pairs:
    print("   Сильных корреляций не обнаружено.")

# 4. ПОИСК НЕЗАВИСИМЫХ КАНАЛОВ
print("\n" + "-" * 50 + "\n3. НЕЗАВИСИМЫЕ КАНАЛЫ (|r| < 0.3 со всеми):\n")

for col in data_cols:
    others = [c for c in data_cols if c != col]
    max_corr = max(abs(corr_matrix[col][others]))
    if max_corr < 0.3:
        print(f"   {col}: макс |r| = {max_corr:.3f} (слабо связан с остальными)")

print(f"""
--------------------------------------------------------------------------------------
ИТОГИ
--------------------------------------------------------------------------------------
- Сильные положительные корреляции (r > 0.7): 9 пар
- Accelerometer1RMS и Accelerometer2RMS дублируют друг друга (r = 0.996)

- Сильных отрицательных корреляций не обнаружено

- Канал Pressure слабо связан с остальными (|r| < 0.11) — уникальная информация
- Сильно коррелирующие каналы можно сократить
""")

# ============================================
# 7. Поиск и анализ шумов (SKAB)
# ============================================
from statsmodels.tsa.seasonal import seasonal_decompose

# Берём эксперимент 1 и канал Temperature
df_exp = df[df['эксперимент'] == '1'].copy()
key_col = 'Temperature'

print("=" * 50 +
      "\n7. ПОИСК И АНАЛИЗ ШУМОВ\n"
      + "=" * 50 +
      f"\n\nЭксперимент: 1, канал: {key_col}"
      f"\nДлина ряда: {len(df_exp)} точек")

# 1. ДЕКОМПОЗИЦИЯ РЯДА
# Период: 60 секунд (цикл работы насоса примерно 1 минута)
period = 60

result = seasonal_decompose(df_exp[key_col].dropna(),
                           model='additive',
                           period=period)

# 2. ВИЗУАЛИЗАЦИЯ ДЕКОМПОЗИЦИИ
fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

axes[0].plot(df_exp.index, result.observed, linewidth=1, color='steelblue')
axes[0].set_ylabel('Исходный ряд', fontsize=9)
axes[0].grid(True, alpha=0.3)

axes[1].plot(df_exp.index, result.trend, linewidth=1, color='orange')
axes[1].set_ylabel('Тренд', fontsize=9)
axes[1].grid(True, alpha=0.3)

axes[2].plot(df_exp.index, result.seasonal, linewidth=1, color='green')
axes[2].set_ylabel('Сезонность', fontsize=9)
axes[2].grid(True, alpha=0.3)

axes[3].plot(df_exp.index, result.resid, linewidth=0.8, color='red', alpha=0.7)
axes[3].set_ylabel('Остатки (шум)', fontsize=9)
axes[3].set_xlabel('Время')
axes[3].grid(True, alpha=0.3)

plt.xticks(rotation=45, fontsize=8)
fig.suptitle(f'Декомпозиция канала {key_col} — эксперимент 1',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# 3. РАСЧЁТ SNR
# Убираем NaN из остатков
resid = result.resid.dropna()
trend = result.trend.dropna()
seasonal = result.seasonal.dropna()

# Выравниваем длину
min_len = min(len(trend), len(seasonal))
trend = trend.iloc[:min_len]
seasonal = seasonal.iloc[:min_len]

signal = trend + seasonal
noise = resid.iloc[:min_len]

var_signal = np.var(signal)
var_noise = np.var(noise)

if var_noise > 0:
    snr = 10 * np.log10(var_signal / var_noise)
else:
    snr = float('inf')

print(f"\nSNR (отношение сигнал/шум):"
      f"\nДисперсия сигнала: {var_signal:.4f}"
      f"\nДисперсия шума:    {var_noise:.4f} \nSNR: {snr:.2f} дБ")

# Интерпретация SNR
if snr > 20:
    snr_quality = "Отлично"
elif snr > 10:
    snr_quality = "Хорошо"
elif snr > 0:
    snr_quality = "Удовлетворительно"
else:
    snr_quality = "Плохо"

print(f"Качество: {snr_quality}")

# 5. ГИСТОГРАММА ОСТАТКОВ
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(noise.dropna(), bins=50, color='steelblue', edgecolor='white', alpha=0.7)
ax.axvline(x=0, color='red', linestyle='--', linewidth=1.5, label='Ноль')
ax.set_xlabel('Значение остатка', fontsize=10)
ax.set_ylabel('Частота', fontsize=10)
ax.set_title('Гистограмма распределения остатков (шума)',
             fontsize=12, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

print(f"""
--------------------------------------------------------------------------------------
ИТОГИ
--------------------------------------------------------------------------------------
- Тренд: падающий (-8.17°C за 6 часов) — Интенсивность: сильная (отражает остывание двигателя)

- Сезонность: период ~60 сек, амплитуда низкая (0.20°C)

- SNR = 21.11 дБ — отлично (шум практически незаметен)

- Распределение остатков близко к нормальному
- Шум случаен, декомпозиция адекватна

- Фильтрация данных не требуется
""")

# ============================================
#--------------------END----------------------
# ============================================