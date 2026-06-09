# ============================================
# ГЛАВА 1. АНАЛИЗ ТАБЛИЧНЫХ ДАННЫХ
# ============================================
print("\n" + "=" * 60 + "\nГЛАВА 1. АНАЛИЗ ДАННЫХ С ТАБЛИЧНЫМИ ДАННЫМИ\n" + "=" * 60)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Загрузка датасета
df = pd.read_csv("SLEEP/sleep_mobile_stress_dataset_15000.csv")

# ============================================
# 1. Загрузка и первичное знакомство
# ============================================
print("\n" + "=" * 50 + "\n1. ЗАГРУЗКА И ПЕРВИЧНОЕ ЗНАКОМСТВО\n" + "=" * 50 +
    f"\nРазмер датасета: {df.shape[0]} строк, {df.shape[1]} столбцов\n"
    + "\n" + "-" * 50 + "\nПЕРЕЧЕНЬ ВСЕХ ПРИЗНАКОВ\n" + "-" * 50)

# Сбор информации в таблицу
features_info = []

for col in df.columns:
    dtype = str(df[col].dtype)
    # Семантика (описание признака)
    semantics = {
        'user_id': 'Уникальный идентификатор пользователя',
        'age': 'Возраст в годах',
        'gender': 'Пол',
        'occupation': 'Род занятий / профессия',
        'daily_screen_time_hours': 'Ежедневное время перед экраном (часы)',
        'phone_usage_before_sleep_minutes': 'Использование телефона перед сном (минуты)',
        'sleep_duration_hours': 'Длительность сна (часы)',
        'sleep_quality_score': 'Оценка качества сна (шкала 1-10)',
        'stress_level': 'Уровень стресса (шкала 1-10)',
        'caffeine_intake_cups': 'Потребление кофеина (чашек в день)',
        'physical_activity_minutes': 'Физическая активность (минуты в день)',
        'notifications_received_per_day': 'Количество полученных уведомлений за день',
        'mental_fatigue_score': 'Уровень умственной усталости (шкала 1-10)'
    }

    # Диапазон значений
    if pd.api.types.is_numeric_dtype(df[col]):
        range_val = f"[{df[col].min():.2f}, {df[col].max():.2f}]"
        if df[col].dtype == 'int64':
            range_val = f"[{int(df[col].min())}, {int(df[col].max())}]"
    else:
        unique_vals = df[col].unique()
        if len(unique_vals) <= 5:
            range_val = str(list(unique_vals))
        else:
            range_val = f"{len(unique_vals)} уникальных значений"

    # Пример значения
    example_val = df[col].iloc[0]
    if pd.api.types.is_numeric_dtype(df[col]):
        example_val = round(example_val, 2) if isinstance(example_val, float) else example_val

    features_info.append({
        '№': len(features_info) + 1,
        'Признак': col,
        'Тип данных': dtype,
        'Семантика': semantics.get(col, 'Нет описания'),
        'Диапазон значений': range_val,
        'Пример': example_val
    })

features_df = pd.DataFrame(features_info)
print(features_df.to_string(index=False))

# Определение целевых признаков
print("""\n* Целевые признаки:
- stress_level (уровень стресса)
- sleep_quality_score (качество сна)
- mental_fatigue_score (умственная усталость)""")

# ============================================
# 2. Диаграммы распределения каждого признака. Библиотека: Matplotlib
# ============================================
print("\n"+"="*50 +
      "\n2. Диаграммы распределения каждого признака. Библиотека: Matplotlib\n"+
      "="*50)

# Целевые признаки
target_cols = ['stress_level', 'sleep_quality_score', 'mental_fatigue_score']

# Числовые признаки
numeric_cols = ['age', 'daily_screen_time_hours', 'phone_usage_before_sleep_minutes',
                'sleep_duration_hours', 'sleep_quality_score', 'stress_level',
                'caffeine_intake_cups', 'physical_activity_minutes',
                'notifications_received_per_day', 'mental_fatigue_score']

# Категориальные признаки
cat_cols = ['gender', 'occupation']

# ========== 1. ГИСТОГРАММЫ ДЛЯ ЧИСЛОВЫХ ПРИЗНАКОВ ==========
fig, axes = plt.subplots(5, 2, figsize=(12, 14))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    color = 'salmon' if col in target_cols else 'steelblue'

    axes[i].hist(df[col], bins=25, color=color, edgecolor='black', alpha=0.7)
    axes[i].axvline(df[col].mean(), color='red', linestyle='--', label=f'среднее={df[col].mean():.1f}')
    axes[i].set_title(f'{col} {"(целевой)" if col in target_cols else ""}', fontsize=11)
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Распределение числовых признаков', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ========== 2. СТОЛБЧАТЫЕ ДИАГРАММЫ ДЛЯ КАТЕГОРИАЛЬНЫХ ==========
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, col in enumerate(cat_cols):
    counts = df[col].value_counts()
    axes[i].bar(counts.index, counts.values, color='lightgreen', edgecolor='black')
    axes[i].set_title(f'{col}')
    axes[i].tick_params(axis='x', rotation=45)
    axes[i].grid(True, alpha=0.3, axis='y')
    # Добавляем цифры на столбцы
    for j, v in enumerate(counts.values):
        axes[i].text(j, v + 50, str(v), ha='center')

plt.suptitle('Распределение категориальных признаков', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nДиаграммы отображены")
# ============================================
# 3. Визуализация признаков средствами seaborn. Все графики: scatter + regline
# ============================================
print("\n"+"="*50 +
      "\n3. Визуализация признаков средствами seaborn. Все графики: scatter + regline\n"+
      "="*50)

sns.set_style("whitegrid")

# ===== Пара 1: Экранное время и Качество сна =====
plt.figure(figsize=(5, 4))
sns.regplot(data=df, x='daily_screen_time_hours', y='sleep_quality_score',
            scatter_kws={'alpha':0.2}, line_kws={'color':'red'})
plt.title('Экранное время - Качество сна')
plt.tight_layout()
plt.show()

print("\nПара 1: Отрицательная корреляция: чем больше экранного времени тем ниже качество сна")

# ===== Пара 2: Стресс м Умственная усталость =====
plt.figure(figsize=(5, 4))
sns.regplot(data=df, x='stress_level', y='mental_fatigue_score',
            scatter_kws={'alpha':0.2}, line_kws={'color':'red'})
plt.title('Стресс - Умственная усталость')
plt.tight_layout()
plt.show()

print("Пара 2: Положительная корреляция: стресс и усталость растут вместе")

# ===== Пара 3: Уровень стресса в зависимости от пола =====
plt.figure(figsize=(5, 4))
sns.regplot(data=df, x='physical_activity_minutes', y='sleep_quality_score',
            scatter_kws={'alpha':0.5}, line_kws={'color':'red', 'linewidth':2})
plt.title('Активность - Качество сна (рост)')
plt.xlabel('Физическая активность (минуты)')
plt.ylabel('Качество сна')
plt.tight_layout()
plt.show()
print("Пара 3: Нейтральная корреляция: физическая активность и качество сна не влияют друг на друга\n")

# ============================================
# 4. Визуализация признаков средствами plotly. (интерактивные графики)
# ============================================
print("="*50 +
      "\n4. Визуализация признаков средствами plotly. (интерактивные графики)\n"+
      "="*50)

# График 1
fig1 = px.scatter(df, x='daily_screen_time_hours', y='sleep_quality_score',
                  color='gender', title='Экранное время - Качество сна')
fig1.show()

# График 2
fig2 = px.scatter(df, x='stress_level', y='mental_fatigue_score',
                  color='age', title='Стресс - Умственная усталость')
fig2.show()

print("\nГрафики отображены")
# ============================================
# 5. Обработка строк с нулевыми значениями (пропуски)
# ============================================
print("\n"+"="*50 +
      "\n5. Обработка строк с нулевыми значениями (пропуски)\n"+
      "="*50)

# Проверка пропусков
missing = df.isnull().sum()
missing_percent = (missing / len(df)) * 100

print("\nКоличество пропусков по каждому признаку:\n" + "-"*50)
for col in df.columns:
    print(f"{col:35} : {missing[col]} пропусков ({missing_percent[col]:.1f}%)")

# ============================================
# 6. Построение тепловой карты
# ============================================
print("\n"+"="*50 +
      "\n6. Построение тепловой карты\n"+
      "="*50)

# Числовые признаки для корреляции
numeric_cols = ['age', 'daily_screen_time_hours', 'phone_usage_before_sleep_minutes',
                'sleep_duration_hours', 'sleep_quality_score', 'stress_level',
                'caffeine_intake_cups', 'physical_activity_minutes',
                'notifications_received_per_day', 'mental_fatigue_score']

# Тепловая карта 1: Корреляция всех числовых признаков
plt.figure(figsize=(12, 10))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f',
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
plt.title('Тепловая карта 1: Корреляция числовых признаков', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Тепловая карта 2: Корреляция только со стрессом (отсортированная)
plt.figure(figsize=(8, 6))
stress_corr = df[numeric_cols].corr()['stress_level'].drop('stress_level').sort_values(ascending=False)
sns.barplot(x=stress_corr.values, y=stress_corr.index)
plt.title('Тепловая карта 2: Влияние признаков на уровень стресса', fontsize=14, fontweight='bold')
plt.xlabel('Коэффициент корреляции со стрессом')
plt.ylabel('Признаки')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.tight_layout()
plt.show()

print("\nТепловая карта отображена")

# ============================================
# 7. Устранение дубликатов строк
# ============================================
print("\n"+"="*50 +
      "\n7. Устранение дубликатов строк\n"+
      "="*50)

# Количество строк до удаления
print(f"\nКоличество строк ДО удаления дубликатов: {len(df)}")

# Проверка на дубликаты
duplicates_count = df.duplicated().sum()
print(f"Найдено дубликатов: {duplicates_count}")

# Удаление дубликатов
df = df.drop_duplicates()

# Количество строк после удаления
print(f"\nКоличество строк ПОСЛЕ удаления дубликатов: {len(df)}")

# Результат
if duplicates_count == 0:
    print("\n" + "-"*50 +
    "\nРЕЗУЛЬТАТ: Дубликаты отсутствуют\n" + "-"*50)
else:
    print(f"\nУдалено {duplicates_count} дубликатов"
          f"\nОсталось {len(df)} уникальных строк")

# ============================================
# 8. Появление нетипичных выбросов по признакам
# ============================================
print("\n"+"="*50 +
      "\n8. Появление нетипичных выбросов по признакам\n"+
      "="*50)

# Числовые признаки
numeric_cols = ['age', 'daily_screen_time_hours', 'phone_usage_before_sleep_minutes',
                'sleep_duration_hours', 'sleep_quality_score', 'stress_level',
                'caffeine_intake_cups', 'physical_activity_minutes',
                'notifications_received_per_day', 'mental_fatigue_score']

# Функция подсчета выбросов
def get_outliers_count(df, col):
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return len(df[(df[col] < lower) | (df[col] > upper)])

# Подсчет выбросов
print("\nВыбросы по признакам:\n" + "-" * 50)
for col in numeric_cols:
    count = get_outliers_count(df, col)
    if count > 0:
        print(f" {col}: {count} выбросов")
    else:
        print(f" {col}: выбросов нет")

# Визуализация
fig, axes = plt.subplots(5, 2, figsize=(14, 14))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    axes[i].boxplot(df[col], vert=True, patch_artist=True,
                    boxprops=dict(facecolor='lightblue'),
                    flierprops=dict(marker='o', markerfacecolor='red', markersize=6))
    axes[i].set_title(f'{col}', fontsize=10)
    axes[i].set_ylabel('Значение')
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Boxplot для выявления выбросов', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ИТОГ
print("\n" + "-"*50 + "\nИТОГОВЫЙ ВЫВОД ПО ВЫБРОСАМ:\n" + "-"*50)

outliers_found = [col for col in numeric_cols if get_outliers_count(df, col) > 0]
if outliers_found:
    print(f"\n Выбросы обнаружены в признаках: {', '.join(outliers_found)}")
else:
    print("\n Выбросы отсутствуют во всех признаках.")

# ============================================
# 9. Анализ диапазонов значений
# ============================================
print("\n"+"="*50 +
      "\n9. Анализ диапазонов значений\n"+
      "="*50)

print("\n Общий boxplot отображен")

# Числовые признаки
numeric_cols = ['age', 'daily_screen_time_hours', 'phone_usage_before_sleep_minutes',
            'sleep_duration_hours', 'sleep_quality_score', 'stress_level',
            'caffeine_intake_cups', 'physical_activity_minutes',
            'notifications_received_per_day', 'mental_fatigue_score']


# ОБЩИЙ BOXPLOT НА ОДНОМ ГРАФИКЕ (БЕЗ НОРМАЛИЗАЦИИ)
fig, ax = plt.subplots(figsize=(14, 6))

bp = ax.boxplot([df[col].dropna() for col in numeric_cols],
                vert=True, patch_artist=True,
                boxprops=dict(facecolor='steelblue', alpha=0.6),
                flierprops=dict(marker='o', markerfacecolor='red', markersize=3, alpha=0.5))

ax.set_xticklabels(numeric_cols, rotation=45, ha='right', fontsize=9)
ax.set_ylabel('Значения (исходный масштаб)', fontsize=10)
ax.set_title('Сравнение диапазонов признаков (исходные единицы)',
             fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ВЫВОДЫ
print("\n" + "-" * 50 + "\nВЫВОДЫ:\n" + "-" * 50)

# Разница масштабов
scales = {}
for col in numeric_cols:
    scales[col] = df[col].max() - df[col].min()

max_col = max(scales, key=scales.get)
min_col = min(scales, key=scales.get)
ratio = scales[max_col] / scales[min_col]

print(f"""
1. Самый широкий диапазон: {max_col} (размах {scales[max_col]:.0f}).
   Самый узкий диапазон: {min_col} (размах {scales[min_col]:.0f}).
   Разница в {ratio:.0f} раз!

2. На общем графике:
   - sleep_quality_score, stress_level, mental_fatigue_score (1-10) — узкие полоски.
   - notifications_received_per_day (20-299) — широкая коробка.
   - caffeine_intake_cups (0-4) — самая узкая.

3. Требуется StandardScaler(нормализация) для линейных моделей и нейросетей.
""")
# ============================================
# 10. Условная фильтрация сэмплов
# ============================================
print("="*50 +
      "\n10. Условная фильтрация сэмплов\n"+
      "="*50)

# Фильтр 1: Люди с высоким стрессом (>7) и плохим сном (<5)
filter1 = df[(df['stress_level'] > 7) & (df['sleep_quality_score'] < 5)]
print(f"\n  ФИЛЬТР 1: Стресс > 7 И Качество сна < 5"
      f"\nКоличество записей: {len(filter1)} "
      f"\nПримеры:"
      f"\n{filter1[['age', 'gender', 'stress_level', 'sleep_quality_score']].head()}")

# Фильтр 2: Высокое экранное время (>8ч) и низкая активность (<30 мин)
filter2 = df[(df['daily_screen_time_hours'] > 8) & (df['physical_activity_minutes'] < 30)]
print(f"\n  ФИЛЬТР 2: Экранное время > 8ч И Активность < 30 мин "
      f"\nКоличество записей: {len(filter2)} "
      f"\nПримеры: "
      f"\n{filter2[['age', 'occupation', 'daily_screen_time_hours', 'physical_activity_minutes']].head()}")

# Фильтр 3: Люди (<18) с высоким кофеином (>3 чашек)
filter3 = df[(df['age'] <= 18) & (df['caffeine_intake_cups'] > 3)]
print(f"\n  ФИЛЬТР 3: Возраст ≤ 18 И Кофеин > 3 чашек "
      f"\nКоличество записей: {len(filter3)} "
      f"\nПримеры: "
      f"\n{filter3[['age', 'occupation', 'caffeine_intake_cups', 'sleep_duration_hours']].head()}"
      + "\n" + "-"*50 + "\nВЫВОД ПО ФИЛЬТРАЦИИ:\n" + "-"*50 + """\n
- Фильтр 1: Выявлены люди из группы риска (высокий стресс + плохой сон)
- Фильтр 2: Выявлены пользователи с вредным образом жизни (много экрана + мало активности)
- Фильтр 3: Выявлена молодежь с высоким потреблением кофеина
""")

# ============================================
# 11. Добавление шума в признаки (исключая целевые)
# ============================================
print("="*50 +
      "\n11. Добавление шума в признаки (исключая целевые)\n"+
      "="*50)

# Сохраняем оригинальные значения для сравнения
df_original = df.copy()

# Целевые признаки
target_cols = ['stress_level', 'sleep_quality_score', 'mental_fatigue_score']

# Признаки для добавления шума (исключая целевые)
noise_cols = ['daily_screen_time_hours', 'physical_activity_minutes']

print(f"\nВыбранные признаки для добавления шума (исключая целевые):"
      f"\n- {noise_cols[0]} (экранное время)"
      f"\n- {noise_cols[1]} (физическая активность)")

# Добавление шума в daily_screen_time_hours (нормальный шум)
np.random.seed(42)
noise1 = np.random.normal(0, 0.5, size=len(df))
df['daily_screen_time_hours_noisy'] = df['daily_screen_time_hours'] + noise1
df['daily_screen_time_hours_noisy'] = df['daily_screen_time_hours_noisy'].clip(lower=0)

# Добавление шума в physical_activity_minutes (равномерный шум)
noise2 = np.random.uniform(-5, 5, size=len(df))
df['physical_activity_minutes_noisy'] = df['physical_activity_minutes'] + noise2
df['physical_activity_minutes_noisy'] = df['physical_activity_minutes_noisy'].clip(lower=0)

print(f"\nРезультат добавления шума:"
      f"\ndaily_screen_time_hours: добавлен шум N(0, 0.5)"
      f"\nphysical_activity_minutes: добавлен шум U(-5, 5)")

# Визуализация изменений
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(df['daily_screen_time_hours'], bins=30, alpha=0.5, label='Оригинал', color='blue')
axes[0].hist(df['daily_screen_time_hours_noisy'], bins=30, alpha=0.5, label='С шумом', color='red')
axes[0].set_title('Экранное время: оригинал vs с шумом')
axes[0].legend()

axes[1].hist(df['physical_activity_minutes'], bins=30, alpha=0.5, label='Оригинал', color='blue')
axes[1].hist(df['physical_activity_minutes_noisy'], bins=30, alpha=0.5, label='С шумом', color='red')
axes[1].set_title('Физическая активность: оригинал vs с шумом')
axes[1].legend()

plt.tight_layout()
plt.show()

# ============================================
# 12. Преобразование числовых данных в категориальные
# ============================================
print("\n"+"="*50 +
      "\n12. Преобразование числовых данных в категориальные\n"+
      "="*50)

# Преобразование возраста в возрастные группы
bins_age = [18, 25, 35, 45, 55, 70]
labels_age = ['18-25', '26-35', '36-45', '46-55', '56-70']
df['age_group'] = pd.cut(df['age'], bins=bins_age, labels=labels_age, right=False)

print(f"\nПреобразование 1: age - age_group (5 групп) "
      f"\nРаспределение по группам: {df['age_group'].value_counts().sort_index()}")

# Преобразование экранного времени в категории
bins_screen = [0, 3, 6, 9, 12]
labels_screen = ['Низкое (<3ч)', 'Среднее (3-6ч)', 'Высокое (6-9ч)', 'Очень высокое (>9ч)']
df['screen_category'] = pd.cut(df['daily_screen_time_hours'], bins=bins_screen, labels=labels_screen, right=False)

print(f"\nПреобразование 2: daily_screen_time_hours - screen_category (4 категории) "
      "\nРаспределение по категориям: \n{df['screen_category'].value_counts().sort_index()}")

# Преобразование физической активности в категории
bins_activity = [0, 30, 60, 90, 120]
labels_activity = ['Низкая', 'Средняя', 'Высокая', 'Очень высокая']
df['activity_category'] = pd.cut(df['physical_activity_minutes'], bins=bins_activity, labels=labels_activity, right=False)

print(f"\nПреобразование 3: physical_activity_minutes - activity_category (4 категории) "
      "\nРаспределение по категориям: \n{df['activity_category'].value_counts().sort_index()}\n\n"
      + "-"*50 + """\nЦель преобразования числовых данных в категориальные:
- Агрегация редких данных (возраст объединен в группы)
- Упрощение анализа и визуализации
- Подготовка данных для некоторых моделей машинного обучения
- Скрытие точных значений (например, возраста) для обезличивания
""")
# ============================================
# 13. Дополнительные преобразования
# ============================================
print("="*50 +
      "\n13. Дополнительные преобразования\n"+
      "="*50)

# Проверка на разные обозначения одного и того же значения
print(f"\nПроверка уникальных значений в категориальных признаках:"
      f"\n1.gender: {df['gender'].unique()} "
      f"\n2.occupation: {df['occupation'].unique()}")

# Все значения уже в едином формате, преобразования не требуются
print("\nВсе категориальные признаки уже в едином формате: "
      "\n- gender: Male/Female (без дублирования)"
      "\n- occupation: все начинаются с заглавной буквы")

# Создание нового составного признака: индекс цифровой нагрузки
# (экранное время + телефон перед сном) / качество сна
df['digital_load_index'] = (df['daily_screen_time_hours'] * 60 + df['phone_usage_before_sleep_minutes']) / (df['sleep_quality_score'] + 1)

print(f"\nСоздан новый признак: digital_load_index "
      f"\nФормула: (экранное_время_в_минутах + телефон_перед_сном) / (качество_сна + 1) "
      f"\nДиапазон значений: от {df['digital_load_index'].min():.1f} "
      f"до {df['digital_load_index'].max():.1f} "
      f"\nСреднее значение: {df['digital_load_index'].mean():.1f}")

# Создание признака: соотношение стресса к физической активности
df['stress_per_activity'] = df['stress_level'] / (df['physical_activity_minutes'] + 1)

print(f"\nСоздан новый признак: stress_per_activity "
      f"\nФормула: стресс / (физическая_активность + 1) "
      f"\nДиапазон значений: от {df['stress_per_activity'].min():.2f} \n"
      f"до {df['stress_per_activity'].max():.2f}" + "\n" + "-"*50 + """\n
Ценность преобразований:
- Новые признаки могут выявить скрытые закономерности
- Позволяют оценить взаимосвязи между разными характеристиками
- Упрощают дальнейший анализ
""")
# ============================================
# 14. Оценка изменения в данных после фильтрации. (повторная визуализация ВСЕХ гистограмм)
# ============================================
print("="*50 +
      "\n14. Оценка изменения в данных после фильтрации. (повторная визуализация ВСЕХ гистограмм)\n"+
      "="*50)


# Создаем фильтрованную версию (убираем экстремальные значения экранного времени)
df_filtered = df[(df['daily_screen_time_hours'] >= 2) & (df['daily_screen_time_hours'] <= 10)]

print(f"\nРазмер данных ДО фильтрации: {len(df)} записей "
      f"\nРазмер данных ПОСЛЕ фильтрации: {len(df_filtered)} записей "
      f"\nУдалено записей: {len(df) - len(df_filtered)}")

# Все числовые признаки для сравнения
numeric_cols = ['age', 'daily_screen_time_hours', 'phone_usage_before_sleep_minutes',
                'sleep_duration_hours', 'sleep_quality_score', 'stress_level',
                'caffeine_intake_cups', 'physical_activity_minutes',
                'notifications_received_per_day', 'mental_fatigue_score']

# ПОВТОРНАЯ ВИЗУАЛИЗАЦИЯ ВСЕХ ГИСТОГРАММ
fig, axes = plt.subplots(5, 2, figsize=(14, 18))
axes = axes.flatten()

for i, col in enumerate(numeric_cols):
    axes[i].hist(df[col], bins=25, alpha=0.5, label='До фильтрации', color='blue')
    axes[i].hist(df_filtered[col], bins=25, alpha=0.5, label='После фильтрации', color='green')
    axes[i].set_title(f'{col}')
    axes[i].legend(fontsize=8)
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Сравнение гистограмм до и после фильтрации', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# Сравнение средних значений для всех признаков
print("\n" + "-"*50 +
      "\nСРАВНЕНИЕ СРЕДНИХ ЗНАЧЕНИЙ (ДО vs ПОСЛЕ):\n" +
      "-"*50 + f"\n{'Признак':<35} {'До':<12} {'После':<12} {'Изменение':<10}\n" + "-" * 50)

for col in numeric_cols:
    mean_before = df[col].mean()
    mean_after = df_filtered[col].mean()
    change = mean_after - mean_before
    arrow = "+" if change > 0 else "-" if change < 0 else "="
    print(f"{col:<35} {mean_before:<12.2f} {mean_after:<12.2f} {arrow} {abs(change):.2f}")

print("\n" + "-"*50 + "\nГИПОТЕЗЫ О ВЛИЯНИИ ИЗМЕНЕНИЙ:\n" + "-"*50 + """\n
 Гипотеза 1: Фильтрация по экранному времени влияет только на связанные признаки
      - sleep_quality_score, mental_fatigue_score могут измениться
      - age, gender, occupation останутся почти без изменений

 Гипотеза 2: Удаление экстремальных значений делает распределения более нормальными
      - Гистограммы становятся более гладкими
      - Выбросы исчезают

 Гипотеза 3: Средние значения признаков изменяются незначительно
      - Так как удаляется небольшой процент данных
      - Общая структура данных сохраняется
""")

# ============================================
# 15. Запросы на группировку
# ============================================
print("="*50 +
      "\n15. Запросы на группировку\n"+
      "="*50)

# Группировка 1: Средние показатели по роду занятий
print("\nГРУППИРОВКА 1: По роду занятий (occupation)\n"
      + "-" * 50)

group1 = df.groupby('occupation').agg({
    'stress_level': 'mean',
    'sleep_quality_score': 'mean',
    'daily_screen_time_hours': 'mean',
    'physical_activity_minutes': 'mean',
    'caffeine_intake_cups': 'mean'
}).round(2).sort_values('stress_level', ascending=False)

print(group1)

# Группировка 2: Средние показатели по полу
print("\nГРУППИРОВКА 2: По полу (gender)\n" + "-" * 50)

group2 = df.groupby('gender').agg({
    'stress_level': 'mean',
    'sleep_quality_score': 'mean',
    'sleep_duration_hours': 'mean',
    'mental_fatigue_score': 'mean'
}).round(2)

print(group2)

# Группировка 3: Двойная группировка (по полу и возрастной группе)
print("\nГРУППИРОВКА 3: По полу и возрастной группе\n"
      + "-" * 50)

# Создаем возрастные группы
bins = [18, 30, 40, 50, 60, 70]
labels = ['18-30', '31-40', '41-50', '51-60', '61-70']
df['age_range'] = pd.cut(df['age'], bins=bins, labels=labels, right=False)

group3 = df.groupby(['gender', 'age_range']).agg({
    'stress_level': ['mean', 'count'],
    'sleep_quality_score': 'mean'
}).round(2)

print(group3)

# Общая сводная таблица (pivot table)
print("\nСВОДНАЯ ТАБЛИЦА: Средний стресс по полу и возрастной группе\n"
      + "-" * 50)

pivot_table = pd.pivot_table(df,
                             values='stress_level',
                             index='age_range',
                             columns='gender',
                             aggfunc='mean',
                             fill_value=0)
print(pivot_table)

print("\n" + "-"*50 + "\nВЫВОД ПО ГРУППИРОВКЕ:\n"
      + "-"*50 + """\n
 Группировка 1 (по профессии):
   - Самый высокий стресс у: Doctors и Lawyers
   - Самый низкий стресс у: Retired и Unemployed

 Группировка 2 (по полу):
   - Уровень стресса примерно одинаков у мужчин и женщин
   - Качество сна незначительно отличается

 Группировка 3 (по полу и возрасту):
   - Позволяет увидеть交叉-эффекты
   - Например, молодые женщины могут иметь более высокий стресс
""")
# ============================================
# 16. Перечень всех категорий по признакам
# ============================================
print("="*50 +
      "\n16. Перечень всех категорий по признакам\n"+
      "="*50)

# Категориальные признаки
cat_cols = ['gender', 'occupation']

for col in cat_cols:
    print(f"\nПризнак: {col} "
          f"\nКоличество категорий: {df[col].nunique()} "
          f"\nКатегории: {list(df[col].unique())}")

    # Частота каждой категории
    print(f"\nРаспределение частот:")
    for val, count in df[col].value_counts().items():
        percent = (count / len(df)) * 100
        print(f"    - {val}: {count} ({percent:.1f}%)")

print("\n" + "-" * 50 +
      "\nИТОГО:\n" + "-" * 50 + """\n- gender: 2 категории (Male, Female) - сбалансированы ~50%/50%
- occupation: 10 категорий (различные профессии) - равномерное распределение
""")
# ============================================
# 17. Диаграммы распределения категориальных данных
# ============================================
print("="*50 +
      "\n17. Диаграммы распределения категориальных данных\n"+
      "="*50)

# Создаем графики
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# График 1: Распределение по полу (круговая диаграмма)
gender_counts = df['gender'].value_counts()
axes[0].pie(gender_counts.values, labels=gender_counts.index, autopct='%1.1f%%',
            colors=['lightblue','lightgreen', 'lightcoral'], startangle=90)
axes[0].set_title('Распределение по полу', fontsize=12, fontweight='bold')

# График 2: Распределение по роду занятий (столбчатая диаграмма)
occupation_counts = df['occupation'].value_counts()
bars = axes[1].bar(occupation_counts.index, occupation_counts.values,
                   color='lightgreen', edgecolor='black')
axes[1].set_title('Распределение по роду занятий', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Профессия')
axes[1].set_ylabel('Количество')
axes[1].tick_params(axis='x', rotation=45, labelsize=9)

# Добавляем цифры на столбцы
for bar, val in zip(bars, occupation_counts.values):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                 str(val), ha='center', va='bottom', fontsize=9)

plt.suptitle('Распределение категориальных признаков', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

print("\nДиаграммы отображен")

print("\n" + "-"*50 +
      "\nВЫВОД ПО ДИАГРАММАМ:\n" + "-"*50 + """\n
 По полу:
   - Male: ~50%
   - Female: ~50%
   - Выборка сбалансирована по полу

 По роду занятий:
   - 10 различных профессий
   - Распределение равномерное (категория ~1000 записей)
   - Нет доминирующих профессий
""")
# ============================================
# 18. Преобразование категориальных данных в числовые
# Способ: Label Encoding (порядковое кодирование)
# ============================================
from sklearn.preprocessing import LabelEncoder

print("="*50 +
      "\n18. Преобразование категориальных данных в числовые\n"+
      "="*50)

# Кодирование признака gender
le_gender = LabelEncoder()
df['gender_encoded'] = le_gender.fit_transform(df['gender'])

print("\nПреобразование 1: gender - gender_encoded\n"
      +"-" * 50+"\nСоответствие категорий числам:")
for i, category in enumerate(le_gender.classes_):
    print(f"   {category} - {i}")

# Кодирование признака occupation
le_occupation = LabelEncoder()
df['occupation_encoded'] = le_occupation.fit_transform(df['occupation'])

print("\nПреобразование 2: occupation - occupation_encoded\n" +
      "-" * 50 + "\nСоответствие категорий числам:")
for i, category in enumerate(le_occupation.classes_):
    print(f"   {category} - {i}")

# результат
print(f"\nПример преобразования: "
      f"{df[['gender', 'gender_encoded', 'occupation', 'occupation_encoded']].head(10)}" +
      "\n" + "-"*50 + "\nВЫВОД:\n" + "-"*50 + """\n
Label Encoding заменяет каждую категорию уникальным числом:
- Плюсы: простота, не создает много новых столбцов
- Минусы: может создать ложный порядок между категориями
- Применение: подходит для порядковых категорий и деревьев решений
""")

# ============================================
# 19. Агрегация данных (редкие категории)
# ============================================

print("="*50 +
      "\n19. Агрегация данных (редкие категории)\n"+
      "="*50)

# Анализ частоты категорий в occupation
occupation_freq = df['occupation'].value_counts()
print(f"\nЧастота категорий 'occupation': {occupation_freq}")

# Проверка на редкие категории (менее 5% от выборки = 750 записей)
threshold = 0.05 * len(df)
rare_occupations = occupation_freq[occupation_freq < threshold].index.tolist()

print(f"\nПорог редкости: {threshold:.0f} записей (5% от {len(df)} записей) "
      f"\nРезультат проверки:")

if len(rare_occupations) == 0:
    print(f"\nРедкие категории ОТСУТСТВУЮТ"
          f"\nВсе 10 профессий имеют частоту выше порога "
          f"\nМинимальная частота: {occupation_freq.min()} записей "
          f"\nМаксимальная частота: {occupation_freq.max()} записей "
          f"\nАгрегация данных НЕ ТРЕБУЕТСЯ")
else:
    print(f"Редкие категории: {rare_occupations} \nТребуется агрегация")
# ============================================
# 20. Введение новой категории
# ============================================
print("\n"+"="*50 +
      "\n20. Введение новой категории\n"+
      "="*50)

# Создание новой категории "Группа риска"
conditions = [
    (df['stress_level'] > 7) & (df['sleep_quality_score'] < 5) & (df['daily_screen_time_hours'] > 8),
    (df['stress_level'] >= 4) & (df['stress_level'] <= 7) & (df['sleep_quality_score'] >= 4) & (df['sleep_quality_score'] <= 7),
    (df['stress_level'] < 4) & (df['sleep_quality_score'] > 7)
]

choices = ['Высокий риск', 'Средний риск', 'Низкий риск']

df['risk_group'] = np.select(conditions, choices, default='Средний риск')

print("\nНовая категория: risk_group\n" + "-" * 50 + "\nУсловия формирования: "
      "\nВысокий риск: стресс > 7, качество сна < 5, экранное время > 8ч "
      "\nСредний риск: стресс 4-7, качество сна 4-7 "
      "\nНизкий риск: стресс < 4, качество сна > 7")

# Подсчет с фиксированным порядком категорий
print("\nРаспределение новой категории:")
risk_counts = df['risk_group'].value_counts()
print(risk_counts)

# Визуализация с правильным порядком и цветами
# Фиксируем порядок категорий
order = ['Высокий риск', 'Средний риск', 'Низкий риск']
counts_sorted = df['risk_group'].value_counts().reindex(order)
colors_sorted = ['red', 'orange', 'green']

plt.figure(figsize=(8, 5))
bars = plt.bar(counts_sorted.index, counts_sorted.values, color=colors_sorted, edgecolor='black')
plt.title('Распределение групп риска', fontsize=12, fontweight='bold')
plt.xlabel('Группа')
plt.ylabel('Количество')
plt.grid(True, alpha=0.3, axis='y')

for bar, val in zip(bars, counts_sorted.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             str(val), ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.show()

# ============================================
# 21. ВЫВОДЫ ПО 1 ГЛАВЕ
# ============================================
print("\n"+"="*50 +
      "\nВЫВОДЫ ПО 1 ГЛАВЕ\n"+
      "="*50)
# ============================================
# 22. Особенности предметной области
# ============================================
print("\n"+"="*50 +
      "\n22. Особенности предметной области\n"+
      "="*50 + """\n
 ПОЧЕМУ БЫЛ СОЗДАН ДАТАСЕТ:

   Датасет создан для исследования взаимосвязи между:
   - Цифровыми привычками (экранное время, телефон перед сном)
   - Образом жизни (физическая активность, кофеин)
   - Психологическим благополучием (стресс, качество сна, усталость)

 ОСОБЕННОСТИ, ВЛИЯЮЩИЕ НА РЕШЕНИЕ ЗАДАЧ:

   1. Синтетическая природа данных
      - Данные сгенерированы искусственно
      - Не отражают реальных медицинских показателей
      - Но позволяют отработать методы анализа

   2. Отсутствие выбросов и пропусков
      - Упрощает начальную обработку
      - Не требует сложных методов очистки

   3. Сбалансированность категорий
      - Все профессии представлены равномерно
      - Пол сбалансирован (50/50)

   4. Линейные взаимосвязи
      - Корреляции заложены искусственно
      - Упрощает визуализацию и интерпретацию

 ВЛИЯНИЕ ОСОБЕННОСТЕЙ НА РЕШЕНИЕ ЗАДАЧ:

   + Плюсы: легко обрабатывать, нет проблем с качеством данных
   - Минусы: синтетические данные могут не отражать реальные закономерности
""")

# ============================================
# 23. Гипотезы для корректного использования датасета
# ============================================
print("="*50 + "\n23. ГИПОТЕЗЫ ДЛЯ КОРРЕКТНОГО ИСПОЛЬЗОВАНИЯ ДАТАСЕТА\n" + "="*50+"""\n
 ГИПОТЕЗА 1: Влияние экранного времени на качество сна

   Предположение:
   Увеличение ежедневного экранного времени более 6 часов приводит
   к снижению качества сна (оценка < 5).

   Обоснование:
   Доказано, что голубой
   свет от экранов подавляет выработку мелатонина - гормона сна.

   Значимые признаки:
   - daily_screen_time_hours (экранное время)
   - sleep_quality_score (качество сна)

 ГИПОТЕЗА 2: Роль физической активности в снижении стресса

   Предположение:
   Регулярная физическая активность (> 60 минут в день) снижает
   уровень стресса (stress_level < 4).

   Обоснование:
   Физическая активность способствует выработке эндорфинов и улучшает
   психоэмоциональное состояние.

   Значимые признаки:
   - physical_activity_minutes (физическая активность)
   - stress_level (уровень стресса)

 ГИПОТЕЗА 3: Влияние кофеина на качество сна

   Предположение:
   Потребление более 3 чашек кофеина в день снижает качество сна.

   Обоснование:
   Кофеин блокирует аденозиновые рецепторы, что мешает засыпанию
   и ухудшает структуру сна.

   Значимые признаки:
   - caffeine_intake_cups (кофеин)
   - sleep_quality_score (качество сна)
""")

# ============================================
# 24. Гипотезы для вероятного неэтичного использования датасета
# ============================================
print("="*50 + "\n24. ГИПОТЕЗЫ ДЛЯ ВЕРОЯТНОГО НЕЭТИЧНОГО ИСПОЛЬЗОВАНИЯ ДАТАСЕТА\n" + "="*50 + """
 ГИПОТЕЗА 1: Дискриминация по профессиональному признаку

   Описание риска:
   Модель, обученная на данном датасете, может сделать вывод, что
   представители определенных профессий (например, Doctors, Lawyers)
   более подвержены стрессу.

   Неэтичное применение:
   - Страховые компании могут повышать тарифы для "стрессовых" профессий
   - Работодатели могут дискриминировать кандидатов по профессии
   - Банки могут отказывать в кредитах

   Аналогия из реальной жизни:
   В книге К. О’Нил «Убийственные большие данные» описан кейс, когда
   модель, обученная на данных по статистике преступлений, ошибочно
   стала прогнозировать, что люди определенной национальности более
   склонны к преступлениям.

 ГИПОТЕЗА 2: Использование для слежки и манипуляции

   Описание риска:
   Датасет содержит информацию о цифровых привычках (экранное время,
   телефон перед сном, уведомления).

   Неэтичное применение:
   - Рекламные компании могут определять уязвимых пользователей
     (с высоким стрессом, плохим сном) для таргетинга
   - Работодатели могут следить за режимом сна и активностью сотрудников
   - Платформы могут манипулировать временем использования приложений

   Почему это опасно:
   Знание о психологическом состоянии пользователя позволяет
   манипулировать его поведением (например, показывать тревожный
   контент в периоды высокого стресса).

 ПОТЕНЦИАЛЬНЫЕ ВОЗМОЖНОСТИ РАСШИРЕНИЯ ДАТАСЕТА

   Какие признаки можно добавить:
   1. Геолокация (страна, город, тип населенного пункта)
   2. Уровень дохода / социально-экономический статус
   3. Образование (уровень, специальность)
   4. Семейное положение, наличие детей
   5. Наличие хронических заболеваний
   6. Употребление алкоголя / курение
   7. Тип устройства (смартфон, компьютер, планшет)

 НОВЫЕ ЗАДАЧИ ПОСЛЕ РАСШИРЕНИЯ:
   1. Предсказание профессионального выгорания
   2. Анализ влияния социально-экономических факторов на стресс
   3. Региональные сравнения качества жизни
   4. Персонализированные рекомендации по улучшению сна
""" + "\n" + "=" * 50 + "\nАНАЛИЗ ЗАВЕРШЁН\n" + "=" * 50)

# ============================================
#--------------------END----------------------
# ============================================
