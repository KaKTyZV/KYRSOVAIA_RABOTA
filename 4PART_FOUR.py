# ============================================
# ГЛАВА 4. АНАЛИЗ ТЕКСТОВЫХ ДАННЫХ (BBC)
# ============================================
print("\n" + "=" * 60 + "\nГЛАВА 4. АНАЛИЗ ТЕКСТОВЫХ ДАННЫХ (BBC)\n" + "=" * 60)

import pandas as pd
import matplotlib.pyplot as plt
import re
from collections import Counter
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================
# 1. ЗАГРУЗКА И ПЕРВИЧНОЕ ЗНАКОМСТВО
# ============================================
print("\n" + "=" * 50 + "\n1. ЗАГРУЗКА И ПЕРВИЧНОЕ ЗНАКОМСТВО\n" + "=" * 50)

df = pd.read_csv("C:/Users/bobv6/Desktop/Kyrsovaia/4part_TEXT_data/bbc_data.csv")
df.columns = ['text', 'label']

print(f"\nЗагружено: {len(df)} текстов, {df['label'].nunique()} категорий "
      f"\nКатегории:")

# Статистика по категориям в процентах
cat_counts = df['label'].value_counts()
for cat, count in cat_counts.items():
    print(f"  {cat}: {count} ({count/len(df)*100:.1f}%)")

# ============================================
# 2. ОЧИСТКА ТЕКСТА (ко всем текстам)
# ============================================
print("\n" + "=" * 50 + "\n2. ОЧИСТКА ТЕКСТА\n" + "=" * 50)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)  # только английские буквы
    text = re.sub(r'\s+', ' ', text).strip()
    return text


df['clean_text'] = df['text'].apply(clean_text)

print(f"\nПример ДО: \n{df['text'].iloc[0][:200]} \nПример ПОСЛЕ: \n{df['clean_text'].iloc[0][:200]}")

# Статистика очистки
total_chars_before = df['text'].str.len().sum()
total_chars_after = df['clean_text'].str.len().sum()
removed_chars = total_chars_before - total_chars_after
removed_percent = removed_chars / total_chars_before * 100
print(f"\nСимволов ДО очистки:    {total_chars_before:,} "
      f"\nСимволов ПОСЛЕ очистки: {total_chars_after:,} "
      f"\nУдалено символов:       {removed_chars:,} ({removed_percent:.1f}%) "
      f"\nСредняя длина ДО:       {df['text'].str.len().mean():.0f} "
      f"\nСредняя длина ПОСЛЕ:    {df['clean_text'].str.len().mean():.0f}")

# ============================================
# 3. ЛЕММАТИЗАЦИЯ (ко всем текстам)
# ============================================
print("\n" + "=" * 50 + "\n3. ЛЕММАТИЗАЦИЯ\n" + "=" * 50)

lemmatizer = WordNetLemmatizer()

def lemmatize_text(text):
    words = word_tokenize(text)
    return ' '.join([lemmatizer.lemmatize(w) for w in words])

df['lemmas'] = df['clean_text'].apply(lemmatize_text)
df['lemmas'] = df['lemmas'].fillna('')

print(f"\nПример ДО лемматизации:"
      f"\n{df['clean_text'].iloc[0][:200]}"
      f"\nПример ПОСЛЕ лемматизации:"
      f"\n{df['lemmas'].iloc[0][:200]}")

# Статистика лемматизации
total_words_before = sum(len(word_tokenize(t)) for t in df['clean_text'])
total_words_after = sum(len(word_tokenize(t)) for t in df['lemmas'])
changed_words = 0
for i in range(2225):
    orig = word_tokenize(df['clean_text'].iloc[i])
    lemm = word_tokenize(df['lemmas'].iloc[i])
    changed_words += sum(1 for a, b in zip(orig, lemm) if a != b)
changed_percent = changed_words / total_words_before * 100

print(f"Слов ДО лемматизации:      {total_words_before:,} "
      f"\nСлов ПОСЛЕ лемматизации: {total_words_after:,} "
      f"\nИзменили форму:          {changed_words:,} ({changed_percent:.1f}%)")

# ============================================
# 4. ПОДСЧЁТ ЧАСТОТЫ СЛОВ (ко всем текстам)
# ============================================
print("\n" + "=" * 50 + "\n4. ТОП-10 САМЫХ ЧАСТЫХ СЛОВ\n" + "=" * 50)

all_words = ' '.join(df['lemmas']).split()
word_counts = Counter(all_words)

for word, count in word_counts.most_common(10):
    print(f"  {word}: {count}")

# График
top_words = word_counts.most_common(10)
words, counts = zip(*top_words)

plt.figure(figsize=(10, 5))
plt.bar(words, counts, color='steelblue')
plt.title('Топ-10 самых частых слов (BBC)')
plt.xlabel('Слово')
plt.ylabel('Частота')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Облако слов
wc = WordCloud(width=800, height=400, max_words=50, background_color='white').generate(' '.join(df['lemmas']))
plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Облако слов (BBC)')
plt.show()

# ============================================
# 5. УДАЛЕНИЕ СТОП-СЛОВ (ко всем текстам)
# ============================================
print("\n" + "=" * 50 + "\n5. ТОП-10 ПОСЛЕ УДАЛЕНИЯ СТОП-СЛОВ\n" + "=" * 50)

# Дополнительные стоп-слова (то, что пропустил NLTK)
extra_stops = [ 'wa', 'ha', 'u', 'mr', 'uk' ]
stop_words = list(stopwords.words('english')) + extra_stops

def remove_stopwords(text):
    words = text.split()
    return ' '.join([w for w in words if w not in stop_words])

df['no_stopwords'] = df['lemmas'].apply(remove_stopwords)

# Топ-10 после удаления стоп-слов
all_words_clean = ' '.join(df['no_stopwords']).split()
word_counts_clean = Counter(all_words_clean)

for word, count in word_counts_clean.most_common(10):
    print(f"  {word}: {count}")

print(f"\nПример ДО удаления стоп-слов: \n{df['lemmas'].iloc[0][:200]} "
      f"\nПример ПОСЛЕ удаления стоп-слов: \n{df['no_stopwords'].iloc[0][:200]}")

# Статистика удаления стоп-слов
words_before_stop = sum(len(t.split()) for t in df['lemmas'].head(500))
words_after_stop = sum(len(t.split()) for t in df['no_stopwords'].head(500))
removed_stop = words_before_stop - words_after_stop
removed_stop_percent = removed_stop / words_before_stop * 100
print(f"\nСлов ДО удаления:       {words_before_stop:,}"
      f"\nСлов ПОСЛЕ удаления:    {words_after_stop:,}"
      f"\nУдалено стоп-слов:      {removed_stop:,} ({removed_stop_percent:.1f}%)"
      f"\nСреднее стоп-слов на текст ДО:    {words_before_stop // 500}"
      f"\nСреднее стоп-слов на текст ПОСЛЕ: {words_after_stop // 500}")

# ============================================
# 6. TF-IDF (ко всем текстам)
# ============================================
print("\n" + "=" * 50 + "\n6. TF-IDF\n" + "=" * 50)

# Берём 500 текстов с удалёнными стоп-словами
texts_for_tfidf = df['no_stopwords'].dropna()

# Создаём и обучаем векторайзер
vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(texts_for_tfidf)

# Словарь
feature_names = vectorizer.get_feature_names_out()
print(f"Размер словаря: {len(feature_names)} слов")

# Топ-20 слов по TF-IDF (суммарная важность)
scores = tfidf_matrix.sum(axis=0).A1
top_indices = scores.argsort()[-20:][::-1]

print("\nТОП-20 СЛОВ ПО TF-IDF:\n" + "-" * 40 + f"\n{'№':<5} {'Слово':<20} {'TF-IDF':<10} \n" + "-" * 40)
for i, idx in enumerate(top_indices):
    print(f"{i+1:<5} {feature_names[idx]:<20} {scores[idx]:<10.2f}")

# Пример вектора для первого текста
print(f"\nПример 1 текста: {texts_for_tfidf.iloc[0]} "
      f"\nРазмер вектора: {tfidf_matrix[0].shape[1]} признаков "
      f"\nНенулевых значений: {tfidf_matrix[0].nnz} ")

# Показываем первые 10 ненулевых значений вектора
row = tfidf_matrix[0].toarray()[0]
nonzero_indices = row.nonzero()[0][:10]
print("\nПервые 10 ненулевых TF-IDF значений для 1 текста:")
for idx in nonzero_indices:
    print(f"  {feature_names[idx]}: {row[idx]:.4f}")

# ============================================
# 7. ИНФОРМАЦИОННЫЙ ПОИСК (по всем текстам)
# ============================================
print("\n" + "=" * 50 + "\n7. ИНФОРМАЦИОННЫЙ ПОИСК\n" + "=" * 50)

def search_texts(query, vectorizer, tfidf_matrix, texts, top_n=5):
    query_clean = clean_text(query)
    query_vec = vectorizer.transform([query_clean])
    similarities = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_indices = similarities.argsort()[-top_n:][::-1]

    results = []
    for idx in top_indices:
        results.append({
            'text': texts.iloc[idx][:150],
            'label': df['label'].iloc[idx],
            'similarity': similarities[idx]
        })
    return results

# Запрос 1: бизнес
print("\nТекст: 'stock market economy'")
for i, res in enumerate(search_texts('stock market economy', vectorizer, tfidf_matrix, df['text']), 1):
    print(f"{i}. [{res['label']}] (похожесть={res['similarity']:.3f}) {res['text']}...")

# Запрос 2: спорт
print("\nТекст: 'film music award actor oscar album song band'")
for i, res in enumerate(search_texts('film music award actor oscar album song band', vectorizer, tfidf_matrix, df['text']), 1):
    print(f"{i}. [{res['label']}] (похожесть={res['similarity']:.3f}) {res['text']}...")

# ============================================
# ВЫВОДЫ ПО 4 ГЛАВЕ
# ============================================
print("\n" + "=" * 50 +
      "\nВЫВОДЫ ПО 4 ГЛАВЕ\n"
      + "=" * 50)

print(f"""
1. Объём данных:
   - {len(df)} текстов
   - 5 категорий: sport ({cat_counts.get('sport', 0)}), business ({cat_counts.get('business', 0)}), 
     politics ({cat_counts.get('politics', 0)}), tech ({cat_counts.get('tech', 0)}), 
     entertainment ({cat_counts.get('entertainment', 0)})
   - Пропусков: {df.isnull().sum().sum()}

2. Очистка текста:
   - Удалено символов: {removed_chars:,} ({removed_percent:.1f}%)
   - Средняя длина: {total_chars_before/len(df):.0f} → {total_chars_after/len(df):.0f} символов

3. Лемматизация:
   - Обработаны все тексты
   - Слов ДО: {total_words_before:,}
   - Изменили форму: {changed_words:,} ({changed_percent:.1f}%)

4. Частотный анализ:
   - Топ-слов до очистки: the, to, a, of, and (служебные части речи)
   - После удаления стоп-слов: said, year, people, new, game

5. Удаление стоп-слов:
   - Удалено: {removed_stop:,} слов ({removed_stop_percent:.1f}%)
   - Среднее на текст: {words_before_stop//500} → {words_after_stop//500} слов

6. TF-IDF:
   - Размер словаря: {len(feature_names)} слов
   - Топ-слова: said ({scores[top_indices[0]]:.0f}), year, game, film, people
   - Ненулевых значений в первом векторе: {tfidf_matrix[0].nnz}

7. Информационный поиск:
   - Запрос 'stock market economy' → business
   - Запрос 'film music award' → entertainment
   - Точность: высокая

8. Пригодность: данные полностью пригодны для классификации новостей.
   Рекомендуется логистическая регрессия или Random Forest как базовая модель.
""")

print("\n" + "=" * 50 + "\nАНАЛИЗ ЗАВЕРШЁН\n" + "=" * 50)

# ============================================
#--------------------END----------------------
# ============================================