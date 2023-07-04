import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Задаємо список документів
with open('file.txt', 'r') as f:
    text = f.read()

# Ініціалізуємо TF-IDF векторизатор
vectorizer = TfidfVectorizer()
# Обчислюємо TF-IDF для кожного документу
tfidf = vectorizer.fit_transform([text])
print(tfidf.toarray())
# Отримуємо список всіх слів
words = vectorizer.get_feature_names_out()

# Розділяємо текст на окремі документи
docs = text.split('\n\n')

# Виводимо топ-10 важливих слів для кожного документу
for i, doc in enumerate(docs):
    print(f"Документ {i+1}:")
    # Обчислюємо TF-IDF для поточного документу
    tfidf_doc = vectorizer.transform([doc])
    # Отримуємо відсортований за важливістю список індексів слів
    indices = tfidf_doc.indices[np.argsort(tfidf_doc.data)][::-1][:10]
    # Виводимо відповідні слова для цих індексів
    top_words = [words[idx] for idx in indices]
    print(", ".join(top_words))
    print()