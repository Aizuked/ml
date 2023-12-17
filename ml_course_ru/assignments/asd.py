from sklearn.metrics import silhouette_score
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

# Загрузка данных
data = pd.read_csv(r'C:\Users\aizyk\PycharmProjects\ml\ml_course_ru\assignments\Sponge\sponge.data',
                   delim_whitespace=True)

# Перевод строчных данных в числовые
le = preprocessing.LabelEncoder()
data = data.apply(le.fit_transform)

# Масштабирование признаков
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data)

# Инициализация моделей кластеризации
models = [
    KMeans(n_clusters=3),
    AgglomerativeClustering(n_clusters=3),
    DBSCAN(eps=0.3)
]

best_score = -1
best_model = None

# Проведение экспериментов
for model in models:
    model.fit(scaled_data)
    score = silhouette_score(scaled_data, model.labels_)
    print(f'Silhouette Score for {model.__class__.__name__}: {score}')

    if score > best_score:
        best_score = score
        best_model = model

print(f'Best Model: {best_model.__class__.__name__}, Score: {best_score}')