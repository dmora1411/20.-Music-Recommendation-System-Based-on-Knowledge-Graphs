#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
pd.set_option('display.max_columns', None)


# In[89]:


df = pd.read_csv('dataset.csv', index_col=0)


# In[4]:


genre_map = df.groupby('track_id')['track_genre'].apply(set).to_dict()

# Удаляем дубликаты треков
df_unique = df.drop_duplicates(subset='track_id')

# Подсчёт уникальных сущностей
n_tracks = df_unique['track_id'].nunique()
n_artists = len(set(artist.strip() for artists in df_unique['artists'] for artist in artists.split(';')))
n_genres = len(set(genre for genres in genre_map.values() for genre in genres))
n_albums = df_unique['album_name'].nunique()



# In[5]:


print("Уникальных треков:", n_tracks)
print("Уникальных артистов:", n_artists)
print("Уникальных жанров:", n_genres)
print("Уникальных альбомов:", n_albums)


# In[6]:


import networkx as nx
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors


# Инициализация графа
G = nx.Graph()

# Строим базовый knowledge graph 
for _, row in df_unique.iterrows():
    track_id = f"track:{row['track_id']}"
    track_name = row['track_name']
    album = row['album_name']
    artists = [a.strip() for a in row['artists'].split(';') if a.strip()]
    genres = genre_map.get(row['track_id'], [])

    # Узел трека
    G.add_node(track_id,
               type="track",
               name=track_name,
               album=album,
               popularity=row['popularity'],
               duration=row['duration_ms'],
               explicit=row['explicit'],
               danceability=row['danceability'],
               energy=row['energy'],
               key=row['key'],
               loudness=row['loudness'],
               mode=row['mode'],
               speechiness=row['speechiness'],
               acousticness=row['acousticness'],
               instrumentalness=row['instrumentalness'],
               liveness=row['liveness'],
               valence=row['valence'],
               tempo=row['tempo'],
               time_signature=row['time_signature'])

    # Артисты и связи
    for artist in artists:
        artist_node = f"artist:{artist}"
        G.add_node(artist_node, type="artist", name=artist)
        G.add_edge(track_id, artist_node, relation="performed_by")

    # Коллаборации
    if len(artists) > 1:
        for a1, a2 in combinations(artists, 2):
            G.add_edge(f"artist:{a1}", f"artist:{a2}", relation="collaborated_on")

    # Жанры
    for genre in genres:
        genre_node = f"genre:{genre}"
        G.add_node(genre_node, type="genre", name=genre)
        G.add_edge(track_id, genre_node, relation="has_genre")

    # Альбом
    album_node = f"album:{album}"
    G.add_node(album_node, type="album", name=album)
    G.add_edge(track_id, album_node, relation="in_album")

# Добавляем связи между похожими треками 
feature_cols = [
    'danceability', 'energy', 'key', 'loudness', 'mode',
    'speechiness', 'acousticness', 'instrumentalness',
    'liveness', 'valence', 'tempo'
]

track_features = df_unique[feature_cols].values
track_ids = df_unique['track_id'].tolist()

# Нормализация признаков
scaler = StandardScaler()
X_scaled = scaler.fit_transform(track_features)

# Поиск ближайших соседей
k = 3
nn = NearestNeighbors(n_neighbors=k+1, metric='cosine')
nn.fit(X_scaled)
distances, indices = nn.kneighbors(X_scaled)

# Добавление рёбер track-track
for i, neighbors in enumerate(indices):
    source_id = f"track:{track_ids[i]}"
    for j in neighbors[1:]:
        target_id = f"track:{track_ids[j]}"
        if not G.has_edge(source_id, target_id):
            G.add_edge(source_id, target_id, relation="similar_track")


# In[20]:


G.remove_edges_from(nx.selfloop_edges(G))


# In[8]:


nx.write_graphml(G, "music_graph.graphml")


# In[21]:


import matplotlib.pyplot as plt
import random

# Выбираем подграф случайных 100 треков + соседей
track_nodes = [n for n, attr in G.nodes(data=True) if attr.get('type') == 'track']
sample_tracks = random.sample(track_nodes, 100)

# Собираем все соседние узлы
subgraph_nodes = set(sample_tracks)
for node in sample_tracks:
    subgraph_nodes.update(G.neighbors(node))

subgraph = G.subgraph(subgraph_nodes)

# Цвет узлов по типу
type_colors = {
    'track': 'skyblue',
    'artist': 'lightgreen',
    'genre': 'lightcoral',
    'album': 'plum'
}
node_colors = [
    type_colors.get(G.nodes[n].get('type', ''), 'gray') for n in subgraph.nodes()
]

plt.figure(figsize=(16, 12))
pos = nx.spring_layout(subgraph, seed=42, k=0.4)
nx.draw(subgraph, pos, node_size=30, node_color=node_colors, edge_color='gray', alpha=0.7, with_labels=False)
# Добавим легенду вручную
from matplotlib.patches import Patch
legend_elements = [Patch(color=color, label=label) for label, color in type_colors.items()]
plt.legend(handles=legend_elements, loc='upper right', fontsize='large')
plt.title("Subgraph Visualization (100 треков и связи)")
plt.axis('off')
plt.show()


# In[23]:


import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

# --- Общая структура графа ---
print("Число узлов:", G.number_of_nodes())
print("Число рёбер:", G.number_of_edges())

# --- Типы узлов ---
node_types = Counter(nx.get_node_attributes(G, 'type').values())
print("\nТипы узлов:")
for t, count in node_types.items():
    print(f"{t}: {count}")

# --- Плотность ---
density = nx.density(G)
print("\nПлотность графа:", round(density, 6))

# --- Связные компоненты ---
n_components = nx.number_connected_components(G)
largest_cc = max(nx.connected_components(G), key=len)
G_lcc = G.subgraph(largest_cc)

print("\nСвязных компонент:", n_components)
print("Размер крупнейшей компоненты:", len(largest_cc))


# --- Средняя степень ---
avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
print("Средняя степень узла:", round(avg_degree, 3))

# --- Распределение степени ---
degree_sequence = [d for n, d in G.degree()]
degree_counts = Counter(degree_sequence)

# --- Центральность (degree centrality) ---
#degree_centrality = nx.degree_centrality(G)
#top_deg = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
#print("\nТоп-10 узлов по degree centrality:")
#for node, centrality in top_deg:
#    print(f"{node}: {round(centrality, 5)}")

# --- График распределения степени ---
plt.figure(figsize=(8, 5))
plt.hist(degree_sequence, bins=50, color='skyblue', edgecolor='black')
plt.title("Распределение степеней узлов")
plt.xlabel("Степень")
plt.ylabel("Количество узлов")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# In[24]:


#Центральность (degree centrality)
degree_centrality = nx.degree_centrality(G)
top_deg = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
print("Топ-10 узлов по degree centrality:")
for node, centrality in top_deg:
    print(f"{node}: {round(centrality, 5)}")


# In[36]:


pagerank_scores = nx.pagerank(G, alpha=0.85)


# In[67]:


def get_recommendations_ppr(sample_id, df_unique, X_scaled, track_ids, pagerank_global, genre_map, G, top_n=10):
    import pandas as pd
    import networkx as nx
    from sklearn.metrics.pairwise import cosine_similarity

    sample_node = f"track:{sample_id}"
    if sample_node not in G:
        raise ValueError("Трек отсутствует в графе.")

    sample_idx = track_ids.index(sample_id)
    sample_vector = X_scaled[sample_idx].reshape(1, -1)

    # Косинусная схожесть
    acoustic_sims = cosine_similarity(sample_vector, X_scaled).flatten()
    pagerank_max = max(pagerank_global.values())

    # Personalized PageRank
    ppr_scores = nx.pagerank(G, personalization={sample_node: 1})

    results = []
    for i, row in enumerate(df_unique.itertuples(index=False)):
        candidate_id = row.track_id
        if candidate_id == sample_id:
            continue

        candidate_node = f"track:{candidate_id}"
        if candidate_node not in G:
            continue

        # Структурная схожесть
        structural_score = ppr_scores.get(candidate_node, 0)

        # Косинусная схожесть
        acoustic_sim = acoustic_sims[i]

        # Глобальный PageRank
        global_pagerank = pagerank_global.get(candidate_node, 0) / pagerank_max

        genre_set = genre_map.get(candidate_id, {'Unknown'})
        main_genre = next(iter(genre_set))

        results.append({
            'Track Name': row.track_name,
            'Artist': row.artists.split(';')[0],
            'Album': row.album_name,
            'Genre': main_genre,
            'PPR': round(structural_score, 6),
            'Acoustic': round(acoustic_sim, 4),
            'PageRank': round(global_pagerank, 6),
            'track_id': candidate_id
        })

    # Сортировка: сначала по структурной (PPR), потом акустика, потом глобальный PR
    df_result = pd.DataFrame(results).sort_values(
        by=['PPR', 'Acoustic', 'PageRank'], ascending=False
    )

    # Убираем повторения по артисту, альбому, жанру
    seen_artists, seen_albums, seen_genres = set(), set(), set()
    final_recs = []
    for _, row in df_result.iterrows():
        if row['Artist'] in seen_artists:
            continue
        if row['Album'] in seen_albums:
            continue
        if row['Genre'] in seen_genres:
            continue

        final_recs.append(row)
        seen_artists.add(row['Artist'])
        seen_albums.add(row['Album'])
        seen_genres.add(row['Genre'])

        if len(final_recs) >= top_n:
            break

    return pd.DataFrame(final_recs)


# In[78]:


import random

# Выбор случайного трека 
sample_idx = random.randint(0, len(df_unique) - 1)
sample_row = df_unique.iloc[sample_idx]
sample_id = sample_row['track_id']

sample_info = {
    'name': sample_row['track_name'],
    'artist': sample_row['artists'].split(';')[0],
    'genre': next(iter(genre_map.get(sample_id, {'Unknown'})))
}

#  Генерация рекомендаций 
recs_df = get_recommendations_ppr(
    sample_id=sample_id,
    df_unique=df_unique,
    X_scaled=X_scaled,
    track_ids=track_ids,
    pagerank_global=pagerank,
    genre_map=genre_map,
    G=G
)

# вывод 
def show_recommendations_table(sample_info, recs_df):
    from IPython.display import display, Markdown

    print("🎵 **Запросный трек**:")
    display(Markdown(
        f"**{sample_info['name']}** — {sample_info['artist']}  \nЖанр: `{sample_info['genre']}`\n"
    ))

    print("🎧 **Рекомендации:**")
    display(recs_df[['Track Name', 'Artist', 'Genre', 'PPR', 'Acoustic', 'PageRank']].style
            .background_gradient(subset=['PPR', 'Acoustic', 'PageRank'], cmap='Greens')
            .format(precision=4)
            .set_table_styles([{'selector': 'th', 'props': [('font-size', '12pt')]}]))



show_recommendations_table(sample_info, recs_df)


# In[85]:


import os
import os

def visualize_graph_recommendations(G, sample_id, sample_title_map, recs_df, max_recs=5, save_dir="graph_outputs"):
    import matplotlib.pyplot as plt
    import networkx as nx

    os.makedirs(save_dir, exist_ok=True)  

    sample_node = f"track:{sample_id}"
    sample_name = sample_title_map.get(sample_id, "Sample")

    for idx, row in recs_df.head(max_recs).iterrows():
        rec_id = row['track_id']
        rec_node = f"track:{rec_id}"
        rec_name = row['Track Name']

        # Соседи
        neighbors_sample = set(G.neighbors(sample_node))
        neighbors_rec = set(G.neighbors(rec_node))

        common_neighbors = neighbors_sample & neighbors_rec
        sample_unique = neighbors_sample - common_neighbors
        rec_unique = neighbors_rec - common_neighbors

        # Собираем узлы подграфа
        sub_nodes = {sample_node, rec_node} | common_neighbors | sample_unique | rec_unique
        sub_G = G.subgraph(sub_nodes)

        # Цвета узлов
        node_colors = []
        for node in sub_G.nodes:
            if node == sample_node:
                node_colors.append('blue')
            elif node == rec_node:
                node_colors.append('orange')
            elif node in common_neighbors:
                node_colors.append('green')
            else:
                node_colors.append('lightgray')

        # Подписи только для треков
        labels = {}
        for node in sub_G.nodes:
            if node.startswith('track:'):
                track_id = node.split(':')[1]
                labels[node] = sample_title_map.get(track_id, track_id)

        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(sub_G, seed=42)
        nx.draw(sub_G, pos, with_labels=False, node_color=node_colors,
                node_size=300, edge_color='gray')
        nx.draw_networkx_labels(sub_G, pos, labels=labels, font_size=8, font_color='black')
        plt.title(f"Связи: {sample_name} ↔ {rec_name}", fontsize=14)
        plt.axis('off')

        #Сохранение 
        safe_sample = sample_name.replace(" ", "_").replace("/", "_")
        safe_rec = rec_name.replace(" ", "_").replace("/", "_")
        filename = f"{safe_sample}_to_{safe_rec}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')

        plt.show()


# In[86]:


sample_title_map = {
    row['track_id']: row['track_name']
    for _, row in df_unique.iterrows()
}


# In[87]:


visualize_graph_recommendations(G, sample_id, sample_title_map, recs_df, max_recs=5)


# In[ ]:
