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


# In[135]:


import networkx as nx
from itertools import combinations

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


# In[8]:


nx.write_graphml(G, "music_graph.graphml")


# In[136]:


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
plt.title("Subgraph Visualization (100 tracks and its neighbors)")
plt.axis('off')
plt.show()


# In[137]:


import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter

#  Общая структура графа 
print("Число узлов:", G.number_of_nodes())
print("Число рёбер:", G.number_of_edges())

# Типы узлов 
node_types = Counter(nx.get_node_attributes(G, 'type').values())
print("\nТипы узлов:")
for t, count in node_types.items():
    print(f"{t}: {count}")

# Плотность 
density = nx.density(G)
print("\nПлотность графа:", round(density, 6))

# Связные компоненты 
n_components = nx.number_connected_components(G)
largest_cc = max(nx.connected_components(G), key=len)
G_lcc = G.subgraph(largest_cc)

print("\nСвязных компонент:", n_components)
print("Размер крупнейшей компоненты:", len(largest_cc))


#  Средняя степень 
avg_degree = sum(dict(G.degree()).values()) / G.number_of_nodes()
print("Средняя степень узла:", round(avg_degree, 3))

#  Распределение степени 
degree_sequence = [d for n, d in G.degree()]
degree_counts = Counter(degree_sequence)


# График распределения степени 
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


# In[156]:


import pandas as pd
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations_ppr(sample_id, df_unique, X_scaled, track_ids, pagerank_global, genre_map, G, top_n=10):
    sample_node = f"track:{sample_id}"
    if sample_node not in G:
        raise ValueError("Трек отсутствует в графе.")

    sample_idx = track_ids.index(sample_id)
    sample_vector = X_scaled[sample_idx].reshape(1, -1)

    acoustic_sims = cosine_similarity(sample_vector, X_scaled).flatten()
    pagerank_max = max(pagerank_global.values())

    ppr_scores = nx.pagerank(G, personalization={sample_node: 1})

    results = []
    for i, row in enumerate(df_unique.itertuples(index=False)):
        candidate_id = row.track_id
        if candidate_id == sample_id:
            continue

        candidate_node = f"track:{candidate_id}"
        if candidate_node not in G:
            continue

        ppr_score = ppr_scores.get(candidate_node, 0)
        if ppr_score == 0:
            continue

        acoustic_sim = acoustic_sims[i]
        global_pr = pagerank_global.get(candidate_node, 0) / pagerank_max
        main_genre = next(iter(genre_map.get(candidate_id, {'Unknown'})))

        results.append({
            'Track Name': row.track_name,
            'Artist': row.artists.split(';')[0],
            'Album': row.album_name,
            'Genre': main_genre,
            'PPR': round(ppr_score, 6),
            'Acoustic': round(acoustic_sim, 4),
            'PageRank': round(global_pr, 6),
            'track_id': candidate_id
        })

    # 1. Сортировка по PPR (по убыванию)
    df_sorted = pd.DataFrame(results).sort_values(by='PPR', ascending=False)

    # 2. Удаление дублей по артисту и альбому (приоритет оставлять сверху)
    seen_artists, seen_albums = set(), set()
    deduped = []
    for _, row in df_sorted.iterrows():
        if row['Artist'] in seen_artists or row['Album'] in seen_albums:
            continue
        deduped.append(row)
        seen_artists.add(row['Artist'])
        seen_albums.add(row['Album'])

    # 3. Ограничение top-N после удаления дублей
    top_ppr_filtered = pd.DataFrame(deduped).head(top_n)

    # 4. Финальная сортировка по Acoustic и PageRank
    final_sorted = top_ppr_filtered.sort_values(
        by=['Acoustic', 'PageRank'], ascending=False
    )

    return final_sorted.reset_index(drop=True)


# In[157]:


df_unique[df_unique['artists'].str.contains("AC/DC", case=False, na=False)].head()


# In[158]:


sample_id = "08mG3Y1vljYA6bvDt4Wqkj"


# Получаем информацию о выбранном треке
sample_row = df_unique[df_unique['track_id'] == sample_id].iloc[0]

sample_info = {
    'name': sample_row['track_name'],
    'artist': sample_row['artists'].split(';')[0],
    'genre': next(iter(genre_map.get(sample_id, {'Unknown'})))
}

# Генерация рекомендаций
recs_df = get_recommendations_ppr(
    sample_id=sample_id,
    df_unique=df_unique,
    X_scaled=X_scaled,
    track_ids=track_ids,
    pagerank_global=pagerank,
    genre_map=genre_map,
    G=G
)

# Вывод таблицы
def show_recommendations_table(sample_info, recs_df):
    from IPython.display import display, Markdown

    print("Selected track input:")
    display(Markdown(
        f"**{sample_info['name']}** — {sample_info['artist']}  \nЖанр: `{sample_info['genre']}`\n"
    ))

    print("Recommendations:")
    display(recs_df[['Track Name', 'Artist', 'Genre', 'PPR', 'Acoustic', 'PageRank']].style
            .background_gradient(subset=['PPR', 'Acoustic', 'PageRank'], cmap='Greens')
            .format(precision=4)
            .set_table_styles([{'selector': 'th', 'props': [('font-size', '12pt')]}]))

# Показываем результат
show_recommendations_table(sample_info, recs_df)


# In[159]:


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
        plt.title(f"Relations: {sample_name} ↔ {rec_name}", fontsize=14)
        plt.axis('off')

        #Сохранение 
        safe_sample = sample_name.replace(" ", "_").replace("/", "_")
        safe_rec = rec_name.replace(" ", "_").replace("/", "_")
        filename = f"{safe_sample}_to_{safe_rec}.png"
        filepath = os.path.join(save_dir, filename)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')

        plt.show()

sample_title_map = {
    row['track_id']: row['track_name']
    for _, row in df_unique.iterrows()
}

visualize_graph_recommendations(G, sample_id, sample_title_map, recs_df, max_recs=5)


