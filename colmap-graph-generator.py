
import sqlite3
from collections import defaultdict
import pickle
from pathlib import Path


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) // 2147483647
    return image_id1, image_id2


def extract_graph(db_loc):
    conn = sqlite3.connect(str(db_loc))
    image_id_to_name = {}
    matches = []

    with conn:
        for idx, name in conn.execute("SELECT image_id, name FROM images;"):
            image_id_to_name[idx] = name

    with conn:
        for x in conn.execute("SELECT pair_id FROM inlier_matches;"):
            i, j = pair_id_to_image_ids(x[0])
            matches.append((i, j))

    graph = defaultdict(set)
    for i, j in matches:
        graph[i].add(j)
        graph[j].add(i)

    return graph, image_id_to_name


data_loc = Path('/home/nm583/data/kitti/dataset/')
total_graph = {}
for seq_n in [0, 3]:
    seq_n_string = str(seq_n).zfill(2)
    sequence_loc = data_loc / 'sequences' / seq_n_string
    db_loc = sequence_loc / 'colmap/database.db'
    image_loc = sequence_loc / 'image_2'
    full_path = lambda x: image_loc / x
    graph, image_id_to_name = extract_graph(db_loc)
    for image_id in image_id_to_name:
        neighs = set(full_path(image_id_to_name[neigh_id]) for neigh_id in graph[image_id])
        total_graph[full_path(image_id_to_name[image_id])] = neighs


with open('/local/sdd/nm583/graph.pickle', 'wb') as f:
    pickle.dump(total_graph, f)
