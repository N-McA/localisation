
import sqlite3
from collections import defaultdict
import pickle
from pathlib import Path


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % 2147483647
    image_id1 = (pair_id - image_id2) // 2147483647
    return image_id1, image_id2


def name_to_number(name):
    return int(name[:-len('.png')])


def extract_graph(db_loc):
    conn = sqlite3.connect(str(db_loc))
    #  print(db_loc)
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

    loop_closures = defaultdict(set)
    for a, b in matches:
        a_n = name_to_number(image_id_to_name[a])
        b_n = name_to_number(image_id_to_name[b])
        distance = abs(a_n - b_n)
        if distance > 150:
            loop_closures[a].add(b)
            loop_closures[b].add(a)

    return graph, loop_closures, image_id_to_name


def kvs_map(f, d):
    '''
    f: 'a -> 'b
    d: dictionary<'a, Set<'a>>
    returns: dictionary<'b, Set<'b>>
    '''
    r = {}
    for k, xs in d.items():
        r[f(k)] = set(f(x) for x in xs)
    return r


data_loc = Path('/local/sdd/nm583/kitti/dataset/')
total_graph = {}
total_loop_closure_graph = {}

train_seqs = range(0, 7)
test_seqs = range(7, 10)
train_paths = []
test_paths = []

for seq_n in range(11):
    seq_n_string = str(seq_n).zfill(2)
    sequence_loc = data_loc / 'sequences' / seq_n_string
    db_loc = sequence_loc / 'colmap/database.db'
    image_loc = sequence_loc / 'image_2'
    graph, loop_closures, image_id_to_name = extract_graph(db_loc)
    full_path = lambda x: image_loc / image_id_to_name[x]
    graph = kvs_map(full_path, graph)
    loop_closures = kvs_map(full_path, loop_closures)
    total_graph = {**graph, **total_graph}
    total_loop_closure_graph = {**loop_closures, **total_loop_closure_graph}
    if seq_n in train_seqs:
        train_paths += list(graph.keys())
    else:
        test_paths += list(graph.keys())
    print("{}: ({}, {}),".format(seq_n, len(graph), len(loop_closures)))

train_paths = sorted(train_paths)
test_paths = sorted(test_paths)

extracted_data_loc = Path('/local/sdd/nm583/extracted_kitti_data')

with (extracted_data_loc / 'graph.pickle').open('wb') as f:
    pickle.dump(total_graph, f)

with (extracted_data_loc / 'loop_closure_graph.pickle').open('wb') as f:
    pickle.dump(total_loop_closure_graph, f)

with (extracted_data_loc / 'train_paths.txt').open('w') as f:
    for path in train_paths:
        print(path, file=f)

with (extracted_data_loc / 'test_paths.txt').open('w') as f:
    for path in test_paths:
        print(path, file=f)
