
from collections import defaultdict
import pickle
from pathlib import Path

import numpy as np


dubrovnik_bundle_file_loc = Path(
    '/home/nm583/sfm_data/Dubrovnik6K/bundle/bundle.db.out')

dubrovnik_root_loc = Path('/home/nm583/sfm_data/Dubrovnik6K')
dubrovnik_relation_loc = \
    Path('/home/nm583/sfm_data/dubrovnik_relation.pickle')

rome_bundle_file_loc = Path(
    '/home/nm583/sfm_data/Rome16K/bundle/bundle.db.out')
rome_root_loc = Path('/home/nm583/sfm_data/Rome16K')
rome_relation_loc = \
    Path('/home/nm583/sfm_data/rome_relation.pickle')


def get_cornell_paths(root_loc, loc):
    paths = []
    with (root_loc / loc).open() as f:
        for line in f:
            path = line.strip().split(' ')[0]
            path = root_loc / path
            paths.append(path)
    return np.array(paths)

def get_dubrovnik_query_paths():
    return get_cornell_paths(dubrovnik_root_loc, 'list.query.txt')

def get_dubrovnik_db_paths():
    return get_cornell_paths(dubrovnik_root_loc, 'list.db.txt')

def get_rome_query_paths():
    return get_cornell_paths(rome_root_loc, 'list.query.txt')

def get_rome_db_paths():
    return get_cornell_paths(rome_root_loc, 'list.db.txt')

def get_geo_locs(path_file_loc, bundle_file_loc):

    orig_paths = get_cornell_paths(path_file_loc)

    def parse_lines(lines):
        # http://www.cs.cornell.edu/~snavely/bundler/bundler-v0.3-manual.html#S1
        header = next(lines).strip()
        assert header == '# Bundle file v0.3'
        n_cs, n_ps = [int(x) for x in next(lines).split(' ')]
        assert n_cs == len(orig_paths)
        positions = []
        for _ in range(n_cs):
            focal_info = next(lines)
            rot_mat = []
            for _ in range(3):
                # ignore rotation
                rot_mat.append([float(x) for x in next(lines).strip().split()])
            rot_mat = np.array(list(reversed(rot_mat)))
            position = np.array(
                [float(x) for x in next(lines).strip().split(' ')])
            positions.append(position)
        for _ in range(3*n_ps):
            next(lines)
        return positions

    with bundle_file_loc.open() as bundle_file:
        positions = parse_lines(bundle_file)

    assert len(positions) == len(orig_paths)

    return {path: pos for path, pos in zip(orig_paths, positions)}


def get_dubrovnik_geo_locs():
    return get_geo_locs(
        Path('/home/nm583/sfm_data/Dubrovnik6K/bundle/list.orig.txt'),
        Path('/home/nm583/sfm_data/Dubrovnik6K/bundle/bundle.orig.out'),
    )  

def get_rome_geo_locs():
    return get_geo_locs(
        Path('/home/nm583/sfm_data/Rome16K/bundle/list.orig.txt'),
        Path('/home/nm583/sfm_data/Rome16K/bundle/bundle.orig.out'),
    ) 

def save_relation(root_loc, bundle_file_loc, output_loc):

    camera_to_point = defaultdict(set)
    point_to_camera = defaultdict(set)

    def parse_view_list(lines, point_idx):
        line = next(lines)
        xs = line.strip().split(' ')
        n_quads = int(xs[0])
        quads = xs[1:]
        assert len(quads) / 4 == n_quads
        for i in range(0, len(quads), 4):
            camera_idx = int(quads[i])
            camera_to_point[camera_idx].add(point_idx)
            point_to_camera[point_idx].add(camera_idx)

    def parse_point(lines, point_idx):
        pos = next(lines)
        color = next(lines)
        return pos, color, parse_view_list(lines, point_idx)

    def parse_lines(lines):
        # http://www.cs.cornell.edu/~snavely/bundler/bundler-v0.3-manual.html#S1
        header = next(lines).strip()
        assert header == '# Bundle file v0.3'
        n_cs, n_ps = [int(x) for x in next(lines).split(' ')]
        n_lines_per_camera = 1 + 3 + 1
        for _ in range(n_cs * n_lines_per_camera):
                # skip the cameras
            next(lines)
        for i in range(n_ps):
            parse_point(lines, i)

    with bundle_file_loc.open() as bundle_file:
        parse_lines(bundle_file)

    list_loc = root_loc / 'list.db.txt'
    image_locs = []
    with list_loc.open() as f:
        for line in f:
            image_locs.append(root_loc / line.strip().split(' ')[0])

    relation = defaultdict(set)
    for camera_idx in range(len(camera_to_point)):
        image_0_path = image_locs[camera_idx]
        pts = camera_to_point[camera_idx]
        for pt in pts:
            for adjacent_cam_idx in point_to_camera[pt]:
                image_1_path = image_locs[adjacent_cam_idx]
                relation[image_0_path].add(image_1_path)
                relation[image_1_path].add(image_0_path)

    with output_loc.open('wb') as f:
        pickle.dump(relation, f)


def get_dubrovnik_relation():
    with dubrovnik_relation_loc.open('rb') as f:
        relation = pickle.load(f)
    return relation

def get_rome_relation():
    if not rome_relation_loc.exists():
        save_relation(rome_root_loc, rome_bundle_file_loc, rome_relation_loc)
    with rome_relation_loc.open('rb') as f:
        relation = pickle.load(f)
    return relation