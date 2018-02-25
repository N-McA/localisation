
from pathlib import Path
import numpy as np
import sqlite3

# Get the colmap scripts
import sys
sys.path.append('/home/nm583/colmap/scripts/python')
from read_model import rotmat2qvec


for seq_n in range(11):
    print(seq_n)
    data_loc = Path('/home/nm583/data/kitti/dataset/')
    seq_n_string = str(seq_n).zfill(2)
    sequence_loc = data_loc / 'sequences' / seq_n_string
    db_loc = sequence_loc / 'colmap/database.db'

    poses = []
    with (data_loc / 'poses/{}.txt'.format(seq_n_string)).open() as f:
        for line in f:
            m = np.fromstring(line, dtype=float, sep=' ')
            m = m.reshape(3, 4)
            m = np.vstack([m, [0, 0, 0, 1]])
            poses.append(m)
    poses = np.array(poses)

    t_vecs = poses[:, :3, 3]
    r_mats = poses[:, :3, :3]
    q_vecs = []
    for r_mat in r_mats:
        q_vecs.append(rotmat2qvec(r_mat))
    q_vecs = np.array(q_vecs)

    image_name_to_q_vec = {}
    image_name_to_t_vec = {}
    for i in range(0, len(q_vecs)):
        image_name = '{}.png'.format(str(i).zfill(6))
        image_name_to_t_vec[image_name] = t_vecs[i]
        image_name_to_q_vec[image_name] = q_vecs[i]

    conn = sqlite3.connect(str(db_loc))

    with conn:
        rows = list(conn.execute("SELECT * FROM images;"))

    for row in rows:
        image_id, image_name = row[:2]
        t_vec = image_name_to_t_vec[image_name]
        q_vec = image_name_to_q_vec[image_name]
        with conn:
            conn.execute('''
                UPDATE images
                SET
                    camera_id=1,
                    prior_tx=?,
                    prior_ty=?,
                    prior_tz=?,
                    prior_qw=?,
                    prior_qx=?,
                    prior_qy=?,
                    prior_qz=?
                WHERE
                    image_id=?;
            ''', (*t_vec, *q_vec, image_id))
