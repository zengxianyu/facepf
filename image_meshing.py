import numpy as np
from scipy.spatial import Delaunay


def image_meshing(verts, R, f, T, idx_bd, idx_nosetip, h, w):
    verts_bd = verts[idx_bd].copy()
    list_bd_append = []
    for rate, tr in zip([1.2], [0.8]):  # np.linspace(1.2, 2, 3):
        verts_bd2 = verts_bd * rate
        verts_bd2[:, :2] -= verts[idx_nosetip][:2] * rate
        verts_bd2[:, 2] -= (verts[idx_nosetip][2] * (rate - tr))

        # verts_bd3 = verts_bd * 2
        # verts_bd3[:, :2] -= verts[id_noisetip][:2]
        # verts_bd3[:, 2] -= (verts[id_noisetip][2]*0.9)
        verts_bd2 = verts_bd2 @ R * f + T
        # verts_bd3 = verts_bd3@R*f+T
        list_bd_append.append(verts_bd2)
    verts_bd_append = np.concatenate(list_bd_append, 0)

    verts_bd = verts_bd @ R * f + T
    xmin, xmax = verts_bd[:, 0].min(), verts_bd[:, 0].max()
    ymin, ymax = verts_bd[:, 1].min(), verts_bd[:, 1].max()
    m_top_bd = verts_bd[:, 1] < (ymin + (ymax - ymin) / 4)
    m_btm_bd = verts_bd[:, 1] >= (ymax - (ymax - ymin) / 4)
    m_left_bd = (verts_bd[:, 0] < (xmin + (xmax - xmin) / 2)) & (~m_top_bd) & (~m_btm_bd)
    m_right_bd = (verts_bd[:, 0] >= (xmin + (xmax - xmin) / 2)) & (~m_top_bd) & (~m_btm_bd)
    verts_bd_top = verts_bd2[m_top_bd]
    verts_bd_btm = verts_bd2[m_btm_bd]
    verts_bd_left = verts_bd2[m_left_bd]
    verts_bd_right = verts_bd2[m_right_bd]
    n_top = m_top_bd.sum()
    n_btm = m_btm_bd.sum()
    n_left = m_left_bd.sum()
    n_right = m_right_bd.sum()
    verts_image_bd_top = np.stack([np.linspace(0, w, n_top), np.array([h] * n_top), verts_bd_top[:, 2]], 1)
    verts_image_bd_btm = np.stack([np.linspace(0, w, n_btm), np.array([0] * n_btm), verts_bd_btm[:, 2]], 1)
    verts_image_bd_left = np.stack([np.array([0] * n_left), np.linspace(0, h, n_left), verts_bd_left[:, 2]], 1)
    verts_image_bd_right = np.stack([np.array([w] * n_right), np.linspace(0, h, n_right), verts_bd_right[:, 2]], 1)
    verts_bd_image = np.concatenate((verts_image_bd_top, verts_image_bd_btm, verts_image_bd_left, verts_image_bd_right), 0)

    mm = (verts_bd_append[:, 0] <= w).astype(np.int)
    verts_bd_append[:, 0] = verts_bd_append[:, 0] * mm + w * (1 - mm)
    mm = (verts_bd_append[:, 1] <= h).astype(np.int)
    verts_bd_append[:, 1] = verts_bd_append[:, 1] * mm + h * (1 - mm)

    verts_bds = np.concatenate((verts_bd, verts_bd_append, verts_bd_image), 0)
    tri = Delaunay(verts_bds[:, :2])
    tris_bds = tri.simplices

    #verts_image = np.concatenate((verts_half, verts_bds), 0)
    #face_image = np.concatenate((face_half, tris_bds + len(verts_half)), 0)
    return verts_bds, tris_bds
