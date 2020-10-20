import numpy as np
from scipy.spatial import Delaunay

def get_full_face(verts, idx_eyes1, idx_eyes2, idx_mouth, idx_half, face_half):
    verts_half = verts[idx_half.astype(np.int)]

    # add eyes
    verts_eyes1 = verts[idx_eyes1]
    tris_eyes1 = Delaunay(verts_eyes1[:, :2])
    tris_eyes1 = tris_eyes1.simplices
    verts_eyes2 = verts[idx_eyes2]
    tris_eyes2 = Delaunay(verts_eyes2[:, :2])
    tris_eyes2 = tris_eyes2.simplices
    verts_ebs = np.concatenate((verts_eyes1, verts_eyes2))
    tris_ebs = np.concatenate((tris_eyes1, tris_eyes2 + len(verts_eyes1)))
    # add mouth
    verts_mouth = verts[idx_mouth]
    mouth_center = verts_mouth.mean(0)
    list_mouth_append = [verts_mouth]
    for _r, _c in zip([0.5, 0.2], [1, 0.9]):
        mouth_center_in = mouth_center.copy()
        mouth_center_in[2] = verts_mouth[:, 2].min() * _c
        verts_mouth2 = verts_mouth * _r
        mc = verts_mouth2.mean(0)
        verts_mouth2 = verts_mouth2 - (mc - mouth_center_in)
        list_mouth_append.append(verts_mouth2)
    mc = verts_mouth2.mean(0)
    mc_z = verts_mouth2[:, 2].min()[..., None]
    mc[2] = mc_z
    list_mouth_append.append(mc[None, ...])
    verts_mouth = np.concatenate(list_mouth_append)
    tris_mouth = Delaunay(verts_mouth[:, :2])
    tris_mouth = tris_mouth.simplices

    face_half = np.concatenate((face_half, tris_ebs + len(verts_half), tris_mouth + len(verts_half) + len(verts_ebs)), 0)
    verts_half = np.concatenate((verts_half, verts_ebs, verts_mouth), 0)
    face_half = face_half.astype(np.int)
    return verts_half, face_half
