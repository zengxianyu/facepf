from PIL import Image
import numpy as np
import os
import pdb
from get_full_face import get_full_face
from image_meshing import image_meshing
from face3d.mesh_numpy.transform import angle2matrix
from face3d.mesh.render import render_texture


# pre-defined values
path_tris = "../ssddata/norm_data/fwh/tris_fwh.npy"
path_pre = "../ssddata/norm_data/fwh"

idx_half = np.load(f"{path_pre}/index_half.npy")
tris_half = np.load(f"{path_pre}/tri_half.npy")
idx_bd = np.load(f"{path_pre}/index_half_bd.npy")
## mouth and eyes
with open(f"{path_pre}/idx_eyes.txt", "r") as f:
    idx_eyes = f.readlines()
idx_eyes = list(map(lambda x: int(x.strip("\n")), idx_eyes))
idx_eyes1 = np.array(idx_eyes, dtype=np.int)
with open(f"{path_pre}/idx_eyes2.txt", "r") as f:
    idx_eyes = f.readlines()
idx_eyes = list(map(lambda x: int(x.strip("\n")), idx_eyes))
idx_eyes2 = np.array(idx_eyes, dtype=np.int)
with open(f"{path_pre}/idx_mouth.txt", "r") as f:
    idx_mouth = f.readlines()
idx_mouth = list(map(lambda x: int(x.strip("\n")), idx_mouth))
idx_mouth = np.array(idx_mouth, dtype=np.int)
idx_nosetip = 8972


delta = 5.0
def sign_mod(val):
    value = val//2
    sign = val%2
    if sign:
        value *= -1
    return value

path_out = "../ssddata/norm_data/fwh/images_aug_pose"
if not os.path.exists(path_out):
    os.mkdir(path_out)
#nexp = 1
#nid = 140
for nid in range(150):
    path_out_sub = f"{path_out}/Tester_{nid+1}"
    if not os.path.exists(path_out_sub):
        os.mkdir(path_out_sub)
    for nexp in range(20):
        path_out_sub = f"{path_out}/Tester_{nid+1}/pose_{nexp}"
        if not os.path.exists(path_out_sub):
            os.mkdir(path_out_sub)
        path_img = f"../ssddata/FaceWarehouse_Data_0_raw/raw/Tester_{nid+1}/TrainingPose/pose_{nexp}.png"
        path_verts = f"../ssddata/norm_data/fwh/mesh_raw_bs/Tester_{nid+1}.npy"
        path_pose = f"../ssddata/norm_data/fwh/pose/Tester_{nid+1}.npz"

        img = Image.open(path_img)
        img = np.array(img)
        h,w,c = img.shape
        pose = np.load(path_pose)
        R_raw, T_raw, s_raw = pose['R'], pose['T'], pose['s']
        verts = np.load(path_verts)
        verts = verts[nexp]
        R_raw = R_raw[nexp]
        T_raw = T_raw[nexp]
        s_raw = s_raw[nexp]
        verts_face, tris_face = get_full_face(verts, idx_eyes1, idx_eyes2, idx_mouth, idx_half, tris_half)
        verts_bkg, tris_bkg = image_meshing(verts, R_raw, s_raw, T_raw, idx_bd, idx_nosetip, h, w)
        verts_image = np.concatenate((verts_face@R_raw*s_raw+T_raw, verts_bkg), 0)
        tris_image = np.concatenate((tris_face, tris_bkg+len(verts_face)))

        verts_n = verts_image.copy()
        verts_n[:, 1] = h-verts_n[:, 1]

        texture_corrdinate = verts_n.copy()
        texture_img = img.copy().astype(np.float32)
        texture_triangle = tris_image.copy()
        for n_aug_y in [0]+list(range(2,20)):#, 2, 3, 4, 5, 6, 7, 8, 9]:
            for n_aug_x  in [0]+list(range(2,16)):# 2, 3, 4, 5, 6, 7]:
                vx = sign_mod(n_aug_x)*delta
                vy = sign_mod(n_aug_y)*delta

                R_delta = angle2matrix([vx,vy,0])
                verts_image_rot = verts_image@R_delta

                verts_n_rot = verts_image_rot.copy()
                verts_n_rot[:, 1] = h-verts_n_rot[:, 1]
                result = render_texture(verts_n_rot, tris_image, texture_img, texture_corrdinate, texture_triangle, h, w,
                                    mapping_type="bilinear")

                # save
                postfix = f"{n_aug_x}_{n_aug_y}"
                Image.fromarray(result.astype(np.uint8)).save(
                    f"{path_out_sub}/{postfix}.png")
                np.savez(f"{path_out_sub}/{postfix}.npz", 
                        vx=vx, vy=vy, vz=0,
                        R_delta=R_delta)
