import torch
import pytorch3d.io
import numpy as np
import pdb
from PIL import Image
import os
import cv2
import pytorch3d.utils
import pytorch3d.transforms
from utils.facewarehouse import Facewarehouse
from utils.inference import write_obj
import sys
from scipy.spatial import Delaunay
from get_full_face import get_full_face
from image_meshing import image_meshing
import sys
sys.path.append("face3d")
from face3d.mesh.render import render_texture

path_data = "../3dface/FaceWarehouse_Data_0"

dset = Facewarehouse(path_data, gbr=False)

#mean_bs = dset.bs.mean(0)
#for i, _bs in enumerate(mean_bs):
#    write_obj("debug_bs/{}.obj".format(i), _bs.T, dset.triangles.T+1)
#pdb.set_trace()

# pre-defined index
idx_half = np.load("../3dface/FaceWarehouse_Data_0/index_half.npy")
tris_half = np.load("../3dface/FaceWarehouse_Data_0/tri_half.npy")
idx_bd = np.load("../3dface/FaceWarehouse_Data_0/index_half_bd.npy")

# mouth and eyes
with open("idx_eyes.txt", "r") as f:
    idx_eyes = f.readlines()
idx_eyes = list(map(lambda x: int(x.strip("\n")), idx_eyes))
idx_eyes1 = np.array(idx_eyes, dtype=np.int)
with open("idx_eyes2.txt", "r") as f:
    idx_eyes = f.readlines()
idx_eyes = list(map(lambda x: int(x.strip("\n")), idx_eyes))
idx_eyes2 = np.array(idx_eyes, dtype=np.int)
with open("idx_mouth.txt", "r") as f:
    idx_mouth = f.readlines()
idx_mouth = list(map(lambda x: int(x.strip("\n")), idx_mouth))
idx_mouth = np.array(idx_mouth, dtype=np.int)
idx_nosetip = dset.fw73[64]

start = int(sys.argv[1])
stop = int(sys.argv[2])


pi = 3.141592653
path_output = "../ssddata/FaceWarehouse_Data_0_Aug_Flip_IM"
if not os.path.exists(path_output):
    os.mkdir(path_output)

for n_id in range(start, stop):
    path_output_id = "{}/Tester_{}".format(path_output, n_id+1)
    if not os.path.exists(path_output_id):
        os.mkdir(path_output_id)
    path_output_id = "{}/Tester_{}/TrainingPose".format(path_output, n_id+1)
    if not os.path.exists(path_output_id):
        os.mkdir(path_output_id)
    bs = dset.bs[n_id]
    for n_exp in range(20):
        print("id {}, exp {}".format(n_id, n_exp))
        rpath_name = dset.id_pose_list[n_id][n_exp]
        img = dset.get_image(rpath_name)

        R_bs, f_bs, T_bs, alpha = np.load("../3dface/FaceWarehouse_Data_0/pose_alpha_matrix/{}.npz"\
                                 .format(rpath_name), allow_pickle=True)['arr_0']
        R_raw, f_raw, T_raw = np.load("../3dface/FaceWarehouse_Data_0_raw/pose/{}.npz".format(rpath_name),
                          allow_pickle=True)['arr_0']

        h, w, c = img.shape
        verts, _, _ = pytorch3d.io.load_obj("../3dface/FaceWarehouse_Data_0_raw/raw/{}.obj".format(rpath_name))
        verts = verts.numpy()
        verts_face, tris_face = get_full_face(verts, idx_eyes1, idx_eyes2, idx_mouth, idx_half, tris_half)
        verts_bkg, tris_bkg = image_meshing(verts, R_raw, f_raw, T_raw, idx_bd, idx_nosetip, h, w)
        verts_image = np.concatenate((verts_face@R_raw*f_raw+T_raw, verts_bkg), 0)
        tris_image = np.concatenate((tris_face, tris_bkg+len(verts_face)))


        verts_n = verts_image.copy()
        verts_n[:, 1] = h-verts_n[:, 1]

        texture_corrdinate = verts_n.copy()
        texture_img = img.copy().astype(np.float32)
        texture_triangle = tris_image.copy()

        for n_aug_x  in [0, 2, 3, 4, 5, 6, 7]:
            for n_aug_y in [0, 2, 3, 4, 5, 6, 7, 8, 9]:
                rpath_name_aug = "{}/{}_{}_{}".format(path_output, rpath_name, n_aug_x, n_aug_y)
                if n_aug_y == 0 and n_aug_x == 0:
                    Image.fromarray(img.astype(np.uint8)).save(rpath_name_aug + ".png")
                    np.savez(rpath_name_aug + ".npz", (R_bs, f_bs, T_bs, alpha))
                    continue
                v_aug_y = n_aug_y//2
                sign_y = n_aug_y%2
                if sign_y:
                    v_aug_y *= -1
                v_aug_x = n_aug_x//2
                sign_x = n_aug_x%2
                if sign_x:
                    v_aug_x *= -1
                R_delta = pytorch3d.transforms.euler_angles_to_matrix(torch.Tensor([v_aug_x*10/180.0*pi,
                                                                                    v_aug_y*10/180.0*pi, 0]), "XYZ")
                R_delta = R_delta.numpy()
                verts_image_rot = verts_image@R_delta
                #verts_image_rot[-n_vbi:] = verts_bd_image
                verts_n_rot = verts_image_rot.copy()
                verts_n_rot[:, 1] = h-verts_n_rot[:, 1]
                result = render_texture(verts_n_rot, tris_image, texture_img, texture_corrdinate, texture_triangle, h, w,
                                    mapping_type="bilinear")

                # save
                R_target = R_bs@R_delta
                T_target = T_bs@R_delta
                f_target = f_bs
                Image.fromarray(result.astype(np.uint8)).save(rpath_name_aug + ".png")
                np.savez(rpath_name_aug + ".npz", (R_target, f_target, T_target, alpha))
