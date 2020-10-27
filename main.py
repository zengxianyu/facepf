from PIL import Image
import numpy as np
from scipy.spatial import Delaunay
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
pi = 3.141592653

nexp = 1
nid = 140
path_img = f"../ssddata/FaceWarehouse_Data_0_raw/raw/Tester_{nid+1}/TrainingPose/pose_{nexp}.png"
path_verts = f"../ssddata/norm_data/fwh/mesh_raw_bs/Tester_{nid+1}.npy"
path_pose = f"../ssddata/norm_data/fwh/pose/Tester_{nid+1}.npz"

vx = 30
vy = 40
vz = 10

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

R_delta = angle2matrix([vx,vy,vz])
verts_image_rot = verts_image@R_delta

verts_n_rot = verts_image_rot.copy()
verts_n_rot[:, 1] = h-verts_n_rot[:, 1]
result = render_texture(verts_n_rot, tris_image, texture_img, texture_corrdinate, texture_triangle, h, w,
                    mapping_type="bilinear")

# save
Image.fromarray(result.astype(np.uint8)).save("debug.png")
#np.savez(rpath_name_aug + ".npz", (R_target, f_target, T_target, alpha))
