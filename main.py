from PIL import Image
import pdb
import numpy as np
from scipy.spatial import Delaunay
from get_full_face import get_full_face
from image_meshing import image_meshing
from readobj import read_obj
from face3d.mesh_numpy.transform import angle2matrix
from face3d.mesh.render import render_texture


# pre-defined values
path_pre = "./annotations"
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

path_img = "./examples/HELEN_232194_1_0.jpg"
path_verts = "./examples/HELEN_232194_1_0.obj"
path_pose = "./examples/pose.npz"

# angles in degree
vx = 0
vy = -30
vz = 0

img = Image.open(path_img)
img = np.array(img)
h,w,c = img.shape
pose = np.load(path_pose)
R_raw, T_raw, s_raw = pose['R'], pose['T'], pose['s']
verts = read_obj(path_verts)
verts = verts.T.copy()
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
Image.fromarray(result.astype(np.uint8)).save("output.png")
