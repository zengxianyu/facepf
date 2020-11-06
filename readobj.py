import numpy as np

def read_obj(obj_name):
    vert = open(obj_name, "r").readlines()
    vert = list(map(lambda x:x.strip("\n"), vert))
    vert = list(filter(lambda x: x.startswith("v "), vert))
    vert = list(map(lambda x: list(map(lambda y:float(y),x.split()[1:])), vert))
    vert = np.array(vert)
    vert = vert.transpose((1,0)) # 3ddfa vertices[xyz,i]
    return vert
