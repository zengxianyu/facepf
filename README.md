# facepf
A python implementation of the face profiling described in the 3DDFA paper. 
The older version of this implementation can be found in the branch ``face3d". 
Then I migrate everything to FLAME[2] face model as I found it's easy to obtain and their code to fit 2d landmarks performs well. 

## Usage
1. Install required envs the same as in [2] or following [there](https://github.com/zengxianyu/photometric_optimization)
2. add [my folk of [2]](https://github.com/zengxianyu/photometric_optimization) as a submodule
```
git clone https://github.com/zengxianyu/facepf.git
cd facepf
git submodule init
git submodule update
```
3. run the script (you may change the angles in line 81 to get the result after rotation of desired angles)
```
cd photometric_optimization
python image_renderer.py
```

## How to get the npy file
Following [there](https://github.com/zengxianyu/photometric_optimization)

## Result
Input image and output after inplane rotation:

![](https://raw.githubusercontent.com/zengxianyu/facepf/master/result.jpg)

## Reference
[1] 3DDFA. https://github.com/cleardusk/3DDFA/issues/182

[2] FLAME photometric optimization. https://github.com/HavenFeng/photometric_optimization
