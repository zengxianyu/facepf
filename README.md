# facepf
A python implementation of the face profiling described in the 3DDFA paper

## Usage
1. Install packages: numpy, scipy, pillow
2. add a submodule from the face3d repo
```
git clone 
git submodule init
git submodule update
```
3. run the script (you may change the angles in line 38-40 to get the result after rotation of desired angles)
```
python main.py
```

## Result
Input inage:

![](https://raw.githubusercontent.com/zengxianyu/facepf/master/examples/HELEN_232194_1_0.jpg)

Output after inplane rotation:

![](https://raw.githubusercontent.com/zengxianyu/facepf/master/output.png)

## Reference
1. https://github.com/cleardusk/3DDFA/issues/182
2. https://github.com/YadiraF/face3d
