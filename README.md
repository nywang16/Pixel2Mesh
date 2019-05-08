# Pixel2Mesh
This repository contains the TensorFlow implementation for the following paper</br>

[Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images (ECCV2018)](http://openaccess.thecvf.com/content_ECCV_2018/papers/Nanyang_Wang_Pixel2Mesh_Generating_3D_ECCV_2018_paper.pdf)</br>

Nanyang Wang*, [Yinda Zhang](http://robots.princeton.edu/people/yindaz/)\*, [Zhuwen Li](http://www.lizhuwen.com/)\*, [Yanwei Fu](http://yanweifu.github.io/), [Wei Liu](http://www.ee.columbia.edu/~wliu/), [Yu-Gang Jiang](http://www.yugangjiang.info/).

The code is based on the [gcn](https://github.com/tkipf/gcn) framework.

#### Citation
If you use this code for your research, please consider citing:

    @inProceedings{wang2018pixel2mesh,
      title={Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images},
      author={Nanyang Wang and Yinda Zhang and Zhuwen Li and Yanwei Fu and Wei Liu and Yu-Gang Jiang},
      booktitle={ECCV},
      year={2018}
    }

# Project Page
The project page is available at http://bigvid.fudan.edu.cn/pixel2mesh

# Dependencies
Requirements:
* Python2.7+ with Numpy and scikit-image
* [Tensorflow (version 1.0+)](https://www.tensorflow.org/install/)
* [TFLearn](http://tflearn.org/installation/)

Our code has been tested with Python 2.7, **TensorFlow 1.3.0**, TFLearn 0.3.2, CUDA 8.0 on Ubuntu 14.04.

# Running the demo
```
git clone https://github.com/nywang16/Pixel2Mesh.git
cd Data/
```
Download the pre-trained model and unzip to the `Data/` folder.
* https://drive.google.com/file/d/1gD-dk-XrAa5mfrgdZSunjaS6pUUWsZgU/view?usp=sharing
```
unzip checkpoint.zip
```

#### Reconstructing shapes
    python demo.py --image Data/examples/plane.png
Run the demo code and the output mesh file is saved in `Data/examples/plane.obj` 

#### Input image, output mesh
<img src="./Docs/images/plane.png" width = "330px" /><img src="./Docs/images/plane.gif" />

# Installation

If you use CD and EMD for training or evaluation, we have included the cuda implementations of [Fan et. al.](https://github.com/fanhqme/PointSetGeneration) in external/

    cd Pixel2Mesh/external/

    Modify the first 3 lines of the makefile to point to your nvcc, cudalib and tensorflow library.

    make


# Dataset

We used the [ShapeNet](https://www.shapenet.org) dataset for 3D models, and rendered views from [3D-R2N2](https://github.com/chrischoy/3D-R2N2):</br>
When using the provided data make sure to respect the shapenet [license](https://shapenet.org/terms).

Below is the complete set of training data. Download it into the `Data/` folder.

https://drive.google.com/open?id=131dH36qXCabym1JjSmEpSQZg4dmZVQid </br>


The training/testing split can be found in `Data/train_list.txt` and `Data/test_list.txt` </br>
    
Each .dat file in the provided data contain: </br>
* The sampled point cloud (with vertex normal) from ShapeNet. We transformed it to corresponding coordinates in camera coordinate based on camera parameters from the Rendering Dataset.

**Input image, ground truth point cloud.**</br>
<img src="./Docs/images/car_example.png" width = "350px" />
![label](./Docs/images/car_example.gif)

# Training
    python train.py
You can change the training data, learning rate and other parameters by editing `train.py`

The total number of training epoch is 30; the learning rate is initialized as 3e-5 and drops to 1e-5 after 25 epochs.

# Evaluation
The evaluation code was released, please refer to `eval_testset.py` for more details.

Notice that the 3D shape are downscaled by a factor of 0.57 to generate rendering. As result, all the numbers shown in experiments used 0.57xRaw Shape for evaluation. This scale may be related to the render proccess, we used the rendering data from 3DR2N2 paper, and this scale was there since then for reason that we don't know.

# Statement
This software is for research purpose only. </br>
Please contact us for the licence of commercial purposes. All rights are preserved.

# Contact
Nanyang Wang (nywang16 AT fudan.edu.cn)

Yinda Zhang (yindaz AT cs.princeton.edu)

Zhuwen Li (lzhuwen AT gmail.com)

Yanwei Fu (yanweifu AT fudan.edu.cn)

Yu-Gang Jiang (ygj AT fudan.edu.cn)

# License
Apache License version 2.0