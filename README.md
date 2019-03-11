# Model-Matching
This software tool could be used to obtain robust 6d poses of objects with 3d point cloud models in the presence of noisy segmentation data

#### Robust 6D Object Pose Estimation with Stochastic Congruent Sets ([pdf](http://bmvc2018.org/contents/papers/1046.pdf))([website](http://paul.rutgers.edu/~cm1074/))
By Chaitanya Mitash, Abdeslam Boularias, Kostas Bekris (Rutgers University).

In Proceedings of British Machine Vision Conference (BMVC), Newcastle, England, UK, 2018

### Citing
To cite the work:

```
@article{mitash2018robust,
  title={Robust 6D object pose estimation with stochastic congruent sets},
  author={Mitash, Chaitanya and Boularias, Abdeslam and Bekris, Kostas},
  journal={arXiv preprint arXiv:1805.06324},
  year={2018}
}
```

### Installation
1. Download the repository.
2. mkdir build
3. cd build
4. cmake ../
5. make

### Inputs
1. RGB and depth images
2. Per-pixel object class probability (scaled to range 0-10000 and stored as uint16). Can be set as a constant mask if probability is not available.

### Outputs
1. ```best_pose_candidate_{object_name}``` 6D pose of the object (3 rows of the transformation matrix) stored in row-major order.
2. ```best_pose.ply``` and ```scene.ply``` visualization of the transformed object model and the scene.

### Running the first example
1. Set the ```repo_path``` in files ```model_preprocess.cpp``` and ```stocs_match_one_object.cpp```
2. Preprocess the 3d model
```./build/model_preprocess "024_bowl"```
3. Run pose estimation
```./build/stocs_single "{path_to_repo}/examples/ycb/" "024_bowl"```

### Running on Packed-dataset 
1. Change the following parameters in the file ```model_preprocess.cpp```
```float voxel_size = 0.005;```

2. Change the following parameters in the file ```stocs_match_one_object.cpp```
```
std::vector<float> cam_intrinsics = {615.957763671875, 308.1098937988281, 615.9578247070312, 246.33352661132812};
float depth_scale = 1/8000.0f;
```

### Running on Linemod
1. Change the following parameters in the file ```model_preprocess.cpp```
```
float voxel_size = 10;
float normal_radius = 5;
float model_scale = 1.0f/1000;
```

2. Change the following parameters in the file ```stocs_match_one_object.cpp```
```
std::vector<float> cam_intrinsics = {572.4114, 325.2611, 573.57043, 242.04899};
float depth_scale = 1/1000.0f;
```
