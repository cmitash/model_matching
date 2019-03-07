# Model-Matching
This software tool could be used to obtain robust 6d poses of objects with 3d point cloud models in the presence of noisy segmentation data

#### Robust 6D Object Pose Estimation with Stochastic Congruent Sets ([pdf](http://bmvc2018.org/contents/papers/1046.pdf))([website](http://paul.rutgers.edu/~cm1074/)
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

### Running the first example
1. Set the ```repo_path``` in files ```model_preprocess.cpp``` and ```stocs_match_one_object.cpp```
2. Preprocess the 3d model
```./build/model_preprocess "024_bowl"```
3. Run pose estimation
```./build/stocs_single "{path_to_repo}/examples/ycb/" "024_bowl"```

### Inputs
1. RGB and depth images
2. Per-pixel object class probability (scaled by 10000). Can be set as a constant mask.

### Outputs
1. ```best_pose_candidate_{object_name}``` 6D pose of the object (3 rows of the transformation matrix) stored in row-major order.
2. ```best_pose.ply``` and ```scene.ply``` visualization of the transformed object model and the scene.
