#ifndef _MODEL_MATCHING_
#define _MODEL_MATCHING_

#include <chrono>

// OpenCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/rgbd.hpp>

// PCL
#include <pcl/io/ply_io.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/surface/mls.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/extract_indices.h>

#include<sampler.hpp>

using PPFMapType = std::map<std::vector<int>, std::vector<std::pair<int, int> > >;
using PCLPointCloud = pcl::PointCloud<pcl::PointXYZRGBNormal>;
using ICPICPPointType = pcl::PointNormal;
using micro = std::chrono::microseconds;

namespace rgbd{

void
load_rgbd_data_sampled(std::string rgb_location,
                  std::string depth_location,
                  std::string class_probability_map_location,
                  cv::Mat& edge_probability_map,
                  std::vector<float> camera_intrinsics,
                  float depth_scale,
                  float voxel_size,
                  float class_probability_threshold,
                  std::vector<Point3D>& point3d);

void 
load_ply_model(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
	             std::vector<Point3D>& point3d,
	             float scale);

void
save_as_ply(std::string location, 
            std::vector<Point3D>& point3d,
            float scale);

void
transform_pointset(std::vector<Point3D>& input,
                  std::vector<Point3D>& output,
                  Eigen::Matrix<Point3D::Scalar, 4, 4> &transform);

void
compute_normal_pcl(pcl::PointCloud<pcl::PointXYZRGBNormal>::Ptr cloud,
                  float radius);

void 
ppf_compute(Point3D point_1, 
            Point3D point_2, 
            float tr_discretization,
            float rot_discretization,
            std::vector<int> &ppf_);

void 
ppf_map_insert(std::map<std::vector<int>, std::vector<std::pair<int, int> > > &PPFMap,
              std::vector<int> ppf_,
              float tr_discretization,
              float rot_discretization,
              std::pair<int, int> paired_index);

void
save_ppf_map(std::string location,
			       std::map<std::vector<int>, std::vector<std::pair<int, int> > > &ppf_map);

void
load_ppf_map(std::string ppf_map_location,
            std::map<std::vector<int>, std::vector<std::pair<int, int> > > &ppf_map);

void
visualize_heatmap(std::vector<Point3D>& point3d,
                  std::vector<int> samples,
                  std::string save_location,
                  int image_width, int image_height,
                  int block_size);

void
generate_local_segmentation_mask(Point3D& p, 
                           cv::Mat edge_probability_map,
                           float max_distance,
                           cv::Mat& closed_list);

void
generate_segmentation_mask(Point3D& p, 
                           cv::Mat edge_probability_map,
                           float max_distance,
                           cv::Mat& closed_list,
                           cv::Mat& segmentation_buffer,
                           int base_num,
                           std::string debug_location);

} //namespace rgbd
#endif