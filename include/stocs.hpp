#ifndef __STOCS__
#define __STOCS__

#include <rgbd.hpp>
#include "accelerators/kdtree.h"
#include "Eigen/Dense"

using Scalar = typename Point3D::Scalar;
using MatrixType = Eigen::Matrix<Scalar, 4, 4, Eigen::DontAlign>;
using VectorType = typename Point3D::VectorType;

static constexpr Scalar kLargeNumber = 1e9;

namespace stocs {

class stocs_estimator {
	public:
		stocs_estimator(std::string model_location,
						PPFMapType& ppf_map_preloaded,
						std::string rgb_location,
						std::string depth_location,
						std::string class_probability_map_location,
						std::string edge_probability_map_location,
						std::string debug_location,
						std::vector<float> camera_intrinsics,
						int image_width, int image_height,
						float read_depth_scale, float write_depth_scale,
						float voxel_size, float distance_threshold,
						int ppf_tr_discretization, int ppf_rot_discretization,
						float edge_threshold, float class_threshold){

			this->debug_location = debug_location;
			this->distance_threshold = distance_threshold;
			this->ppf_tr_discretization = ppf_tr_discretization;
			this->ppf_rot_discretization = ppf_rot_discretization;
			this->edge_threshold = edge_threshold;
			this->class_threshold = class_threshold;
			this->image_width = image_width;
			this->image_height = image_height;

			all_transforms.clear();
			all_pose.clear();
			best_lcp = 0;
			best_index = -1;

			load_object_info(model_location, ppf_map_preloaded);
			load_scene_info (rgb_location, depth_location,
							class_probability_map_location,
							edge_probability_map_location,
							camera_intrinsics,
							read_depth_scale,
							write_depth_scale,
							voxel_size, debug_location + "/sampled_scene.ply");

			// Move the centroid of the point sets to 0 to get robust rotation
			centroid_shift();
			kdtree_initialize();

			previous_segment = cv::Mat::zeros (image_height, image_width, CV_8UC1);
			segmentation_buffer = cv::Mat::zeros (image_height, image_width, CV_8UC1);
		};

		~stocs_estimator(){};

		void
		load_object_info(std::string model_location,
						PPFMapType& ppf_map_preloaded);

		void 
		load_scene_info (std::string rgb_location,
						std::string depth_location,
						std::string class_probability_map_location,
						std::string edge_probability_map_location,
						std::vector<float> camera_intrinsics,
						float read_depth_scale,
						float write_depth_scale,
						float voxel_size,
						std::string dst_scene_location);

		bool
		sample_class_base(std::vector<int> &base_indices,
						float &invariant1,
						float &invariant2);

		bool
		sample_instance_base(std::vector<int> &base_indices,
						float &invariant1,
						float &invariant2,
						std::vector<Point3D>& segment,
						float dispersion,
						int base_num);

		bool
		find_congruent_sets_on_model(std::vector<int> &base_indices,
									float invariant1, float invariant2,
									std::vector<Quadrilateral>* quadrilaterals);

		bool 
		get_rigid_transform_from_congruent_pair(std::vector<int> &base_indices,
		        								Quadrilateral &congruent_quad,
		        								int base_index);

		Scalar
		compute_alignment_score_for_rigid_transform(const Eigen::Ref<const MatrixType> &mat);

		void
		compute_best_transform();

		void 
		kdtree_initialize();

		void
		centroid_shift();

		VectorType
		get_scene_centroid() {return centroid_scene_;}

		std::vector< PoseCandidate* > 
		get_pose_candidates() {
			return all_pose;
		}

		Scalar
		get_best_score() {
			return best_lcp;
		}

		PoseCandidate*
		get_best_pose() {
			if (best_index == -1)
				return NULL;
				
			return all_pose[best_index];
		}

		void
		visualize_best_pose() {

			if (best_index == -1)
				return;

			// save all transformed models
			std::vector<Point3D> point3d_model_pose;

			point3d_model_pose.clear();
			rgbd::transform_pointset(point3d_model, point3d_model_pose, all_transforms[best_index]);
			rgbd::save_as_ply(debug_location + "/best_pose.ply", point3d_model_pose, 1);
			rgbd::save_as_ply(debug_location + "/scene.ply", point3d_scene, 1);
		}


	protected:
		std::vector<Point3D> point3d_scene;
		cv::Mat edge_probability_map;
		cv::Mat previous_segment;
		cv::Mat segmentation_buffer;

		Super4PCS::KdTree<Scalar> kd_tree_;
		VectorType centroid_scene_;

		std::vector<Point3D> point3d_model;
		PPFMapType ppf_map;
		VectorType centroid_model_;

		std::vector<MatrixType> all_transforms;
		std::vector< PoseCandidate* > all_pose;

		std::string debug_location;
		float distance_threshold;
		int ppf_tr_discretization;
		int ppf_rot_discretization;
		float edge_threshold;
		float class_threshold;

		Scalar best_lcp;
		int best_index;

		int image_width, image_height;

};

void 
pre_process_model(std::string src_model_location,
					float normal_radius,
					float read_depth_scale,
					float write_depth_scale,
					float voxel_size,
					float ppf_tr_discretization,
					float ppf_rot_discretization,
					std::string dst_model_location,
					std::string dst_ppf_map_location);

} // namespace stocs

#endif