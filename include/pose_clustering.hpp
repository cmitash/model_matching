#ifndef __CLUSTERING__
#define __CLUSTERING__

#include <rgbd.hpp>

namespace clustering {
using PointType = pcl::PointNormal;

void
greedy_clustering(std::vector< PoseCandidate* > &hypotheses_set,
				float acceptable_fraction,
				float best_score,
				int maximum_pose_count,
				float min_distance,
				float min_angle,
				Eigen::Vector3f sym_info,
				std::vector< PoseCandidate* > &clustered_hypotheses_set);

void 
point_to_plane_icp(pcl::PointCloud<PointType>::Ptr segment, 
					pcl::PointCloud<PointType>::Ptr model, 
					Eigen::Matrix4f &offset_transform);

void 
trimmed_icp(pcl::PointCloud<PointType>::Ptr segment_cloud, 
				pcl::PointCloud<PointType>::Ptr model_cloud, 
				Eigen::Matrix4f &offset_transform,
				std::string save_destination);

}



#endif