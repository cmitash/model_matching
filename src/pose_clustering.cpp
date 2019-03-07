#include <pose_clustering.hpp>

namespace clustering {

static void 
quaternion_to_euler(Eigen::Quaternionf& q,
					Eigen::Vector3f& euler_angles){

	// roll (x-axis rotation)
	double sinr = +2.0 * (q.w() * q.x() + q.y() * q.z());
	double cosr = +1.0 - 2.0 * (q.x() * q.x() + q.y() * q.y());
	euler_angles[0] = atan2(sinr, cosr);

	// pitch (y-axis rotation)
	double sinp = +2.0 * (q.w() * q.y() - q.z() * q.x());
	if (fabs(sinp) >= 1)
		euler_angles[1] = copysign(M_PI / 2, sinp); // use 90 degrees if out of range
	else
		euler_angles[1] = asin(sinp);

	// yaw (z-axis rotation)
	double siny = +2.0 * (q.w() * q.z() + q.x() * q.y());
	double cosy = +1.0 - 2.0 * (q.y() * q.y() + q.z() * q.z());  
	euler_angles[2] = atan2(siny, cosy);
}

static void 
get_pose_diff(Eigen::Matrix4f test_pose, 
			Eigen::Matrix4f base_pose, 
			Eigen::Vector3f sym_info, 
			float& mean_rotation_error, 
			float& translation_error){

	Eigen::Matrix3f test_rotation, base_rotation, rotation_diff;
	Eigen::Vector3f rotation_error_xyz;

	for(int ii = 0; ii < 3; ii++)
		for(int jj=0; jj < 3; jj++){
			test_rotation(ii,jj) = test_pose(ii,jj);
			base_rotation(ii,jj) = base_pose(ii,jj);
		}

	test_rotation = test_rotation.inverse().eval();
	rotation_diff = test_rotation * base_rotation;

	Eigen::Quaternionf difference_quaternion(rotation_diff);
	quaternion_to_euler(difference_quaternion, rotation_error_xyz);
	rotation_error_xyz = rotation_error_xyz*180.0/M_PI;

	for(int dim = 0; dim < 3; dim++) {

		rotation_error_xyz(dim) = fabs(rotation_error_xyz(dim));
		if (sym_info(dim) == 90){
			rotation_error_xyz(dim) = abs(rotation_error_xyz(dim) - 90);
			rotation_error_xyz(dim) = std::min(rotation_error_xyz(dim), 90 - rotation_error_xyz(dim));
		}
		else if(sym_info(dim) == 180){
			rotation_error_xyz(dim) = std::min(rotation_error_xyz(dim), 180 - rotation_error_xyz(dim));
		}
		else if(sym_info(dim) == 360){
			rotation_error_xyz(dim) = 0;
		}
	}

	// mean_rotation_error = (rotation_error_xyz(0) + rotation_error_xyz(1) + rotation_error_xyz(2))/3;
	mean_rotation_error = std::max(std::max(rotation_error_xyz(0), rotation_error_xyz(1)), rotation_error_xyz(2));

	translation_error = sqrt(pow(base_pose(0,3) - test_pose(0,3), 2) + 
						pow(base_pose(1,3) - test_pose(1,3), 2) + 
						pow(base_pose(2,3) - test_pose(2,3), 2));
}

static bool 
sort_poses(PoseCandidate* &a, PoseCandidate* &b) {

	return (a->lcp > b->lcp);
}

void
greedy_clustering(std::vector< PoseCandidate* > &hypotheses_set,
				float acceptable_fraction,
				float best_score,
				int maximum_pose_count,
				float min_distance,
				float min_angle,
				Eigen::Vector3f sym_info,
				std::vector< PoseCandidate* > &clustered_hypotheses_set) {

	clustered_hypotheses_set.clear();

	std::vector< PoseCandidate* > pruned_hypotheses_set;
	
	for(auto pose_it : hypotheses_set) {

		if(pose_it->lcp > acceptable_fraction * best_score)
			pruned_hypotheses_set.push_back(pose_it);
	}

	std::sort(pruned_hypotheses_set.begin(), pruned_hypotheses_set.end(), sort_poses);

	for(auto candidate_it: pruned_hypotheses_set) {

		bool inValid = false;
		for(auto cluster_it: clustered_hypotheses_set) {

			float mean_rotation_error, translation_error;
			get_pose_diff(candidate_it->transform, cluster_it->transform, sym_info, mean_rotation_error, translation_error);

			if(mean_rotation_error < min_angle && translation_error < min_distance) {
				inValid = true;
				break;
			}
		}

		if(inValid == false)
			clustered_hypotheses_set.push_back(candidate_it);

		if(clustered_hypotheses_set.size() > maximum_pose_count)
			break;
	}
}

void point_to_plane_icp(pcl::PointCloud<PointType>::Ptr segment_cloud, 
					pcl::PointCloud<PointType>::Ptr model_cloud, 
					Eigen::Matrix4f &offset_transform) {

	pcl::IterativeClosestPointWithNormals<PointType, PointType>::Ptr icp ( new pcl::IterativeClosestPointWithNormals<PointType, PointType> () );

	icp->setMaximumIterations ( 5 );
	icp->setMaxCorrespondenceDistance (0.035);
	icp->setInputSource ( segment_cloud );
	icp->setInputTarget ( model_cloud );
	icp->align ( *segment_cloud );

	if ( icp->hasConverged() )
		offset_transform = icp->getFinalTransformation();
	else
		offset_transform.setIdentity();

}

} //namespace clustering