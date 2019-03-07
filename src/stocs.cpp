#include <stocs.hpp>

#include <boost/functional/hash.hpp>

#include "Eigen/Core"
#include "Eigen/Geometry"                 // MatrixBase.homogeneous()
#include "Eigen/SVD"                      // Transform.computeRotationScaling()

#include "accelerators/normalset.h"
#include "pairCreationFunctor.h"
#include "accelerators/bbox.h"
#include "accelerators/kdtree.h"

namespace std {
  template<typename... T>
  struct hash<tuple<T...>>
  {
      size_t operator()(tuple<T...> const& arg) const noexcept
      {
          return boost::hash_value(arg);
      }
  };
}

namespace stocs {

// supposed to be used offline. Otherwise this can be optimized by reading directly into point3d.
void 
pre_process_model(std::string src_model_location,
								float normal_radius,
								float read_depth_scale,
								float write_depth_scale,
								float voxel_size,
								float ppf_tr_discretization,
								float ppf_rot_discretization,
								std::string dst_model_location,
								std::string dst_ppf_map_location) {

	std::vector<Point3D> point3d, point3d_sampled;
	rgbd::UniformDistSampler sampler;
	std::map<std::vector<int>, std::vector<std::pair<int, int> > > ppf_map;

	PCLPointCloud::Ptr cloud (new PCLPointCloud);
	pcl::io::loadPLYFile(src_model_location, *cloud);
	rgbd::compute_normal_pcl(cloud, normal_radius);

	//adding a negative sign. Reference frame is inside the object, so by default normals face inside.
	for (int i = 0; i < cloud->points.size(); i++) {
		cloud->points[i].normal[0] = -cloud->points[i].normal[0];
		cloud->points[i].normal[1] = -cloud->points[i].normal[1];
		cloud->points[i].normal[2] = -cloud->points[i].normal[2];
	}

	pcl::VoxelGrid<pcl::PointXYZRGBNormal> sor;
	sor.setInputCloud (cloud);
	sor.setLeafSize (voxel_size, voxel_size, voxel_size);
	sor.filter (*cloud);

	rgbd::load_ply_model(cloud, point3d_sampled, read_depth_scale);
	std::cout << "After sampling |M|= " << point3d_sampled.size() << std::endl;

	float max_distance = 0;
	for (int id1 = 0; id1 < point3d_sampled.size(); id1++){
		for (int id2 = 0; id2 < point3d_sampled.size(); id2++){

			if(id1 == id2)
				continue;

			std::vector<int> ppf_;
			rgbd::ppf_compute(point3d_sampled[id1], point3d_sampled[id2], ppf_tr_discretization, ppf_rot_discretization, ppf_);
			rgbd::ppf_map_insert(ppf_map, ppf_, ppf_tr_discretization, ppf_rot_discretization, std::make_pair(id1, id2));

			float d = (point3d_sampled[id1].pos() - point3d_sampled[id2].pos()).norm();
			if (d > max_distance)
				max_distance = d;

		}
	}

	std::cout << "max distance is: " << max_distance << std::endl;

	rgbd::save_ppf_map(dst_ppf_map_location, ppf_map);
	rgbd::save_as_ply(dst_model_location, point3d_sampled, write_depth_scale);
}

void
stocs_estimator::load_object_info(std::string model_location,
								PPFMapType& ppf_map_preloaded) {
	point3d_model.clear();

	PCLPointCloud::Ptr cloud (new PCLPointCloud);
	pcl::io::loadPLYFile(model_location, *cloud);
	rgbd::load_ply_model(cloud, point3d_model, 1.0f);
	ppf_map = ppf_map_preloaded;

	std::cout << "|M| = " << point3d_model.size() << ",  |map(M)| = " << ppf_map.size() << std::endl;
}

void 
stocs_estimator::load_scene_info (std::string rgb_location,
								std::string depth_location,
								std::string class_probability_map_location,
								std::string edge_probability_map_location,
								std::vector<float> camera_intrinsics,
								float read_depth_scale,
								float write_depth_scale,
								float voxel_size,
								std::string dst_scene_location) {

	point3d_scene.clear();
	std::vector<Point3D> tmp_point3d_sampled;
	rgbd::UniformDistSampler sampler;
	
	struct stat buffer;
	if(stat (edge_probability_map_location.c_str(), &buffer) == 0)
		edge_probability_map = cv::imread(edge_probability_map_location, CV_8UC1);
	else
		edge_probability_map = cv::Mat::zeros (image_height, image_width, CV_8UC1);

	rgbd::load_rgbd_data_sampled(rgb_location,
						depth_location,
						class_probability_map_location,
						edge_probability_map,
						camera_intrinsics,
						read_depth_scale,
                  		voxel_size,
                  		class_threshold,
                  		point3d_scene);
	
	rgbd::save_as_ply(dst_scene_location, point3d_scene, write_depth_scale);
}

static int 
sample_point_from_distribution(std::vector<Point3D>& point3d) {
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

	std::default_random_engine generator (seed);
	std::vector<float> probability_distribution;
	int sampled_index = -1;

	for(auto p:point3d)
		probability_distribution.push_back(p.probability());

	std::discrete_distribution<int> p_dist (probability_distribution.begin(),probability_distribution.end());
	sampled_index = p_dist(generator);
	
	return sampled_index;
}

// Compute the closest points between two 3D line segments and obtain the two
// invariants corresponding to the closet points. This is the "intersection"
// point that determines the invariants. Since the 4 points are not exactly
// planar, we use the center of the line segment connecting the two closest
// points as the "intersection".
template < typename VectorType, typename Scalar>
static Scalar
segment_distance_and_invariants(const VectorType& p1, const VectorType& p2,
							const VectorType& q1, const VectorType& q2,
							Scalar& invariant1, Scalar& invariant2) {

  static const Scalar kSmallNumber = 0.0001;
  VectorType u = p2 - p1;
  VectorType v = q2 - q1;
  VectorType w = p1 - q1;
  Scalar a = u.dot(u);
  Scalar b = u.dot(v);
  Scalar c = v.dot(v);
  Scalar d = u.dot(w);
  Scalar e = v.dot(w);
  Scalar f = a * c - b * b;
  // s1,s2 and t1,t2 are the parametric representation of the intersection.
  // they will be the invariants at the end of this simple computation.
  Scalar s1 = 0.0;
  Scalar s2 = f;
  Scalar t1 = 0.0;
  Scalar t2 = f;

  if (f < kSmallNumber) {
    s1 = 0.0;
    s2 = 1.0;
    t1 = e;
    t2 = c;
  } else {
    s1 = (b * e - c * d);
    t1 = (a * e - b * d);
    if (s1 < 0.0) {
      s1 = 0.0;
      t1 = e;
      t2 = c;
    } else if (s1 > s2) {
      s1 = s2;
      t1 = e + b;
      t2 = c;
    }
  }

  if (t1 < 0.0) {
    t1 = 0.0;
    if (-d < 0.0)
      s1 = 0.0;
    else if (-d > a)
      s1 = s2;
    else {
      s1 = -d;
      s2 = a;
    }
  } else if (t1 > t2) {
    t1 = t2;
    if ((-d + b) < 0.0)
      s1 = 0;
    else if ((-d + b) > a)
      s1 = s2;
    else {
      s1 = (-d + b);
      s2 = a;
    }
  }
  invariant1 = (std::abs(s1) < kSmallNumber ? 0.0 : s1 / s2);
  invariant2 = (std::abs(t1) < kSmallNumber ? 0.0 : t1 / t2);

  return ( w + (invariant1 * u) - (invariant2 * v)).norm();
}

static bool
try_sampled_base(std::vector<Point3D> base_3d, float &invariant1, float &invariant2, int& id1, int& id2, int& id3, int& id4) {

  float min_distance = std::numeric_limits<float>::max();
  int best1, best2, best3, best4;
  best1 = best2 = best3 = best4 = -1;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 4; ++j) {
      if (i == j) continue;
      int k = 0;
      while (k == i || k == j) k++;
      int l = 0;
      while (l == i || l == j || l == k) l++;
      double local_invariant1;
      double local_invariant2;
      // Compute the closest points on both segments, the corresponding
      // invariants and the distance between the closest points.
      Scalar segment_distance = segment_distance_and_invariants(
                  base_3d[i].pos(), base_3d[j].pos(),
                  base_3d[k].pos(), base_3d[l].pos(),
                  local_invariant1, local_invariant2);
      
      // Retail the smallest distance and the best order so far.
      if (segment_distance < min_distance) {
        min_distance = segment_distance;
        best1 = i;
        best2 = j;
        best3 = k;
        best4 = l;
        invariant1 = local_invariant1;
        invariant2 = local_invariant2;
      }
    }
  }

  if(best1 < 0 || best2 < 0 || best3 < 0 || best4 < 0 ) return false;

  std::array<int, 4> tmpId = {id1, id2, id3, id4};
  id1 = tmpId[best1];
  id2 = tmpId[best2];
  id3 = tmpId[best3];
  id4 = tmpId[best4];

  return true;
}

static bool
ComputeRigidTransformation(const std::array< std::pair<Point3D, Point3D>,4>& pairs,
						const Eigen::Matrix<Scalar, 3, 1>& centroid1,
						Eigen::Matrix<Scalar, 3, 1> centroid2,
						Eigen::Ref<MatrixType> transform,
						Scalar& rms_,
						bool computeScale ) {

  rms_ = kLargeNumber;

  if (pairs.size() == 0 || pairs.size() % 2 != 0)
      return false;


  Scalar kSmallNumber = 1e-6;

  // We only use the first 3 pairs. This simplifies the process considerably
  // because it is the planar case.

  const VectorType& p0 = pairs[0].first.pos();
  const VectorType& p1 = pairs[1].first.pos();
  const VectorType& p2 = pairs[2].first.pos();
        VectorType  q0 = pairs[0].second.pos();
        VectorType  q1 = pairs[1].second.pos();
        VectorType  q2 = pairs[2].second.pos();

  Scalar scaleEst (1.);

  VectorType vector_p1 = p1 - p0;
  if (vector_p1.squaredNorm() == 0) return kLargeNumber;
  vector_p1.normalize();
  VectorType vector_p2 = (p2 - p0) - ((p2 - p0).dot(vector_p1)) * vector_p1;
  if (vector_p2.squaredNorm() == 0) return kLargeNumber;
  vector_p2.normalize();
  VectorType vector_p3 = vector_p1.cross(vector_p2);

  VectorType vector_q1 = q1 - q0;
  if (vector_q1.squaredNorm() == 0) return kLargeNumber;
  vector_q1.normalize();
  VectorType vector_q2 = (q2 - q0) - ((q2 - q0).dot(vector_q1)) * vector_q1;
  if (vector_q2.squaredNorm() == 0) return kLargeNumber;
  vector_q2.normalize();
  VectorType vector_q3 = vector_q1.cross(vector_q2);

  Eigen::Matrix<Scalar, 3, 3> rotation = Eigen::Matrix<Scalar, 3, 3>::Identity();

  Eigen::Matrix<Scalar, 3, 3> rotate_p;
  rotate_p.row(0) = vector_p1;
  rotate_p.row(1) = vector_p2;
  rotate_p.row(2) = vector_p3;

  Eigen::Matrix<Scalar, 3, 3> rotate_q;
  rotate_q.row(0) = vector_q1;
  rotate_q.row(1) = vector_q2;
  rotate_q.row(2) = vector_q3;

  rotation = rotate_p.transpose() * rotate_q;

  // Discard singular solutions. The rotation should be orthogonal.
  if (((rotation * rotation).diagonal().array() - Scalar(1) > kSmallNumber).any())
      return false;

  //FIXME
  // Compute rms and return it.
  rms_ = Scalar(0.0);
  {
      VectorType first, transformed;

      //cv::Mat first(3, 1, CV_64F), transformed;
      for (int i = 0; i < 3; ++i) {
          first = scaleEst*pairs[i].second.pos() - centroid2;
          transformed = rotation * first;
          rms_ += (transformed - pairs[i].first.pos() + centroid1).norm();
      }
  }

  rms_ /= Scalar(pairs.size());

  Eigen::Transform<Scalar, 3, Eigen::Affine> etrans (Eigen::Transform<Scalar, 3, Eigen::Affine>::Identity());

  // compute rotation and translation
  {
      etrans.scale(scaleEst);       // apply scale factor
      etrans.translate(centroid1);  // translation between quads
      etrans.rotate(rotation);           // rotate to align frames
      etrans.translate(-centroid2); // move to congruent quad frame

      transform = etrans.matrix();
  }

  return true;
}

bool
stocs_estimator::sample_class_base(std::vector<int> &base_indices,
								float &invariant1,
								float &invariant2) {
	
	float plane_threshold = 0.015;
	float min_distance_base = 0.01;
	float internal_angle_threshold = 30;

	// a copy of points need to be created; every base will start from the prior
	for(int i = 0; i < point3d_scene.size(); i++) {
		auto p1 = point3d_scene[i].pixel();
		int isPresent = (int)previous_segment.at<unsigned char>(p1.first, p1.second);

		if(isPresent)
			point3d_scene[i].update_class_probability(1.0);
		
		point3d_scene[i].reset_probability();
	}

	// SAMPLE Point 1
	int base_index1 = sample_point_from_distribution(point3d_scene);

	if(point3d_scene[base_index1].probability() == 0.0) {
		std::cout << "FAILED SAMPLING:: Zero probability returned!!!" << std::endl;
		return false;
	}

	// visualize probability distribution
	// rgbd::visualize_heatmap(point3d_scene, {base_index1}, debug_location + "/p_dist_1.png", image_width, image_height, 8);

	// update distribution
	for (int i=0; i<point3d_scene.size(); i++) {
		std::vector<int> ppf;
		rgbd::ppf_compute(point3d_scene[base_index1],
						point3d_scene[i], 
						ppf_tr_discretization,
						ppf_rot_discretization,
						ppf);

		auto map_it = ppf_map.find(ppf);

    	if(map_it == ppf_map.end() || i == base_index1)
			point3d_scene[i].update_probability(0);
	}

	// SAMPLE Point 2
	int base_index2 = sample_point_from_distribution(point3d_scene);

	if(point3d_scene[base_index2].probability() == 0.0) {
		std::cout << "FAILED SAMPLING:: Zero probability returned!!!" << std::endl;
		return false;
	}

	VectorType v_1 = point3d_scene[base_index2].pos() - point3d_scene[base_index1].pos();
	v_1.normalize();

	// visualize probability distribution
	// rgbd::visualize_heatmap(point3d_scene, {base_index1, base_index2}, debug_location + "/p_dist_2.png", image_width, image_height, 8);

	// update distribution
	for (int i=0; i<point3d_scene.size(); i++) {
		VectorType v_2 = point3d_scene[i].pos() - point3d_scene[base_index1].pos();
		v_2.normalize();

    	float int_angle = acos(v_1.dot(v_2))*180/M_PI;
   		int_angle = std::min(int_angle, 180-int_angle);

		std::vector<int> ppf;
		rgbd::ppf_compute(point3d_scene[base_index2],
						point3d_scene[i], 
						ppf_tr_discretization,
						ppf_rot_discretization,
						ppf);

		auto map_it = ppf_map.find(ppf);

    	if(map_it == ppf_map.end() || i == base_index2 || int_angle < internal_angle_threshold)
			point3d_scene[i].update_probability(0);
	}

	// SAMPLE Point 3
	int base_index3 = sample_point_from_distribution(point3d_scene);

	if(point3d_scene[base_index3].probability() == 0.0) {
		std::cout << "FAILED SAMPLING:: Zero probability returned!!!" << std::endl;
		return false;
	}

	// visualize probability distribution
	// rgbd::visualize_heatmap(point3d_scene, {base_index1, base_index2, base_index3}, debug_location + "/p_dist_3.png", image_width, image_height, 8);

	// update distribution
	for (int i=0; i<point3d_scene.size(); i++) {
		// The 4th point will be a one that is close to be planar
	    double x1 = point3d_scene[base_index1].x();
	    double y1 = point3d_scene[base_index1].y();
	    double z1 = point3d_scene[base_index1].z();
	    double x2 = point3d_scene[base_index2].x();
	    double y2 = point3d_scene[base_index2].y();
	    double z2 = point3d_scene[base_index2].z();
	    double x3 = point3d_scene[base_index3].x();
	    double y3 = point3d_scene[base_index3].y();
	    double z3 = point3d_scene[base_index3].z();

	    // Fit a plane
	    float denom = (-x3 * y2 * z1 + x2 * y3 * z1 + x3 * y1 * z2 - x1 * y3 * z2 -
	                    x2 * y1 * z3 + x1 * y2 * z3);

	    float planar_distance = 10000;
	    if (denom != 0) {
			float A = (-y2 * z1 + y3 * z1 + y1 * z2 - y3 * z2 - y1 * z3 + y2 * z3) / denom;
			float B = (x2 * z1 - x3 * z1 - x1 * z2 + x3 * z2 + x1 * z3 - x2 * z3) / denom;
			float C = (-x2 * y1 + x3 * y1 + x1 * y2 - x3 * y2 - x1 * y3 + x2 * y3) / denom;

			planar_distance = std::abs(A * point3d_scene[i].x() + B * point3d_scene[i].y() + C * point3d_scene[i].z() - 1.0);
		}

		std::vector<int> ppf;
		rgbd::ppf_compute(point3d_scene[base_index3],
						point3d_scene[i], 
						ppf_tr_discretization,
						ppf_rot_discretization,
						ppf);
		auto map_it = ppf_map.find(ppf);

		if(planar_distance > plane_threshold ||
			(point3d_scene[i].pos()- point3d_scene[base_index1].pos()).norm() < min_distance_base || 
			(point3d_scene[i].pos()- point3d_scene[base_index2].pos()).norm() < min_distance_base || 
			(point3d_scene[i].pos()- point3d_scene[base_index3].pos()).norm() < min_distance_base ||
			 map_it == ppf_map.end() || i == base_index3) {

			point3d_scene[i].update_probability(0);
		}
	}

	// SAMPLE Point 3
	int base_index4 = sample_point_from_distribution(point3d_scene);

	if(point3d_scene[base_index4].probability() == 0.0) {
		std::cout << "FAILED SAMPLING:: Zero probability returned!!!" << std::endl;
		return false;
	}

	// visualize probability distribution
	// rgbd::visualize_heatmap(point3d_scene, {base_index1, base_index2, base_index3, base_index4}, debug_location + "/p_dist_4.png", image_width, image_height, 8);

	base_indices[0] = base_index1;
	base_indices[1] = base_index2;
	base_indices[2] = base_index3;
	base_indices[3] = base_index4;

	bool valid = try_sampled_base( {point3d_scene[base_index1], point3d_scene[base_index2], point3d_scene[base_index3], point3d_scene[base_index4]},
									invariant1, invariant2, base_indices[0], base_indices[1], base_indices[2], base_indices[3]);

	return valid;
}

static void
prune_edge_pixels(std::vector<Point3D>& point3d,
				cv::Mat edge_probability_map) {

	for(int i=0; i< point3d.size(); i++) {
		int u = point3d[i].pixel().first;
    	int v = point3d[i].pixel().second;

    	float edge_probability = (float)(255.0 - edge_probability_map.at<unsigned char>(u,v))/255.0;
    	
    	if(edge_probability == 1){
    		point3d[i].update_probability(0);
    	}
	}
}

static float
get_edge_probability_in_segment(Point3D p1, Point3D p2, cv::Mat edge_probability_map) {

	float max_probability = 0;

	// opencv points have format (column, row)
	cv::Point pixel1(p1.pixel().second, p1.pixel().first);
	cv::Point pixel2(p2.pixel().second, p2.pixel().first);

	cv::LineIterator line_iterator(edge_probability_map, pixel1, pixel2, 8);

	for(int i = 0; i < line_iterator.count; i++, line_iterator++) {
		// float edge_probability = (float)edge_probability_map.at<unsigned short>(line_iterator.pos())/10000;
		float edge_probability = (float)(255.0 - edge_probability_map.at<unsigned char>(line_iterator.pos()))/255.0;

		if(edge_probability > max_probability)
			max_probability = edge_probability;
	}

	return max_probability;
}

bool
stocs_estimator::sample_instance_base(std::vector<int> &base_indices,
									float &invariant1,
									float &invariant2,
									std::vector<Point3D>& segment,
									float dispersion,
									int base_num) {
	
	float plane_threshold = 0.015;
	float min_distance_base = 0.01;
	float internal_angle_threshold = 30;

	// a copy of points need to be created; every base will start from the prior
	for(int i = 0; i < point3d_scene.size(); i++) {
		auto p1 = point3d_scene[i].pixel();
		int isPresent = (int)previous_segment.at<unsigned char>(p1.first, p1.second);

		if(isPresent)
			point3d_scene[i].update_class_probability(dispersion);
		
		point3d_scene[i].reset_probability();
	}
  	
  	prune_edge_pixels(point3d_scene, edge_probability_map);
 
	// SAMPLE Point 1
	int base_index1 = sample_point_from_distribution(point3d_scene);

	if(point3d_scene[base_index1].probability() == 0.0) {
		// std::cout << "FAILED SAMPLING:: Zero probability returned!!!" << std::endl;
		return false;
	}

	// visualize probability distribution
	// rgbd::visualize_heatmap(point3d_scene, {base_index1}, debug_location + "/p_dist_1.png", image_width, image_height, 8);

	// update distribution
	float max_pixel_distance = 0;
	for (int i=0; i<point3d_scene.size(); i++) {
		std::vector<int> ppf;
		rgbd::ppf_compute(point3d_scene[base_index1],
						point3d_scene[i], 
						ppf_tr_discretization,
						ppf_rot_discretization,
						ppf);

		auto map_it = ppf_map.find(ppf);

    	if(map_it == ppf_map.end() || i == base_index1)
			point3d_scene[i].update_probability(0);

		if(point3d_scene[i].probability() != 0) {
			auto p1 = point3d_scene[base_index1].pixel();
			auto p2 = point3d_scene[i].pixel();

			float dist = std::sqrt(std::pow((p1.first - p2.first), 2) + std::pow((p1.second - p2.second), 2));
			
			if (dist > max_pixel_distance)
				max_pixel_distance = dist;
		}
	}

	cv::Mat segmentation_mask = cv::Mat::zeros (image_height, image_width, CV_8UC1);

	rgbd::generate_segmentation_mask(point3d_scene[base_index1], edge_probability_map, max_pixel_distance, segmentation_mask, segmentation_buffer, base_num, debug_location);
	
	cv::imwrite(debug_location + "/seg_mask_" + std::to_string(base_num) + ".png", segmentation_mask);
	segmentation_mask.copyTo(previous_segment);

	for (int i=0; i<point3d_scene.size(); i++) {

		if(point3d_scene[i].probability() != 0) {
			int isValid = (int)segmentation_mask.at<unsigned char>(point3d_scene[i].pixel().first, point3d_scene[i].pixel().second);

			if(isValid)
				segment.push_back(point3d_scene[i]);
			else
				point3d_scene[i].update_probability(0);
		}
	}


	// SAMPLE Point 2
	int base_index2 = sample_point_from_distribution(point3d_scene);

	if(point3d_scene[base_index2].probability() == 0.0) {
		// std::cout << "FAILED SAMPLING:: Zero probability returned!!!" << std::endl;
		return false;
	}

	VectorType v_1 = point3d_scene[base_index2].pos() - point3d_scene[base_index1].pos();
	v_1.normalize();

	// visualize probability distribution
	//rgbd::visualize_heatmap(point3d_scene, {base_index1, base_index2}, debug_location + "/p_dist_2.png", image_width, image_height, 8);

	// update distribution
	for (int i=0; i<point3d_scene.size(); i++) {
		VectorType v_2 = point3d_scene[i].pos() - point3d_scene[base_index1].pos();
		v_2.normalize();

    	float int_angle = acos(v_1.dot(v_2))*180/M_PI;
   		int_angle = std::min(int_angle, 180-int_angle);

		std::vector<int> ppf;
		rgbd::ppf_compute(point3d_scene[base_index2],
						point3d_scene[i], 
						ppf_tr_discretization,
						ppf_rot_discretization,
						ppf);

		auto map_it = ppf_map.find(ppf);

    	if(map_it == ppf_map.end() || i == base_index2 || int_angle < internal_angle_threshold)
			point3d_scene[i].update_probability(0);
	}

	// SAMPLE Point 3
	int base_index3 = sample_point_from_distribution(point3d_scene);

	if(point3d_scene[base_index3].probability() == 0.0) {
		// std::cout << "FAILED SAMPLING:: Zero probability returned!!!" << std::endl;
		return false;
	}

	// visualize probability distribution
	// rgbd::visualize_heatmap(point3d_scene, {base_index1, base_index2, base_index3}, debug_location + "/p_dist_3.png", image_width, image_height, 8);

	// update distribution
	for (int i=0; i<point3d_scene.size(); i++) {
		// The 4th point will be a one that is close to be planar
	    double x1 = point3d_scene[base_index1].x();
	    double y1 = point3d_scene[base_index1].y();
	    double z1 = point3d_scene[base_index1].z();
	    double x2 = point3d_scene[base_index2].x();
	    double y2 = point3d_scene[base_index2].y();
	    double z2 = point3d_scene[base_index2].z();
	    double x3 = point3d_scene[base_index3].x();
	    double y3 = point3d_scene[base_index3].y();
	    double z3 = point3d_scene[base_index3].z();

	    // Fit a plane
	    float denom = (-x3 * y2 * z1 + x2 * y3 * z1 + x3 * y1 * z2 - x1 * y3 * z2 -
	                    x2 * y1 * z3 + x1 * y2 * z3);

	    float planar_distance = 10000;
	    if (denom != 0) {
			float A = (-y2 * z1 + y3 * z1 + y1 * z2 - y3 * z2 - y1 * z3 + y2 * z3) / denom;
			float B = (x2 * z1 - x3 * z1 - x1 * z2 + x3 * z2 + x1 * z3 - x2 * z3) / denom;
			float C = (-x2 * y1 + x3 * y1 + x1 * y2 - x3 * y2 - x1 * y3 + x2 * y3) / denom;

			planar_distance = std::abs(A * point3d_scene[i].x() + B * point3d_scene[i].y() + C * point3d_scene[i].z() - 1.0);
		}

		std::vector<int> ppf;
		rgbd::ppf_compute(point3d_scene[base_index3],
						point3d_scene[i], 
						ppf_tr_discretization,
						ppf_rot_discretization,
						ppf);
		auto map_it = ppf_map.find(ppf);

		if(planar_distance > plane_threshold ||
			(point3d_scene[i].pos()- point3d_scene[base_index1].pos()).norm() < min_distance_base || 
			(point3d_scene[i].pos()- point3d_scene[base_index2].pos()).norm() < min_distance_base || 
			(point3d_scene[i].pos()- point3d_scene[base_index3].pos()).norm() < min_distance_base ||
			 map_it == ppf_map.end() || i == base_index3) {

			point3d_scene[i].update_probability(0);
		}
	}

	// SAMPLE Point 4
	int base_index4 = sample_point_from_distribution(point3d_scene);

	if(point3d_scene[base_index4].probability() == 0.0) {
		// std::cout << "FAILED SAMPLING:: Zero probability returned!!!" << std::endl;
		return false;
	}

	// visualize probability distribution
	// rgbd::visualize_heatmap(point3d_scene, {base_index1, base_index2, base_index3, base_index4}, debug_location + "/p_dist_4.png", image_width, image_height, 8);

	base_indices[0] = base_index1;
	base_indices[1] = base_index2;
	base_indices[2] = base_index3;
	base_indices[3] = base_index4;

	bool valid = try_sampled_base( {point3d_scene[base_index1], point3d_scene[base_index2], point3d_scene[base_index3], point3d_scene[base_index4]},
									invariant1, invariant2, base_indices[0], base_indices[1], base_indices[2], base_indices[3]);

	return valid;
}

bool
stocs_estimator::find_congruent_sets_on_model(std::vector<int> &base_indices,
											float invariant1, float invariant2,
											std::vector<Quadrilateral>* quadrilaterals) {

	std::vector<Point3D> base_3D_(4);	
	PairCreationFunctor<Scalar> pcfunctor_(point3d_model);
	pcfunctor_.synch3DContent();

	std::vector<std::pair<int, int> > P_pairs, Q_pairs;
	std::vector<int> ppf_1, ppf_2;

	base_3D_[0] = point3d_scene[base_indices[0]];
	base_3D_[1] = point3d_scene[base_indices[1]];
	base_3D_[2] = point3d_scene[base_indices[2]];
	base_3D_[3] = point3d_scene[base_indices[3]];

	// computing point pair features
	rgbd::ppf_compute(base_3D_[0], base_3D_[1], ppf_tr_discretization, ppf_rot_discretization, ppf_1);
	rgbd::ppf_compute(base_3D_[2], base_3D_[3], ppf_tr_discretization, ppf_rot_discretization, ppf_2);

	// std::cout << "ppf1: " << ppf_1[0] << " " << ppf_1[1] << " " << ppf_1[2] << " " << ppf_1[3] << std::endl;
	// std::cout << "ppf2: " << ppf_2[0] << " " << ppf_2[1] << " " << ppf_2[2] << " " << ppf_2[3] << std::endl;

	P_pairs.clear();
	Q_pairs.clear();

	auto it_1 = ppf_map.find(ppf_1);
	if(it_1 != ppf_map.end())
		P_pairs = it_1->second;

	auto it_2 = ppf_map.find(ppf_2);
	if(it_2 != ppf_map.end())
  		Q_pairs = it_2->second;

    if (P_pairs.size() == 0 || Q_pairs.size() == 0)
      return false;

	typedef  Super4PCS::IndexedNormalSet
			< VectorType,   //! \brief Point type used internally
			  3,       //! \brief Nb dimension
			  7,       //! \brief Nb cells/dim normal
			  Scalar>  //! \brief Scalar type
    IndexedNormalSet3D;

	quadrilaterals->clear();

	// Compute the angle formed by the two vectors of the basis
	const Scalar alpha =
	      (base_3D_[1].pos() - base_3D_[0].pos()).normalized().dot(
	      (base_3D_[3].pos() - base_3D_[2].pos()).normalized());

	// 1. Datastructure construction
	const Scalar eps = pcfunctor_.getNormalizedEpsilon(distance_threshold);

	IndexedNormalSet3D nset (eps);

	for (size_t i = 0; i <  P_pairs.size(); ++i) {

		const VectorType p1 = pcfunctor_.points[P_pairs[i].first];
		const VectorType p2 = pcfunctor_.points[P_pairs[i].second];

		const VectorType  n  = (p2 - p1).normalized();

		nset.addElement((p1+ Scalar(invariant1) * (p2 - p1)).eval(), n, i);
	}


	std::set< std::pair<unsigned int, unsigned int > > comb;	

	unsigned int j = 0;
	std::vector<unsigned int> nei;

	// 2. Query time
	for (unsigned int i = 0; i < Q_pairs.size(); ++i) {

		const VectorType p1 = pcfunctor_.points[Q_pairs[i].first];
		const VectorType p2 = pcfunctor_.points[Q_pairs[i].second];

		const VectorType pq1 = point3d_model[Q_pairs[i].first].pos();
		const VectorType pq2 = point3d_model[Q_pairs[i].second].pos();

		nei.clear();

		const VectorType query = p1 + invariant2 * ( p2 - p1 );
		const VectorType queryQ = pq1 + invariant2 * (pq2 - pq1);

		const VectorType queryn = (p2 - p1).normalized();

		nset.getNeighbors( query, queryn, alpha, nei);

		VectorType invPoint;
		for (unsigned int k = 0; k != nei.size(); k++){
	  		const int id = nei[k];

			const VectorType& pp1 = point3d_model[P_pairs[id].first].pos();
			const VectorType& pp2 = point3d_model[P_pairs[id].second].pos();

			invPoint = pp1 + (pp2 - pp1) * invariant1;

			// use also distance_threshold for inv 1 and 2 in 4PCS
			if ((queryQ-invPoint).squaredNorm() <= distance_threshold){
	      		comb.emplace(id, i);
	  		}
		}
	}

	for (std::set< std::pair<unsigned int, unsigned int > >::const_iterator it = comb.cbegin(); it != comb.cend(); it++) {
	    const unsigned int & id = (*it).first;
	    const unsigned int & i  = (*it).second;

	    quadrilaterals->emplace_back(P_pairs[id].first, P_pairs[id].second,
	                                 Q_pairs[i].first,  Q_pairs[i].second);
	}

	return quadrilaterals->size() != 0;
}

bool 
stocs_estimator::get_rigid_transform_from_congruent_pair(std::vector<int> &base_indices,
        												Quadrilateral &congruent_quad,
        												int base_index){

  std::array<std::pair<Point3D, Point3D>,4> congruent_points;

  // get references to the basis coordinates
  const Point3D& b1 = point3d_scene[base_indices[0]];
  const Point3D& b2 = point3d_scene[base_indices[1]];
  const Point3D& b3 = point3d_scene[base_indices[2]];
  const Point3D& b4 = point3d_scene[base_indices[3]];

  // Centroid of the basis, computed once and using only the three first points
  VectorType centroid1 = (b1.pos() + b2.pos() + b3.pos()) / Scalar(3);

  // Centroid of the sets, computed in the loop using only the three first points
  VectorType centroid2;

  // set the basis coordinates in the congruent quad array
  congruent_points[0].first = b1;
  congruent_points[1].first = b2;
  congruent_points[2].first = b3;
  congruent_points[3].first = b4;

  Eigen::Matrix<Scalar, 4, 4> transform;

  const int a = congruent_quad.vertices[0];
  const int b = congruent_quad.vertices[1];
  const int c = congruent_quad.vertices[2];
  const int d = congruent_quad.vertices[3];
  congruent_points[0].second = point3d_model[a];
  congruent_points[1].second = point3d_model[b];
  congruent_points[2].second = point3d_model[c];
  congruent_points[3].second = point3d_model[d];

  centroid2 = (congruent_points[0].second.pos() +
               congruent_points[1].second.pos() +
               congruent_points[2].second.pos()) / Scalar(3.);

  Scalar rms = -1;

  const bool ok =
  ComputeRigidTransformation(congruent_points,   // input congruent quads
                             centroid1,          // input: basis centroid
                             centroid2,          // input: candidate quad centroid
                             transform,          // output: transformation
                             rms,                // output: rms error of the transformation between the basis and the congruent quad
                             false
                             );             // state: compute scale ratio ?

  if(ok && rms >= Scalar(0.)) {
    all_transforms.push_back(transform);

    Eigen::Matrix4f transformation;
    transformation = transform;
    // The transformation has been computed between the two point clouds centered
    // at the origin, we need to recompute the translation to apply it to the original clouds
    {
        Eigen::Matrix<Scalar, 3, 3> rot, scale;
        Eigen::Transform<Scalar, 3, Eigen::Affine> (transformation).computeRotationScaling(&rot, &scale);
        transformation.col(3) = (centroid1 + centroid_scene_ - ( rot * scale * (centroid2 + centroid_model_))).homogeneous();
    }

    // score is not computed at this time...so set to 0
    PoseCandidate* pose = new PoseCandidate(transformation, 0, base_index);
    all_pose.push_back(pose);
  }

  return true;
}

void
stocs_estimator::centroid_shift() {
	centroid_scene_ = VectorType::Zero();
	centroid_model_ = VectorType::Zero();

	for (int i = 0; i < point3d_scene.size(); ++i) {
        centroid_scene_ += point3d_scene[i].pos();
    }
    for (int i = 0; i < point3d_model.size(); ++i) {
        centroid_model_ += point3d_model[i].pos();
    }

    centroid_scene_ /= Scalar(point3d_scene.size());
    centroid_model_ /= Scalar(point3d_model.size());

    for (int i = 0; i < point3d_scene.size(); ++i) {
        point3d_scene[i].pos() -= centroid_scene_;
    }
    for (int i = 0; i < point3d_model.size(); ++i) {
        point3d_model[i].pos() -= centroid_model_;
    }
}

void 
stocs_estimator::kdtree_initialize() {

	size_t number_of_points_scene = point3d_scene.size();
	std::cout << "|S|: " << number_of_points_scene << std::endl;

	// Build the kdtree.
	kd_tree_ = Super4PCS::KdTree<Scalar>(number_of_points_scene);

	for (size_t i = 0; i < number_of_points_scene; ++i) {
		kd_tree_.add(point3d_scene[i].pos());
	}

	kd_tree_.finalize();
}

void
stocs_estimator::compute_best_transform() {
	
	std::cout << "Transforms to verify: " << all_transforms.size() << std::endl;

	Scalar max_score = 0;
	int index = -1;

	for (int i = 0; i < all_transforms.size(); i++) {
		Scalar lcp = compute_alignment_score_for_rigid_transform(all_transforms[i]);
		all_pose[i]->lcp = lcp;

		if (lcp > max_score){
			max_score = lcp;
			index = i;
		}
	}

	best_lcp = max_score;
	best_index = index;

	std::cout << "best index: " << best_index << ", maximum score: " << best_lcp << std::endl;
}

Scalar
stocs_estimator::compute_alignment_score_for_rigid_transform(const Eigen::Ref<const MatrixType> &mat) {

  	// We allow factor 2 scaling in the normalization.
	const Scalar epsilon = distance_threshold;
	float weighted_match = 0;

	const size_t number_of_points_model = point3d_model.size();
	const Scalar sq_eps = epsilon*epsilon;

	for (int i = 0; i < number_of_points_model; ++i) {

		// Use the kdtree to get the nearest neighbor
		Super4PCS::KdTree<Scalar>::Index resId = 
			kd_tree_.doQueryRestrictedClosestIndex(
		            (mat * point3d_model[i].pos().homogeneous()).head<3>(),
		            sq_eps);

		if ( resId != Super4PCS::KdTree<Scalar>::invalidIndex() ) {

		    VectorType n_q = mat.block<3,3>(0,0)*point3d_model[i].normal();

		    float angle_n = std::acos(point3d_scene[resId].normal().dot(n_q))*180/M_PI;
		    
		    // angle_n = std::min(angle_n, float(fabs(180-angle_n)));
		    
		    if(angle_n < 30){
		      weighted_match += point3d_scene[resId].class_probability();
		    }

		}
	}

  return weighted_match / Scalar(number_of_points_model);

}

} //namespace stocs