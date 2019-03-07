#include <iostream>
#include <fstream>

#include <rgbd.hpp>

namespace rgbd {

void
load_ply_model(PCLPointCloud::Ptr cloud, 
              std::vector<Point3D>& point3d,
              float scale) {
  
  typename Point3D::VectorType n;
  typename Point3D::VectorType rgb;

  for(auto v: cloud->points) {
    // check if normal is finite
    if(std::isfinite(v.normal[0]) && std::isfinite(v.normal[1]) && std::isfinite(v.normal[2])) {
      point3d.emplace_back( v.x*scale, v.y*scale, v.z*scale);
      
      n << v.normal[0], v.normal[1], v.normal[2];
      rgb << v.r, v.g, v.b;

      //normalizes and sets the normal
      point3d.back().set_normal(n);
      point3d.back().set_rgb(rgb);
    }
  }
}

void
save_as_ply(std::string location, 
            std::vector<Point3D>& point3d,
            float scale) {

  PCLPointCloud::Ptr cloud (new PCLPointCloud);
  for(auto v: point3d) {
    pcl::PointXYZRGBNormal p;
    p.x = v.x()*scale; 
    p.y = v.y()*scale; 
    p.z = v.z()*scale;
    p.normal[0] = v.normal()[0];
    p.normal[1] = v.normal()[1];
    p.normal[2] = v.normal()[2];
    p.r = v.rgb()[0]; 
    p.g = v.rgb()[1]; 
    p.b = v.rgb()[2];

    cloud->points.push_back(p);
  }
  pcl::io::savePLYFile(location, *cloud);
}

void
transform_pointset(std::vector<Point3D>& input,
                  std::vector<Point3D>& output,
                  Eigen::Matrix<Point3D::Scalar, 4, 4> &transform) {

  for (int i = 0; i < input.size(); ++i) { 
    auto pt = (transform * input[i].pos().homogeneous()).head<3>();
    Point3D new_pt(pt[0], pt[1], pt[2]);

    output.push_back(new_pt);
  }
  
}

void
compute_normal_pcl(PCLPointCloud::Ptr cloud,
                  float radius) {

  pcl::NormalEstimation<pcl::PointXYZRGBNormal, pcl::PointXYZRGBNormal> ne;
  pcl::search::KdTree<pcl::PointXYZRGBNormal>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZRGBNormal> ());
  ne.setInputCloud (cloud);
  ne.setSearchMethod (tree);
  ne.setRadiusSearch (radius);
  ne.compute (*cloud);

}

int 
ppf_closest_bin(int value, int discretization) {

  int lower_limit = value - (value % discretization);
  int upper_limit = lower_limit + discretization;

  int dist_from_lower = value - lower_limit;
  int dist_from_upper = upper_limit - value;

  int closest = (dist_from_lower < dist_from_upper)? lower_limit:upper_limit;

  return closest;
}

void 
ppf_compute(Point3D point_1, 
            Point3D point_2, 
            float tr_discretization,
            float rot_discretization,
            std::vector<int> &ppf_) {

  typename Point3D::VectorType p1 = point_1.pos();
  typename Point3D::VectorType p2 = point_2.pos();
  typename Point3D::VectorType n1 = point_1.normal();
  typename Point3D::VectorType n2 = point_2.normal();
  typename Point3D::VectorType u = p1 - p2;

  int ppf_1 = int(u.norm()*1000);
  int ppf_2 = int(atan2(n1.cross(u).norm(), n1.dot(u))*180/M_PI);
  int ppf_3 = int(atan2(n2.cross(u).norm(), n2.dot(u))*180/M_PI);
  int ppf_4 = int(atan2(n1.cross(n2).norm(), n1.dot(n2))*180/M_PI);

  ppf_.push_back(ppf_closest_bin(ppf_1, tr_discretization));
  ppf_.push_back(ppf_closest_bin(ppf_2, rot_discretization));
  ppf_.push_back(ppf_closest_bin(ppf_3, rot_discretization));
  ppf_.push_back(ppf_closest_bin(ppf_4, rot_discretization));
}

void 
ppf_map_insert(std::map<std::vector<int>, std::vector<std::pair<int, int> > > &PPFMap,
              std::vector<int> ppf_,
              float tr_discretization,
              float rot_discretization,
              std::pair<int, int> paired_index){

  for (int p1 = ppf_[0] - tr_discretization; p1 < ppf_[0] + tr_discretization; p1 += tr_discretization)
    for (int p2 = ppf_[1] - 2*rot_discretization; p2 < ppf_[1] + 2*rot_discretization; p2 += rot_discretization)
      for (int p3 = ppf_[2] - 2*rot_discretization; p3 < ppf_[2] + 2*rot_discretization; p3 += rot_discretization)
        for (int p4 = ppf_[3] - 2*rot_discretization; p4 < ppf_[3] + 2*rot_discretization; p4 += rot_discretization) {
          
          // distances less than 5mm are not allowed to be sampled
          if(p1 <= 5 || p2 < 0 || p3 < 0 || p4 < 0)
            continue;

          std::vector<int> temp_ppf_ = {p1, p2, p3, p4};

          auto it = PPFMap.find(temp_ppf_);
          if (it == PPFMap.end()){

            std::vector<std::pair<int,int> > indices;
            indices.push_back(paired_index);
            PPFMap.insert (std::pair<std::vector<int>, std::vector<std::pair<int, int> > >(temp_ppf_,indices));
          }
          else{

            it->second.push_back(paired_index);
          }

        }
}

void
save_ppf_map(std::string location,
            std::map<std::vector<int>, std::vector<std::pair<int, int> > > &ppf_map) {

  auto it = ppf_map.begin();
  while (it!=ppf_map.end()){
    std::ofstream ptr_file;
    ptr_file.open (location, std::ofstream::out | std::ofstream::app);
    ptr_file << it->first[0] << " " << it->first[1] << " " << it->first[2] << " " << it->first[3] << " " << it->second.size() << " ";
    
    for (int ii=0; ii<it->second.size(); ii++) {
      ptr_file << it->second[ii].first << " " << it->second[ii].second << " ";
    }

    ptr_file << '\n';
    ptr_file.close();
    it++;
  }
}

void
load_ppf_map(std::string ppf_map_location,
            std::map<std::vector<int>, std::vector<std::pair<int, int> > > &ppf_map) {

  std::ifstream ppf_file;
  std::vector<int> ppf_feature(4);
  std::vector<std::pair<int,int> > index_pairs;
  int index1, index2;
  int pair_count;

  ppf_file.open (ppf_map_location, std::ofstream::in);

  while(ppf_file >> ppf_feature[0] >> ppf_feature[1] >> ppf_feature[2] >> ppf_feature[3] >> pair_count){
    index_pairs.clear();
    for(int ii=0; ii<pair_count; ii++){
      ppf_file >> index1 >> index2;
      index_pairs.push_back(std::make_pair(index1, index2));
    }
    ppf_map.insert (std::pair<std::vector<int>, std::vector<std::pair<int,int> > >(ppf_feature, index_pairs));
  }
  ppf_file.close();
}

void
load_rgbd_data_sampled(std::string rgb_location,
                  std::string depth_location,
                  std::string class_probability_map_location,
                  cv::Mat& edge_probability_map,
                  std::vector<float> camera_intrinsics,
                  float depth_scale,
                  float voxel_size,
                  float class_probability_threshold,
                  std::vector<Point3D>& point3d) {

  cv::Mat rgb_image;
  cv::Mat depth_image;
  cv::Mat class_probability_map;
  cv::Mat surface_normals;
  cv::Mat_<cv::Vec3f> surface_normals3f;
  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZRGB>);

  rgb_image = cv::imread(rgb_location, CV_LOAD_IMAGE_COLOR);
  depth_image = cv::imread(depth_location, CV_16UC1);
  class_probability_map = cv::imread(class_probability_map_location, CV_16UC1);

  // compute surface normals
  cv::Mat K = (cv::Mat_<double>(3, 3) << camera_intrinsics[0], 0, camera_intrinsics[1], 0, camera_intrinsics[2], camera_intrinsics[3], 0, 0, 1);
  cv::rgbd::RgbdNormals normals_computer(depth_image.rows, depth_image.cols, CV_32F, K, 5, cv::rgbd::RgbdNormals::RGBD_NORMALS_METHOD_LINEMOD);

  normals_computer(depth_image, surface_normals);
  surface_normals.convertTo(surface_normals3f, CV_32FC3);

  for (int i = 0; i < depth_image.rows; i++) {
    for (int j = 0; j < depth_image.cols; j++) {

        float depth = (float)depth_image.at<unsigned short>(i,j)*depth_scale;

        pcl::PointXYZRGB pt;
        pt.x = (float)((j - camera_intrinsics[1]) * depth / camera_intrinsics[0]);
        pt.y = (float)((i - camera_intrinsics[3]) * depth / camera_intrinsics[2]);
        pt.z = depth;

        cv::Vec3b rgb_val = rgb_image.at<cv::Vec3b>(i,j);
        uint32_t rgb = ((uint32_t)rgb_val.val[2] << 16 | (uint32_t)rgb_val.val[1] << 8 | (uint32_t)rgb_val.val[0]);
        pt.rgb = *reinterpret_cast<float*>(&rgb);

        cloud->points.push_back(pt);

      }
  }  

  pcl::VoxelGrid<pcl::PointXYZRGB> sor;
  sor.setInputCloud (cloud);
  sor.setLeafSize (voxel_size, voxel_size, voxel_size);
  sor.filter (*cloud);

  pcl::RadiusOutlierRemoval<pcl::PointXYZRGB> outrem;
  outrem.setInputCloud(cloud);
  outrem.setRadiusSearch(2*voxel_size + 0.005);
  outrem.setMinNeighborsInRadius (10);
  outrem.filter (*cloud);

  for(auto pt: cloud->points) {
    typename Point3D::VectorType n;
    typename Point3D::VectorType rgb;
    
    if(std::isnan(pt.z) || pt.z <= 0 || pt.z > 2.0)
      continue;
    
    Eigen::Matrix3f mat;
    mat << camera_intrinsics[0], 0, camera_intrinsics[1],
            0, camera_intrinsics[2], camera_intrinsics[3],
            0, 0, 1;

    Eigen::Vector3f point2D = mat * Eigen::Vector3f(pt.x, pt.y, pt.z);
    int col = point2D[0]/point2D[2];
    int row = point2D[1]/point2D[2];

    float class_probability = (float)class_probability_map.at<unsigned short>(row,col)*(1.0/10000);
    float edge_probability = (float)(255.0 - edge_probability_map.at<unsigned char>(row,col))/255.0;

    if(class_probability < class_probability_threshold)
      continue;

    cv::Vec3f cv_normal = surface_normals3f(row, col);
    cv::Vec3b cv_color = rgb_image.at<cv::Vec3b>(row, col);

    // surface normal invalidation conditions
    if(std::isnan(cv_normal[0]) || std::isnan(cv_normal[1]) || std::isnan(cv_normal[2]))
      continue;
    if(cv_normal[0] == 0 && cv_normal[1] == 0 && cv_normal[2] == 0)
      continue;

    point3d.emplace_back( pt.x, pt.y, pt.z);
    n << cv_normal[0], cv_normal[1], cv_normal[2];
    rgb << cv_color.val[2], cv_color.val[1], cv_color.val[0];

    //normalizes and sets the normal
    point3d.back().set_normal(n);
    point3d.back().set_rgb(rgb);
    point3d.back().set_pixel(std::make_pair(row, col));

    point3d.back().set_probability(class_probability, edge_probability);
  }

} // function: load_rgbd_data

void
visualize_heatmap(std::vector<Point3D>& point3d,
                  std::vector<int> samples,
                  std::string save_location,
                  int image_width, int image_height,
                  int block_size){
  cv::Mat heatmap;
  cv::Mat input = cv::Mat::zeros (image_height, image_width, CV_8UC1);
  for(auto p: point3d) {
    int u = p.pixel().first;
    int v = p.pixel().second;

    for (int i = u - block_size/2; i <= u + block_size/2; i++)
      for (int j = v - block_size/2; j <= v + block_size/2; j++)

        if (i >= 0 && i < image_height && j >= 0 && j < image_width) {
          input.at<unsigned char>(i,j) = (unsigned char)(p.probability()*255.0);
        }
      
  }

  cv::applyColorMap(input, heatmap, cv::COLORMAP_JET);

  for (int s: samples) {
    cv::Point p(point3d[s].pixel().second, point3d[s].pixel().first);
    circle(heatmap, p, 8, cv::Scalar( 0, 255, 0 ), 2);
  }
  
  cv::imwrite(save_location, heatmap);
}

void
generate_segmentation_mask(Point3D& p, 
                           cv::Mat edge_probability_map,
                           float max_distance,
                           cv::Mat& closed_list,
                           cv::Mat& segmentation_buffer,
                           int base_num,
                           std::string debug_location) {
    
    int image_width = edge_probability_map.cols;
    int image_height = edge_probability_map.rows;

    auto loc = p.pixel();
    int segment_index = segmentation_buffer.at<unsigned char>(loc.first, loc.second);
    if (segment_index != 0) {

      closed_list = cv::imread(debug_location + "/seg_mask_" + std::to_string(segment_index) + ".png", CV_8UC1);
      return;
    }

    std::queue<std::pair<int, int> > open_list;
    open_list.push(p.pixel());

    while (!open_list.empty()) {

      auto curr = open_list.front();
      closed_list.at<unsigned char>(curr.first, curr.second) = 255;
      segmentation_buffer.at<unsigned char>(curr.first, curr.second) = base_num;
      
      open_list.pop();

      // add neighbors
      int n_dist = 1;
      for(int i=curr.first - n_dist; i <= curr.first + n_dist; i += n_dist) {
        for(int j = curr.second - n_dist; j <= curr.second + n_dist; j += n_dist) {
          
          if(i < 0 || j < 0 || i >= image_height || j >= image_width)
            continue;

          float edge_probability = (float)(255.0 - edge_probability_map.at<unsigned char>(i,j))/255.0;
          int expanded = (int)closed_list.at<unsigned char>(i,j);
          float dist = std::sqrt(std::pow((p.pixel().first - i), 2) + std::pow((p.pixel().second - j), 2));

          if(expanded == 0 && edge_probability == 0 && dist < max_distance) {
            open_list.push(std::make_pair(i,j));
            closed_list.at<unsigned char>(i, j) = 255;
            segmentation_buffer.at<unsigned char>(i, j) = base_num;
          }

        }
      }
      
    } // End While
}

void
generate_local_segmentation_mask(Point3D& p, 
                           cv::Mat edge_probability_map,
                           float max_distance,
                           cv::Mat& closed_list) {
    typedef struct{
      std::pair<int, int> pixel;
      float depth;
    } Node;

    int image_width = edge_probability_map.cols;
    int image_height = edge_probability_map.rows;

    Node start;
    start.pixel = p.pixel();
    start.depth = 0;

    std::queue<Node> open_list;
    open_list.push(start);

    while (!open_list.empty()) {

      auto curr = open_list.front();
      closed_list.at<unsigned char>(curr.pixel.first, curr.pixel.second) = 255;
      
      open_list.pop();

      // add neighbors
      for(int i=curr.pixel.first-1; i<=curr.pixel.first+1; i++) {
        for(int j=curr.pixel.second-1; j<=curr.pixel.second+1; j++) {
          
          if(i < 0 || j < 0 || i >= image_height || j >= image_width)
            continue;

          float edge_probability = (float)(255.0 - edge_probability_map.at<unsigned char>(i,j))/255.0;
          int expanded = (int)closed_list.at<unsigned char>(i,j);
          float dist = std::sqrt(std::pow((p.pixel().first - i), 2) + std::pow((p.pixel().second - j), 2));
          float curr_depth = curr.depth + 1;

          if(expanded == 0 && edge_probability == 0 && dist < max_distance && curr_depth < 100) {
            Node neighbor;
            neighbor.pixel = std::make_pair(i,j);
            neighbor.depth = curr.depth + 1;
            open_list.push(neighbor);
            closed_list.at<unsigned char>(i, j) = 255;
          }

        }
      }
      
    } // End While
}

} // namespace rgbd