#include <stocs.hpp>

std::string repo_path = "/media/bowen/e25c9489-2f57-42dd-b076-021c59369fec/github/stocs";

// All values in m
float voxel_size = 0.01;
float normal_radius = 0.005;
float model_scale = 1.0;

// All values in mm
int ppf_tr_discretization = 5;
int ppf_rot_discretization = 5;

int
main (int argc, char** argv) {

	if (argc < 2) {
		std::cout << "Enter name of the object model!!" << std::endl;
		exit(-1);
	} 
	std::string object_name = argv[1];

	std::string model_path = repo_path + "/models/" + object_name;

	system(("rm -rf " + model_path + "/model_search.ply").c_str());
	system(("rm -rf " + model_path + "/ppf_map").c_str());

	stocs::pre_process_model(model_path + "/textured_vertices.ply",
							normal_radius,
							model_scale,
							1.0f,
							voxel_size,
							ppf_tr_discretization,
							ppf_rot_discretization,
							model_path + "/model_search.ply",
							model_path + "/ppf_map");

 	return 0;
}