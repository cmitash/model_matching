#ifndef _SAMPLER_
#define _SAMPLER_

#include <vector>
#include <array>
#include "point3d.hpp"

namespace rgbd {

struct UniformDistSampler{
private:
    template <typename _Scalar>
    class HashTable {
    public:
        using Scalar = _Scalar;

    private:
        const uint64_t MAGIC1 = 100000007;
        const uint64_t MAGIC2 = 161803409;
        const uint64_t MAGIC3 = 423606823;
        const uint64_t NO_DATA = 0xffffffffu;
        Scalar voxel_;
        Scalar scale_;
        using VoxelType = std::array<int,3>;
        std::vector<VoxelType> voxels_;
        std::vector<uint64_t> data_;

    public:
        HashTable(int maxpoints, Scalar voxel) : voxel_(voxel), scale_(1.0f / voxel) {
            uint64_t n = maxpoints;
            voxels_.resize(n);
            data_.resize(n, NO_DATA);
        }
        template <typename Point>
        uint64_t& operator[](const Point& p) {
            // TODO: use eigen power here.
            VoxelType c {int(floor(p.x() * scale_)),
                         int(floor(p.y() * scale_)),
                         int(floor(p.z() * scale_))};

            uint64_t key = (MAGIC1 * c[0] + MAGIC2 * c[1] + MAGIC3 * c[2]) % data_.size();
            while (1) {
                if (data_[key] == NO_DATA) {
                    voxels_[key] = c;
                    break;
                } else if (voxels_[key] == c) {
                    break;
                }
                key++;
                if (key == data_.size()) key = 0;
            }
            return data_[key];
        }
    };
public:
    template <typename Point>
    inline
    void operator() (const std::vector<Point>& inputset,
                     const float delta,
                     std::vector<Point>& output) const {
      int num_input = inputset.size();
      output.clear();
      HashTable<typename Point::Scalar> hash(num_input, delta);
      for (int i = 0; i < num_input; i++) {
        uint64_t& ind = hash[inputset[i]];
        if (ind >= num_input) {
          output.push_back(inputset[i]);
          ind = output.size();
        }
      }
    }
}; // UniformDistSampler

} // namespace rgbd

#endif