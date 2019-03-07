#ifndef _POINT_3D_
#define _POINT_3D_

#include <vector>
#include <iostream>
#include <fstream>
#include <array>

#include "Eigen/Core"

class Point3D {
 public:
  using Scalar = float; //_Scalar;
  using VectorType = Eigen::Matrix<Scalar, 3, 1>;

  inline Point3D(Scalar x, Scalar y, Scalar z) : pos_({ x, y, z}) {}
  inline Point3D(const Point3D& other):
      pos_(other.pos_),
      normal_(other.normal_),
      pixel_(other.pixel_),
      class_probability_(other.class_probability_),
      edge_probability_ (other.edge_probability_),
      current_probability_(other.current_probability_),
      rgb_(other.rgb_) {}
  template<typename Scalar>
  explicit inline Point3D(const Eigen::Matrix<Scalar, 3, 1>& other):
      pos_({ other(0), other(1), other(2) }){
  }

  inline Point3D() {}
  inline VectorType& pos() { return pos_ ; }
  inline const VectorType& pos() const { return pos_ ; }
  inline const VectorType& rgb() const { return rgb_; }
  inline const float& probability() const { return current_probability_; }
  inline const float& class_probability() const { return class_probability_; }

  inline const std::pair<int, int>& pixel() const { return pixel_; }

  inline const VectorType& normal() const { return normal_; }
  inline void set_rgb(const VectorType& rgb) {
      rgb_ = rgb;
  }
  inline void set_normal(const VectorType& normal) {
      normal_ = normal.normalized();
  }
  inline void set_pixel(const std::pair<int, int> &pixel) {
      pixel_ = pixel;
  }
  inline void set_probability(float class_probability, float edge_probability) {
      class_probability_ = class_probability;
      edge_probability_ = edge_probability;
      current_probability_ = class_probability;
  }
  inline void update_class_probability(float decay_fraction) {
      class_probability_ = decay_fraction*class_probability_;
  }
  inline void update_probability(float new_probability) {
      current_probability_ = new_probability;
  }
  inline void reset_probability() {
      current_probability_ = class_probability_;
  }
  inline void normalize() {
    pos_.normalize();
  }
  inline bool hasColor() const { return rgb_.squaredNorm() > Scalar(0.001); }

  Scalar& x() { return pos_.coeffRef(0); }
  Scalar& y() { return pos_.coeffRef(1); }
  Scalar& z() { return pos_.coeffRef(2); }

  Scalar x() const { return pos_.coeff(0); }
  Scalar y() const { return pos_.coeff(1); }
  Scalar z() const { return pos_.coeff(2); }



 private:
  // Normal.
  VectorType pos_{0.0f, 0.0f, 0.0f};
  // Normal.
  VectorType normal_{0.0f, 0.0f, 0.0f};
  // Color.
  VectorType rgb_{-1.0f, -1.0f, -1.0f};
  // Corresponding pixel
  std::pair<int, int> pixel_;
  // Class and edge probability
  float class_probability_ = 0;
  float edge_probability_ = 0;
  float current_probability_ = 0;

};

// struct Base {
//     std::array <int, 4> vertices;
//     float invariant1, invariant2;
//     std::vector<>

//     inline Base(int vertex0, int vertex1, int vertex2, int vertex3) {
//         vertices = { vertex0, vertex1, vertex2, vertex3 };
//         invariant1 = 0; invariant2 = 0;
//     }

//     inline bool operator== (const Base& rhs) const {
//         return  vertices[0] == rhs[0] &&
//                 vertices[1] == rhs[1] &&
//                 vertices[2] == rhs[2] &&
//                 vertices[3] == rhs[3];
//     }

//     int  operator[](int idx) const { return vertices[idx]; }
//     int& operator[](int idx)       { return vertices[idx]; }
// };

// Holds congruent sets from model
struct Quadrilateral {
    std::array <int, 4> vertices;

    inline Quadrilateral(int vertex0, int vertex1, int vertex2, int vertex3) {
        vertices = { vertex0, vertex1, vertex2, vertex3 };
    }

    inline bool operator< (const Quadrilateral& rhs) const {
        return    vertices[0] != rhs[0] ? vertices[0] < rhs[0]
                : vertices[1] != rhs[1] ? vertices[1] < rhs[1]
                : vertices[2] != rhs[2] ? vertices[2] < rhs[2]
                : vertices[3] < rhs[3];
    }

    inline bool operator== (const Quadrilateral& rhs) const {
        return  vertices[0] == rhs[0] &&
                vertices[1] == rhs[1] &&
                vertices[2] == rhs[2] &&
                vertices[3] == rhs[3];
    }

    int  operator[](int idx) const { return vertices[idx]; }
    int& operator[](int idx)       { return vertices[idx]; }
};

class PoseCandidate {
  public:
    Eigen::Matrix4f transform;
    float lcp;
    int base_index;

    PoseCandidate(Eigen::Matrix4f transform, float lcp, float base_index) {
      this->transform = transform;
      this->lcp = lcp;
      this->base_index = base_index;
    }

    ~PoseCandidate(){}

};

#endif