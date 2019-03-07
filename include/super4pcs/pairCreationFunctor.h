#ifndef PAIRCREATIONFUNCTOR_H
#define PAIRCREATIONFUNCTOR_H

#include <iostream>
#include <vector>

#include "accelerators/pairExtraction/bruteForceFunctor.h"
#include "accelerators/pairExtraction/intersectionFunctor.h"
#include "accelerators/pairExtraction/intersectionPrimitive.h"

static int approximate_bin(int val, int disc) {
  int lower_limit = val - (val % disc);
  int upper_limit = lower_limit + disc;

  int dist_from_lower = val - lower_limit;
  int dist_from_upper = upper_limit - val;

  int closest = (dist_from_lower < dist_from_upper)? lower_limit:upper_limit;

  return closest;
}

template <typename _Scalar>
struct PairCreationFunctor{

public:
  using Scalar      = _Scalar;
  using PairsVector = std::vector<std::pair<int, int>>;
  using VectorType  = typename Point3D::VectorType;

  // Processing data
  Scalar norm_threshold;
  double pair_normals_angle;
  double pair_distance;
  double pair_distance_epsilon;
  std::vector<int> ppf_;

  // Shared data
  const std::vector<Point3D>& Q_;

  PairsVector* pairs;

  std::vector<unsigned int> ids;


  // Internal data
  typedef Eigen::Matrix<Scalar, 3, 1> Point;
  typedef Super4PCS::Accelerators::PairExtraction::HyperSphere
  < typename PairCreationFunctor::Point, 3, Scalar> Primitive;

  std::vector< /*Eigen::Map<*/typename PairCreationFunctor::Point/*>*/ > points;
  std::vector< Primitive > primitives;

private:
  VectorType segment1;
  std::vector<Point3D> base_3D_;
  int base_point1_, base_point2_;

  typename PairCreationFunctor::Point _gcenter;
  Scalar _ratio;
  static const typename PairCreationFunctor::Point half;

public:
  inline PairCreationFunctor(
    const std::vector<Point3D>& Q)
    :Q_(Q),
     pairs(NULL), _ratio(1.f)
    { }

private:
  inline Point worldToUnit(
    const Eigen::MatrixBase<typename PairCreationFunctor::Point> &p) const {
    static const Point half = Point::Ones() * Scalar(0.5f);
    return (p-_gcenter) / _ratio + half;
  }


public:
  inline Point unitToWorld(
    const Eigen::MatrixBase<typename PairCreationFunctor::Point> &p) const {
    static const Point half = Point::Ones() * Scalar(0.5f);
    return (p - half) * _ratio + _gcenter;
  }


  inline Scalar unitToWorld( Scalar d) const {
    return d * _ratio;
  }


  inline Point getPointInWorldCoord(int i) const {
    return unitToWorld(points[i]);
  }


  inline void synch3DContent(){
    points.clear();
    primitives.clear();

    Super4PCS::AABB3D<Scalar> bbox;

    unsigned int nSamples = Q_.size();

    points.reserve(nSamples);
    primitives.reserve(nSamples);

    // Compute bounding box on fine data to be SURE to have all points in the
    // unit bounding box
    for (unsigned int i = 0; i < nSamples; ++i) {
        const VectorType &q = Q_[i].pos();
      points.push_back(q);
      bbox.extendTo(q);
    }

    _gcenter = bbox.center();
    // add a delta to avoid to have elements with coordinate = 1
    _ratio = std::max(bbox.depth() + 0.001,
             std::max(bbox.width() + 0.001,
                      bbox.height()+ 0.001));

    // update point cloud (worldToUnit use the ratio and gravity center
    // previously computed)
    // Generate primitives
    for (unsigned int i = 0; i < nSamples; ++i) {
      points[i] = worldToUnit(points[i]);

      primitives.emplace_back(points[i], Scalar(1.));
      ids.push_back(i);
    }

    // std::cout << "Work with " << points.size() << " points" << std::endl;
  }

  inline void setRadius(Scalar radius) {
    const Scalar nRadius = radius/_ratio;
    for(typename std::vector< Primitive >::iterator it = primitives.begin();
        it != primitives.end(); ++it)
      (*it).radius() = nRadius;
  }

  inline Scalar getNormalizedEpsilon(Scalar eps){
    return eps/_ratio;
  }

  inline void setBase( int base_point1, int base_point2,
                       const std::vector<Point3D>& base_3D){
    base_3D_     = base_3D;
    base_point1_ = base_point1;
    base_point2_ = base_point2;

    segment1 = (base_3D_[base_point2_].pos() -
                base_3D_[base_point1_].pos()).normalized();
  }


  inline void beginPrimitiveCollect(int /*primId*/){ }
  inline void endPrimitiveCollect(int /*primId*/){ }


  //! FIXME Pair filtering is the same than 4pcs. Need refactoring
  inline void process(int i, int j){
  }
};

#endif // PAIRCREATIONFUNCTOR_H
