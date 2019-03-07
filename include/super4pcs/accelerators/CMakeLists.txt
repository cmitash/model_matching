set(CMAKE_INCLUDE_CURRENT_DIR ON)

set(accel_ROOT "${SUPER4PCS_DIR}/accelerators")

#############################################

set(accel_INCLUDE
    ${accel_ROOT}/kdtree.h
    ${accel_ROOT}/pairExtraction/bruteForceFunctor.h
    ${accel_ROOT}/pairExtraction/intersectionFunctor.h
    ${accel_ROOT}/pairExtraction/intersectionNode.h
    ${accel_ROOT}/pairExtraction/intersectionPrimitive.h
    ${accel_ROOT}/normalset.h
    ${accel_ROOT}/normalset.hpp
    ${accel_ROOT}/bbox.h
    ${accel_ROOT}/utils.h)

add_library(super4pcs_accel STATIC ${accel_INCLUDE})
include_directories(${accel_ROOT})
set_target_properties(super4pcs_accel PROPERTIES LINKER_LANGUAGE CXX)

