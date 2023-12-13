#ifndef EIGEN_HVX_GENERAL_BLOCK_KERNEL_H
#define EIGEN_HVX_GENERAL_BLOCK_KERNEL_H

// Only support 128B HVX now.
// Floating-point operations are only supported since V68.
#if defined __HVX__ && (__HVX_LENGTH__ == 128) && __HVX_ARCH__ >= 68

namespace Eigen {
namespace internal {

}  // end namespace internal
}  // end namespace Eigen

#endif  // __HVX__ && (__HVX_LENGTH__ == 128) && __HVX_ARCH__ >= 68

#endif  // EIGEN_HVX_GENERAL_BLOCK_KERNEL_H
