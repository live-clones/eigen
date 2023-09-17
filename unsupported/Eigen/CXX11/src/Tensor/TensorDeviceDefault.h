// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

#ifndef EIGEN_CXX11_TENSOR_TENSOR_DEVICE_DEFAULT_H
#define EIGEN_CXX11_TENSOR_TENSOR_DEVICE_DEFAULT_H


// IWYU pragma: private
#include "./InternalHeaderCheck.h"

namespace Eigen {

// Default device for the machine (typically a single cpu core)
struct DefaultDevice {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void* allocate(size_t num_bytes) const {
    return internal::aligned_malloc(num_bytes);
  }
  
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void deallocate(void* buffer) const {
    internal::aligned_free(buffer);
  }

  #ifdef __USING_SINGLE_TYPE_CONTRACTIONS__
  /**
   * This version assumes that all elements are of the same type T.
  */
  template<typename T>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void* allocate_elements(size_t num_elements) const {

    const size_t element_size = sizeof(T);
    size_t num_bytes = num_elements * element_size;
    size_t align = numext::maxi(EIGEN_MAX_ALIGN_BYTES, 1);
    num_bytes = divup<Index>(num_bytes, align) * align;

    void * result = allocate(num_bytes);

    if (NumTraits<T>::RequireInitialization) {
      char * mem_pos = reinterpret_cast<char*>(result);
      for (size_t i = 0; i < num_elements; ++i) {
        new(mem_pos)T();
        mem_pos += element_size;
      }
    }

    return result;
  }

  /**
   * This version assumes that all elements are of the same type T.
  */
  template<typename T>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void deallocate_elements(void* buffer, size_t num_elements) const {

    if (NumTraits<T>::RequireInitialization) {

      T * block = reinterpret_cast<T*>(buffer);

      for (size_t i = 0; i < num_elements; ++i) {
        T *element = &block[i];
        element->~T();
      }
    }

    deallocate(buffer);
  }
  #else

  /**
   * This function allocates the memory block for the scalars.
   * The scalars are initialized using the default constructor whether they need initialization.
   * A scalar S needs inialization if it is declared as NumTraits<S>::RequireInitialization
   * 
   * \param left_block_count number of elements in one left block
   * \param right_block_count number of elements in one right block
   * \param blocks container with the pointers of each block
   * \param num_left_blocks number of left blocks, used by the ThreadPoolDevice. Default is 1
   * \param num_right_blocks number of right blocks, used by the ThreadPoolDevice. Default is 1
   * \param num_slices number of slices, used by the ThreadPoolDevice. Default is 1
   * 
  */
  template<typename LEFT_T, typename RIGHT_T>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void* allocate_blocks(const size_t left_block_count,  
                                                       const size_t right_block_count, 
                                                       std::vector<void*> &blocks,      
                                                       const size_t num_left_blocks = 1, 
                                                       const size_t num_right_blocks = 1,
                                                       const size_t num_slices = 1
                                                       ) const {
    
    eigen_assert(blocks.empty());                     
    eigen_assert(left_block_count || right_block_count);   
    eigen_assert(num_left_blocks || num_right_blocks);  
    eigen_assert(num_slices);                                                      
    
    const size_t left_element_size = sizeof(LEFT_T);
    const size_t right_element_size = sizeof(RIGHT_T);
    size_t left_num_bytes = left_block_count * left_element_size;
    size_t right_num_bytes = right_block_count * right_element_size;

    size_t align = numext::maxi(EIGEN_MAX_ALIGN_BYTES, 1);
    left_num_bytes = divup<Index>(left_num_bytes, align) * align;
    right_num_bytes = divup<Index>(right_num_bytes, align) * align;

    size_t total_size_bytes = (left_num_bytes * num_left_blocks + right_num_bytes * num_right_blocks) * num_slices;

    void* result = allocate(total_size_bytes);

    char * mem_pos = reinterpret_cast<char*>(result);

    blocks.reserve(num_left_blocks * num_slices + num_right_blocks * num_slices);

    for (size_t slice = 0; slice < num_slices; ++slice) {

      for (size_t block = 0; block < num_left_blocks; ++block) {
        if (left_block_count > 0) {
          blocks.emplace_back(reinterpret_cast<void*>(mem_pos));
          if(NumTraits<LEFT_T>::RequireInitialization) {
            for (size_t i = 0; i < left_block_count; ++i) {
              new(mem_pos)LEFT_T();
              mem_pos += left_element_size;
            }
          } else {
            mem_pos += left_num_bytes;
          }
        } else {
          blocks.emplace_back(nullptr);
        }
      }
      for (size_t block = 0; block < num_right_blocks; ++block) {
        if (right_block_count > 0) {
          blocks.emplace_back(reinterpret_cast<void*>(mem_pos));
          if(NumTraits<RIGHT_T>::RequireInitialization) {
            for (size_t i = 0; i < right_block_count; ++i) {
              new(mem_pos)RIGHT_T();
              mem_pos += right_element_size;
            }
          } else {
            mem_pos += right_num_bytes;
          }
        } else {
          blocks.emplace_back(nullptr);
        }
      }

    }

    return result;
  }

  /**
   * This function deallocates the memory block scalars.
   * The scalars are finalized by the default destructor if they were initialized.
   * 
   * \param left_num_elements number of elements in one left block
   * \param right_num_elements number of elements in one right block
   * \param num_left_blocks number of left blocks, used by the ThreadPoolDevice. Default is 1
   * \param num_right_blocks number of right blocks, used by the ThreadPoolDevice. Default is 1
   * \param num_slices number of slices, used by the ThreadPoolDevice. Default is 1
   * 
  */
  template<typename LEFT_T, typename RIGHT_T>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void deallocate_blocks(void* buffer, 
                                                        const size_t left_num_elements, 
                                                        const size_t right_num_elements,
                                                        const size_t num_left_blocks = 1,
                                                        const size_t num_right_blocks = 1,
                                                        const size_t num_slices = 1
                                                        ) const {

    if (NumTraits<LEFT_T>::RequireInitialization || NumTraits<RIGHT_T>::RequireInitialization) {


      eigen_assert(left_num_elements || right_num_elements);   
      eigen_assert(num_left_blocks || num_right_blocks);  
      eigen_assert(num_slices);  

      const size_t left_element_size = sizeof(LEFT_T);
      const size_t right_element_size = sizeof(RIGHT_T);

      const size_t left_block_count = left_num_elements * num_left_blocks;
      const size_t right_block_count = right_num_elements * num_right_blocks;

      size_t left_num_bytes = left_block_count * left_element_size;
      size_t right_num_bytes = right_block_count * right_element_size;

      size_t align = numext::maxi(EIGEN_MAX_ALIGN_BYTES, 1);
      left_num_bytes = divup<Index>(left_num_bytes, align) * align; 
      right_num_bytes = divup<Index>(right_num_bytes, align) * align;

      char* mem_pos = reinterpret_cast<char*>(buffer);

      for (size_t slice = 0; slice < num_slices; ++slice) {

        if (NumTraits<LEFT_T>::RequireInitialization) {
          LEFT_T * left_block = reinterpret_cast<LEFT_T*>(mem_pos);
          size_t block_index = 0;

          for (size_t block = 0; block < num_left_blocks; ++block) {
            for (size_t i = 0; i < left_num_elements; ++i) {
              LEFT_T *element = &left_block[block_index++];
              element->~LEFT_T();
            }
          }
        }

        mem_pos += left_num_bytes;

        if (NumTraits<RIGHT_T>::RequireInitialization) {
          RIGHT_T * right_block = reinterpret_cast<RIGHT_T*>(mem_pos);
          size_t block_index = 0;

          for (size_t block = 0; block < num_right_blocks; ++block) {
            for (size_t i = 0; i < right_num_elements; ++i) {
              RIGHT_T *element = &right_block[block_index++];
              element->~RIGHT_T();
            }
          }
        }

        mem_pos += right_num_bytes;

      }
    }

    deallocate(buffer);
  }

  #endif

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void* allocate_temp(size_t num_bytes) const {
    return allocate(num_bytes);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void deallocate_temp(void* buffer) const {
    deallocate(buffer);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpy(void* dst, const void* src, size_t n) const {
    ::memcpy(dst, src, n);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpyHostToDevice(void* dst, const void* src, size_t n) const {
    memcpy(dst, src, n);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memcpyDeviceToHost(void* dst, const void* src, size_t n) const {
    memcpy(dst, src, n);
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void memset(void* buffer, int c, size_t n) const {
    ::memset(buffer, c, n);
  }
  template<typename T>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void fill(T* begin, T* end, const T& value) const {
#ifdef EIGEN_GPU_COMPILE_PHASE
    // std::fill is not a device function, so resort to simple loop.
    for (T* it = begin; it != end; ++it) {
      *it = value;
    }
#else
    std::fill(begin, end, value);
#endif
  }
  template<typename Type>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Type get(Type data) const { 
    return data;
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t numThreads() const {
#if !defined(EIGEN_GPU_COMPILE_PHASE)
    // Running on the host CPU
    return 1;
#elif defined(EIGEN_HIP_DEVICE_COMPILE)
    // Running on a HIP device
    return 64;
#else
    // Running on a CUDA device
    return 32;
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t firstLevelCacheSize() const {
#if !defined(EIGEN_GPU_COMPILE_PHASE) && !defined(SYCL_DEVICE_ONLY)
    // Running on the host CPU
    return l1CacheSize();
#elif defined(EIGEN_HIP_DEVICE_COMPILE)
    // Running on a HIP device
    return 48*1024; // FIXME : update this number for HIP
#else
    // Running on a CUDA device, return the amount of shared memory available.
    return 48*1024;
#endif
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE size_t lastLevelCacheSize() const {
#if !defined(EIGEN_GPU_COMPILE_PHASE) && !defined(SYCL_DEVICE_ONLY)
    // Running single threaded on the host CPU
    return l3CacheSize();
#elif defined(EIGEN_HIP_DEVICE_COMPILE)
    // Running on a HIP device
    return firstLevelCacheSize(); // FIXME : update this number for HIP
#else
    // Running on a CUDA device
    return firstLevelCacheSize();
#endif
  }
  
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void synchronize() const {
    // Nothing.  Default device operations are synchronous.
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE int majorDeviceVersion() const {
#if !defined(EIGEN_GPU_COMPILE_PHASE)
    // Running single threaded on the host CPU
    // Should return an enum that encodes the ISA supported by the CPU
    return 1;
#elif defined(EIGEN_HIP_DEVICE_COMPILE)
    // Running on a HIP device
    // return 1 as major for HIP
    return 1;
#else
    // Running on a CUDA device
    return EIGEN_CUDA_ARCH / 100;
#endif
  }
};

}  // namespace Eigen

#endif // EIGEN_CXX11_TENSOR_TENSOR_DEVICE_DEFAULT_H
