/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow{

template <typename T>
void RepeatCPUImpl(const Tensor& input,
                   const typename TTypes<int32>::ConstFlat& repeats_flat,
                   int axis, Tensor* output);
                   
template <typename T>
void RepeatCPUImplV2(DeviceBase* d, const Tensor& input,
                     const typename TTypes<int32>::ConstFlat& repeats_flat,
                     int axis, int64 cost_per_unit, Tensor* output);

#if GOOGLE_CUDA
template <typename T>
void RepeatGPUImpl(const Eigen::GpuDevice& d, const Tensor& input,
                   const typename TTypes<int32>::ConstFlat& repeats_flat,
                   int axis, Tensor* output);
                   
#endif // GOOGLE_CUDA

} //end namespace tensorflow
