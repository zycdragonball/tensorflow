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
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/bounds_check.h"

namespace tensorflow{

int64 fetch_input_offset(int64 output_offset,
                         int64 group_input_size,
                         int64 group_output_size,
                         int64 batch_size,
                         Eigen::Tensor<int32, 1> batch_index) {
  int64 group_offset = output_offset / group_output_size;
  int64 batch_offset = batch_index((output_offset%group_output_size) / batch_size);
  int64 elem_offset = output_offset % batch_size;
  
  return group_offset*group_input_size + batch_offset*batch_size + elem_offset;
}

Eigen::Tensor<int32, 1> fetch_batch_index(
    typename TTypes<int32>::ConstFlat repeats_flat,
    int size) {
  Eigen::Tensor<int32, 1> result({size});
  int ptr = 0;
  const int N = repeats_flat.size();
  for (int i = 0; i < N; ++i) {
    int repeat = repeats_flat(i);
    for (int j = 0; j < repeat; ++j) {
      result(ptr) = i;
      ptr++;
    }
  }
  
  return result;
}

template <typename T>
void RepeatCPUImpl(const Tensor& input,
                   typename TTypes<int32>::ConstFlat repeats_flat,
                   int axis, Tensor* output, int old_dim, int new_dim) {
  auto input_flat = input.flat<T>();
  auto output_flat = output->flat<T>();
  
  //A batch is inner dimensions < axis
  //A group is inner dimensions <= axis
  int64 batch_size = 1;
  int32 dims = input.shape().dims();
  for (int32 i = axis + 1; i < dims; ++i) {
    batch_size *= input.shape().dim_size(i);
  }
  
  int64 out_num_elem = output->NumElements();
  if (repeats_flat.size() == 1) {
    for (int64 i = 0; i < out_num_elem; ++i) {
      int64 batch_offset = i / (batch_size*repeats_flat(0));
      int64 elem_offset = i % batch_size;
      output_flat(i) = input_flat(batch_offset*batch_size + elem_offset);
    }
  } else {
    auto batch_index = fetch_batch_index(repeats_flat, new_dim);
    int64 group_input_size = batch_size * old_dim;
    int64 group_output_size = batch_size * new_dim;
    for (int64 i = 0; i < out_num_elem; ++i) {
      output_flat(i) = input_flat(fetch_input_offset(i, group_input_size,
                                                     group_output_size,
                                                     batch_size,
                                                     batch_index));
    }
  }
}

template <typename T>
class RepeatOp : public OpKernel {
 public:
  explicit RepeatOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("axis", &axis_));
  }
  
  void Compute(OpKernelContext* context) override {
    const Tensor& input = context->input(0);
    const Tensor& repeats = context->input(1);
    
    OP_REQUIRES(context, TensorShapeUtils::IsVector(repeats.shape()) ||
                         TensorShapeUtils::IsScalar(repeats.shape()),
                errors::InvalidArgument("`repeats` expects a scalar or a 1-D vector."));
    OP_REQUIRES(context, repeats.NumElements() == input.dim_size(axis_) ||
                         repeats.NumElements() == 1,
                errors::InvalidArgument(
                    "Expected `repeats` argument to be a vector of length ",
                    input.dim_size(axis_), " or 1, but got length ",
                    repeats.NumElements()));
    OP_REQUIRES(context, FastBoundsCheck(axis_, input.dims()),
                errors::InvalidArgument("Expected 0 <= `axis` < ", input.dims()));
    
    TensorShape output_shape = input.shape();
    auto repeats_flat = repeats.flat<int32>();
    const int old_dim = input.shape().dim_size(axis_);
    int new_dim = 0;
    if (repeats.NumElements() == 1) {
      new_dim = repeats_flat(0) * old_dim;
    } else {
      const int N = repeats_flat.size();
      for (int i = 0; i < N; ++i) {
        new_dim += repeats_flat(i);
      }
    }
    output_shape.set_dim(axis_, new_dim);
    
    Tensor* output = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    
    RepeatCPUImpl<T>(input, repeats_flat, axis_, output, old_dim, new_dim);    
  }
  
 private:
  int32 axis_;
  
};

//TODO: add gradient op


#define REGISTER_KERNEL(type)                    \
  REGISTER_KERNEL_BUILDER(                       \
      Name("Repeat")                             \
      .Device(DEVICE_CPU)                        \
      .TypeConstraint<type>("T"),                \
      RepeatOp<type>)

TF_CALL_ALL_TYPES(REGISTER_KERNEL);

#undef REGISTER_KERNEL
} //end namespace tensorflow
