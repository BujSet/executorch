/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#version 450 core

#define PRECISION ${PRECISION}

#define BUF_T ${buffer_scalar_type(DTYPE)}
#define VEC4_T ${texel_type(DTYPE)}
#define SCALAR_T ${texel_component_type(DTYPE)}

#include "indexing_utils.h"

$if DTYPE == "half":
  #extension GL_EXT_shader_16bit_storage : require

layout(std430) buffer;

layout(set = 0, binding = 0, ${IMAGE_FORMAT[DTYPE]}) uniform PRECISION restrict writeonly ${IMAGE_T[2][DTYPE]} image_out;
layout(set = 0, binding = 1) buffer  PRECISION restrict readonly Buffer {
  BUF_T buffer_in[];
};

layout(push_constant) uniform PRECISION restrict Block {
  ivec4 sizes;
  ivec4 original_sizes;
};

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

layout(constant_id = 3) const int packed_dim = C_DIM;

/*
 * Computes special prepacking for a depthwise convolution. Each shader invocation
 * calculates the input buffer location to read into the desired texel. This
 * packing was originally developed on CPU here:
 * https://github.com/pytorch/pytorch/blob/d63e7d0aa2e0a1b1fd7518f917224774afe97bae/aten/src/ATen/native/vulkan/ops/Convolution.cpp#L58-L118
 */
void main() {
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const ivec4 idx = to_tensor_idx(pos, sizes, packed_dim);

  if (any(greaterThanEqual(idx, sizes))) {
    return;
  }

  // Map tensor_idx to normal buffer_i
  const ivec4 p0 = tidx_to_nchwi(idx, sizes, packed_dim);

  // Compute modified tensor_idx by inverting the CPU function
  const int N = original_sizes.w;
  const int C = original_sizes.z;
  const int H = original_sizes.y;
  const int W = original_sizes.x;
  const int Y = sizes.y;

  const ivec4 p1 = p0 / W;
  const ivec4 p2 = p1 / H;

  const ivec4 n = (p2 % Y) * 4 + (p2 / Y);
  const ivec4 h = p1 % H;
  const ivec4 w = p0 % W;

  // Map modified tensor_idx to modifed buffer_i
  // Zero out if modified tensor idx is out of bounds
  const ivec4 buf_i = n * C*H*W + h * W + w;
  const bvec4 mask = bvec4(lessThan(n, ivec4(N)));

  VEC4_T texel = VEC4_T(0);
  if (mask.x) {
    texel.x = SCALAR_T(buffer_in[buf_i.x]);
  }
  if (mask.y) {
    texel.y = SCALAR_T(buffer_in[buf_i.y]);
  }
  if (mask.z) {
    texel.z = SCALAR_T(buffer_in[buf_i.z]);
  }
  if (mask.w) {
    texel.w = SCALAR_T(buffer_in[buf_i.w]);
  }

  imageStore(image_out, pos.xy, texel);
}
