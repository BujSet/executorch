/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cmath>
#include <cstring>

#include <executorch/kernels/portable/cpu/util/functional_util.h>
#include <executorch/runtime/kernel/kernel_includes.h>
#include <executorch/runtime/platform/assert.h>

namespace torch {
namespace executor {
namespace native {

using executorch::aten::Tensor;

Tensor& sign_out(KernelRuntimeContext& ctx, const Tensor& in, Tensor& out) {
  (void)ctx;

  // Resize for dynamic shape
  ET_KERNEL_CHECK_MSG(
      ctx,
      resize_tensor(out, in.sizes()) == Error::Ok,
      InvalidArgument,
      out,
      "Failed to resize output tensor.");

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_dim_order(in, out), InvalidArgument, out);

  ET_KERNEL_CHECK(
      ctx, tensors_have_same_shape_and_dtype(in, out), InvalidArgument, out);

  if (in.scalar_type() == executorch::aten::ScalarType::Bool) {
    memcpy(out.mutable_data_ptr(), in.const_data_ptr(), in.nbytes());
  } else {
    ET_SWITCH_REALHBF16_TYPES(in.scalar_type(), ctx, "sign.out", CTYPE, [&] {
      apply_unary_map_fn(
          [](const CTYPE val_in) {
            if (std::isnan(val_in)) {
              return val_in;
            } else {
              return static_cast<CTYPE>((val_in > 0) - (val_in < 0));
            }
          },
          in.const_data_ptr<CTYPE>(),
          out.mutable_data_ptr<CTYPE>(),
          in.numel());
    });
  }

  return out;
}

} // namespace native
} // namespace executor
} // namespace torch
