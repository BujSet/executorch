#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Test the end-to-end flow of selective build, using 3 APIs:
# 1. Select all ops
# 2. Select from a list of ops
# 3. Select from a yaml file
# 4. (TODO) Select from a serialized model (.pte)
set -e

# shellcheck source=/dev/null
source "$(dirname "${BASH_SOURCE[0]}")/../../.ci/scripts/utils.sh"


# BUCK2 examples; test internally in fbcode/xplat
# 1. `--config executorch.select_ops=all`: select all ops from the dependency
#       kernel libraries, register all of them into ExecuTorch runtime.
# 2. `--config executorch.select_ops=list`: Only select ops from `ops` kwarg
#       in `et_operator_library` macro.
# 3. `--config executorch.select_ops=yaml`: Only select from a yaml file from
#       `ops_schema_yaml_target` kwarg in `et_operator_library` macro
# 4. `--config executorch.select_ops=dict`: Only select ops from `ops_dict`
#       kwarg in `et_operator_library` macro. Add `dtype_selective_build = True`
#       to executorch_generated_lib to select dtypes specified in the dictionary.

# Other configs:
# - `--config executorch.max_kernel_num=N`: Only allocate memory for the
#       required number of operators. Users can retrieve N from `selected_operators.yaml`.


# CMake examples; test in OSS. Check the README for more information.
test_cmake_select_all_ops() {
    local model_name=$1
    local model_export_name="${model_name}.pte"
    echo "Exporting ${model_name}"
    ${PYTHON_EXECUTABLE} -m examples.portable.scripts.export --model_name="${model_name}"
    local example_dir=examples/dtype_selective_build
    local build_dir=cmake-out/${example_dir}
    rm -rf ${build_dir}
    local start_time=`date +%s.%N`
    retry cmake -DCMAKE_BUILD_TYPE=Release \
            -DEXECUTORCH_SELECT_ALL_OPS=ON \
            -DCMAKE_INSTALL_PREFIX=cmake-out \
            -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
            -B${build_dir} \
            ${example_dir}

    echo "Building ${example_dir}"
    cmake --build ${build_dir} -j9 --config Release
    local end_time=`date +%s.%N`
    local runtime=$( echo "$end_time-$start_time" | bc -l )
    
    strip ${build_dir}/selective_build_test 
    ls -lah ${build_dir}/selective_build_test
    local STAT_OUTPUT=$(stat --format=%s ${build_dir}/selective_build_test)
    echo "${model_name},False,True,OFF,${STAT_OUTPUT},${runtime}" >> results.txt
}

test_cmake_select_ops_in_list() {
    local operator_name=$1
    local example_dir=examples/dtype_selective_build
    local build_dir=cmake-out/${example_dir}
    rm -rf ${build_dir}
    local start_time=`date +%s.%N`
    retry cmake -DCMAKE_BUILD_TYPE=Release \
            -DMAX_KERNEL_NUM=22 \
            -DEXECUTORCH_SELECT_OPS_LIST=${operator_name} \
            -DCMAKE_INSTALL_PREFIX=cmake-out \
            -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
            -B${build_dir} \
            ${example_dir}

    echo "Building ${example_dir}"
    cmake --build ${build_dir} -j9 --config Release
    local end_time=`date +%s.%N`
    local runtime=$( echo "$end_time-$start_time" | bc -l )

    strip ${build_dir}/selective_build_test 
    ls -lah ${build_dir}/selective_build_test
    local STAT_OUTPUT=$(stat --format=%s ${build_dir}/selective_build_test)
    echo "${operator_name},${STAT_OUTPUT},${runtime}" >> results.txt
}

test_cmake_select_ops_in_yaml() {
    echo "Exporting custom_op_1"
    ${PYTHON_EXECUTABLE} -m examples.portable.custom_ops.custom_ops_1
    local example_dir=examples/dtype_selective_build
    local build_dir=cmake-out/${example_dir}
    rm -rf ${build_dir}
    retry cmake -DCMAKE_BUILD_TYPE=Release \
            -DEXECUTORCH_SELECT_OPS_YAML=ON \
            -DCMAKE_INSTALL_PREFIX=cmake-out \
            -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
            -B${build_dir} \
            ${example_dir}

    echo "Building ${example_dir}"
    cmake --build ${build_dir} -j9 --config Release

    echo 'Running selective build test'
    ${build_dir}/selective_build_test --model_path="./custom_ops_1.pte"

    echo "Removing custom_ops_1.pte"
    rm "./custom_ops_1.pte"
}


test_cmake_select_ops_in_model() {
    local dtype_select=$1
    local model_name=$2
    local model_export_name="${model_name}.pte"
    echo "Exporting ${model_name}"
    ${PYTHON_EXECUTABLE} -m examples.portable.scripts.export --model_name="${model_name}"
    local example_dir=examples/dtype_selective_build
    local build_dir=cmake-out/${example_dir}
    rm -rf ${build_dir}
    local start_time=`date +%s.%N`
    retry cmake -DCMAKE_BUILD_TYPE="$CMAKE_BUILD_TYPE" \
            -DEXECUTORCH_SELECT_OPS_FROM_MODEL="./${model_export_name}" \
            -DEXECUTORCH_DTYPE_SELECTIVE_BUILD=${dtype_select} \
            -DEXECUTORCH_OPTIMIZE_SIZE=ON \
            -DCMAKE_INSTALL_PREFIX=cmake-out \
            -DPYTHON_EXECUTABLE="$PYTHON_EXECUTABLE" \
            -B${build_dir} \
            ${example_dir}

    echo "Building ${example_dir}"
    cmake --build ${build_dir} -j9 --config $CMAKE_BUILD_TYPE
    local end_time=`date +%s.%N`
    local runtime=$( echo "$end_time-$start_time" | bc -l )

    strip ${build_dir}/selective_build_test 
    local STAT_OUTPUT=$(stat --format=%s ${build_dir}/selective_build_test)

    if [[ ${dtype_select} == "ON" ]]; then
        local dtype_count=$(python3 get_number_dtypes_per_op.py)
        echo "${model_name},False,False,${dtype_select},${STAT_OUTPUT},${runtime},${dtype_count}" >> results.txt
    else
        echo "${model_name},False,False,${dtype_select},${STAT_OUTPUT},${runtime}" >> results.txt
    fi
}

if [[ -z $PYTHON_EXECUTABLE ]];
then
  PYTHON_EXECUTABLE=python3
fi

if [[ -z $CMAKE_BUILD_TYPE ]];
then
  CMAKE_BUILD_TYPE=Release
fi


if [[ $1 == "cmake" ]];
then
    cmake_install_executorch_lib $CMAKE_BUILD_TYPE
    #test_cmake_select_all_ops
    #test_cmake_select_ops_in_list
    #test_cmake_select_ops_in_yaml
    #test_cmake_select_ops_in_model
    echo "Model,UseNoOps,IncludeAllOps,ModelDtypeSelect,StrippedBinarySize,CompilationTime(sec),NumOps,OpsWith1Dtype,OpsWith2Dtypes,OpsWith3+Dtypes" >> results.txt
    # Problematic models, dtype selective build is larger here for some reason
        #"llama2"
        #"llama"
    models=(
        "phi_4_mini"
        "qwen2_5"
    )
        #"llama3_2_vision_encoder"
    for item in "${models[@]}"; do
        test_cmake_select_all_ops "$item"
    done
    for item in "${models[@]}"; do
        test_cmake_select_ops_in_model "ON" "$item"
        test_cmake_select_ops_in_model "OFF" "$item"
    done
    #qwen2_5_ops="aten::_softmax.out,aten::add.out,aten::any.out,aten::bmm.out,aten::cat.out,aten::clone.out,aten::embedding.out,aten::eq.Scalar_out,aten::expand_copy.out,aten::full_like.out,aten::logical_not.out,aten::mean.out,aten::mm.out,aten::mul.Scalar_out,aten::mul.out,aten::permute_copy.out,aten::rsqrt.out,aten::scalar_tensor.out,aten::select_copy.int_out,aten::sigmoid.out,aten::slice_copy.Tensor_out,aten::squeeze_copy.dims_out,aten::sub.out,aten::unsqueeze_copy.out,aten::where.self_out"
    #test_cmake_select_ops_in_list "$qwen2_5_ops"
elif [[ $1 == "buck2" ]];
then
    test_buck2_select_all_ops
    test_buck2_select_ops_in_list
    test_buck2_select_ops_in_dict
    test_buck2_select_ops_from_yaml
fi
