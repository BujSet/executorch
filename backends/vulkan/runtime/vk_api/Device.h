/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// @lint-ignore-every CLANGTIDY facebook-hte-BadMemberName

#include <executorch/backends/vulkan/runtime/vk_api/vk_api.h>

#include <sstream>
#include <vector>

namespace vkcompute {
namespace vkapi {

enum class DeviceType : uint32_t {
  UNKNOWN,
  NVIDIA,
  MALI,
  ADRENO,
  SWIFTSHADER,
};

struct PhysicalDevice final {
  // Handle
  VkPhysicalDevice handle;

  // Properties obtained from Vulkan
  VkPhysicalDeviceProperties properties;
  VkPhysicalDeviceMemoryProperties memory_properties;

  // Additional features available from extensions
#ifdef VK_KHR_16bit_storage
  VkPhysicalDevice16BitStorageFeatures shader_16bit_storage;
#endif /* VK_KHR_16bit_storage */
#ifdef VK_KHR_8bit_storage
  VkPhysicalDevice8BitStorageFeatures shader_8bit_storage;
#endif /* VK_KHR_8bit_storage */
#ifdef VK_KHR_shader_float16_int8
  VkPhysicalDeviceShaderFloat16Int8Features shader_float16_int8_types;
#endif /* VK_KHR_shader_float16_int8 */

  // Available GPU queues
  std::vector<VkQueueFamilyProperties> queue_families;

  // Metadata
  uint32_t num_compute_queues;
  bool supports_int16_shader_types;
  bool has_unified_memory;
  bool has_timestamps;
  float timestamp_period;
  size_t min_ubo_alignment;

  // Device identity
  std::string device_name;
  DeviceType device_type;

  explicit PhysicalDevice(VkPhysicalDevice);
};

struct DeviceHandle final {
  VkDevice handle;

  explicit DeviceHandle(VkDevice);
  ~DeviceHandle();
};

void find_requested_device_extensions(
    VkPhysicalDevice physical_device,
    std::vector<const char*>& enabled_extensions,
    const std::vector<const char*>& requested_extensions);

std::string get_device_type_str(const VkPhysicalDeviceType type);

std::string get_memory_properties_str(const VkMemoryPropertyFlags flags);

std::string get_queue_family_properties_str(const VkQueueFlags flags);

} // namespace vkapi
} // namespace vkcompute
