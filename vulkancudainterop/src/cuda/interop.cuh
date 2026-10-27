#ifndef CUDA_INTEROP_CUH
#define CUDA_INTEROP_CUH
#include <glm/glm.hpp>
#include <vulkan/vulkan.h>

struct NativeHandle {
  union {
    void *win32;
    int fd;
  };
};

struct MandelbulbRenderParams {
  glm::vec3 cameraPosition;
  float pad0 = 0.0f;
  glm::vec3 cameraForward;
  float pad1 = 0.0f;
  glm::vec3 cameraRight;
  float pad2 = 0.0f;
  glm::vec3 cameraUp;
  float pad3 = 0.0f;

  float tanHalfFovY = 0.41421356f;
  float aspect = 1.0f;
  float zoom = 1.0f;
  float maxDistance = 50.0f;
  float hitEpsilon = 0.0015f;

  int maxMarchSteps = 180;
  int maxFractalIterations = 12;
  float power = 8.0f;
  float bailout = 8.0f;

  float normalEpsilon = 0.0008f;
  int enableLighting = 1;
  int enableShadows = 1;
  int shadowSteps = 20;

  float shadowStrength = 8.0f;
  float voxelBounds = 1.5f;
  glm::vec3 voxelCenter{0.0f, 0.0f, 0.0f};
  float pad4 = 0.0f;
};

// Exports a vulkan memory allocation into CUDA.
// mem - native win32 handle to the memory allocation
// size - entire size of the allocation
// offset - offset to the start of the image to be used by CUDA
// width, height - image dimensions
// the expected format is hardcoded in the function body (and in the function
// name)
void exportVulkanImageToCuda_R8G8B8A8Unorm(NativeHandle mem, VkDeviceSize size,
                                           VkDeviceSize offset, uint32_t width,
                                           uint32_t height);
// frees the exported resources
// the resources are stored in global variables for simplicity so resources must
// be freed before a new export
void freeExportedVulkanImage();
void renderCuda(const MandelbulbRenderParams &params);

bool initVolumeTexture(int resolution);
void voxelizeMandelbulbChunk(int resolution, float bounds, glm::vec3 center, float power, float bailout, int maxIter, int startZ, int endZ);
void freeVolumeTexture();
void copyVolumeHostToDevice(int resolution, const float* h_data);
void copyVolumeDeviceToHost(int resolution, float* h_data);

void freeExportedSemaphores();
void exportSemaphoresToCuda(NativeHandle cudaWaitsForVulkanSemaphoreHandle,
                            NativeHandle vulkanWaitsForCudaSemaphoreHandle);

#endif // CUDA_INTEROP_CUH
