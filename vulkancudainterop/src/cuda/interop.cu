#include "interop.cuh"
#include "stdio.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <memory>

#define checkCudaError(ans)                                                    \
  {                                                                            \
    gpuAssert((ans), __FILE__, __LINE__);                                      \
  }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
  }
}

uint32_t imageWidth, imageHeight;
cudaExternalMemory_t
    cudaExtMemImageBuffer; // memory handler to the imported memory allocation
cudaMipmappedArray_t
    cudaMipmappedImageArray;       // the image interpreted as a mipmapped array
cudaSurfaceObject_t surfaceObject; // surface object to the first mip level of
                                   // the array. Allows write

cudaArray_t volumeArray = nullptr;
cudaSurfaceObject_t volumeSurface = 0;
__device__ cudaTextureObject_t volumeTexture = 0;

__device__ float map(float x, float fromMin, float fromMax, float toMin,
                     float toMax) {
  return toMin + (toMax - toMin) * (x - fromMin) / (fromMax - fromMin);
}

__device__ float3 f3_add(const float3 &a, const float3 &b) {
  return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 f3_sub(const float3 &a, const float3 &b) {
  return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 f3_mul(const float3 &a, float s) {
  return make_float3(a.x * s, a.y * s, a.z * s);
}

__device__ float f3_dot(const float3 &a, const float3 &b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 f3_cross(const float3 &a, const float3 &b) {
  return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z,
                     a.x * b.y - a.y * b.x);
}

__device__ float f3_length(const float3 &v) { return sqrtf(f3_dot(v, v)); }

__device__ float3 f3_normalize(const float3 &v) {
  float len = f3_length(v);
  if (len < 1e-8f)
    return make_float3(0.0f, 0.0f, 0.0f);
  return f3_mul(v, 1.0f / len);
}

__device__ float3 f3_reflect(const float3 &i, const float3 &n) {
  return f3_sub(i, f3_mul(n, 2.0f * f3_dot(n, i)));
}

__device__ float saturate(float x) { return fminf(fmaxf(x, 0.0f), 1.0f); }

__device__ float mandelbulbDE(const float3 &p, int maxIter, float power,
                              float bailout, float cScale,
                              const float3 &cOffset, int *iterOut) {
  float3 z = make_float3(0.0f, 0.0f, 0.0f);
  float3 c = f3_add(f3_mul(p, cScale), cOffset);
  float dr = 1.0f;
  float r = 0.0f;
  int i = 0;

  for (; i < maxIter; ++i) {
    r = f3_length(z);
    if (r > bailout)
      break;

    float safeR = fmaxf(r, 1e-6f);
    float theta = acosf(fminf(fmaxf(z.z / safeR, -1.0f), 1.0f));
    float phi = atan2f(z.y, z.x);
    dr = powf(safeR, power - 1.0f) * power * dr + 1.0f;

    float zr = powf(safeR, power);
    theta *= power;
    phi *= power;

    float sinTheta = sinf(theta);
    float3 bulb = make_float3(sinTheta * cosf(phi), sinTheta * sinf(phi),
                              cosf(theta));
    z = f3_add(f3_mul(bulb, zr), c);
  }

  if (iterOut)
    *iterOut = i;

  if (r < 1e-6f)
    return 0.0f;
  return 0.5f * logf(r) * r / dr;
}

void freeVolumeTexture() {
  cudaTextureObject_t h_volumeTexture;
  checkCudaError(cudaMemcpyFromSymbol(&h_volumeTexture, volumeTexture, sizeof(cudaTextureObject_t)));

  if (h_volumeTexture) {
    cudaDestroyTextureObject(h_volumeTexture);
    h_volumeTexture = 0;
    checkCudaError(cudaMemcpyToSymbol(volumeTexture, &h_volumeTexture, sizeof(cudaTextureObject_t)));
  }
  if (volumeSurface) {
    cudaDestroySurfaceObject(volumeSurface);
    volumeSurface = 0;
  }
  if (volumeArray) {
    cudaFreeArray(volumeArray);
    volumeArray = nullptr;
  }
}

bool initVolumeTexture(int resolution) {
  freeVolumeTexture();

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaExtent extent = make_cudaExtent(resolution, resolution, resolution);

  cudaError_t err = cudaMalloc3DArray(&volumeArray, &channelDesc, extent, cudaArraySurfaceLoadStore);
  if (err != cudaSuccess) {
      fprintf(stderr, "Failed to allocate 3D array: %s\n", cudaGetErrorString(err));
      volumeArray = nullptr;
      return false;
  }

  cudaResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = volumeArray;

  checkCudaError(cudaCreateSurfaceObject(&volumeSurface, &resDesc));

  cudaTextureDesc texDesc;
  memset(&texDesc, 0, sizeof(texDesc));
  texDesc.addressMode[0] = cudaAddressModeClamp;
  texDesc.addressMode[1] = cudaAddressModeClamp;
  texDesc.addressMode[2] = cudaAddressModeClamp;
  texDesc.filterMode = cudaFilterModeLinear;
  texDesc.readMode = cudaReadModeElementType;
  texDesc.normalizedCoords = 1;

  cudaTextureObject_t h_volumeTexture;
  checkCudaError(cudaCreateTextureObject(&h_volumeTexture, &resDesc, &texDesc, nullptr));
  checkCudaError(cudaMemcpyToSymbol(volumeTexture, &h_volumeTexture, sizeof(cudaTextureObject_t)));
  return true;
}

void copyVolumeHostToDevice(int resolution, const float* h_data) {
  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcPtr = make_cudaPitchedPtr((void*)h_data, resolution * sizeof(float), resolution, resolution);
  copyParams.dstArray = volumeArray;
  copyParams.extent = make_cudaExtent(resolution, resolution, resolution);
  copyParams.kind = cudaMemcpyHostToDevice;
  checkCudaError(cudaMemcpy3D(&copyParams));
}

void copyVolumeDeviceToHost(int resolution, float* h_data) {
  cudaMemcpy3DParms copyParams = {0};
  copyParams.srcArray = volumeArray;
  copyParams.dstPtr = make_cudaPitchedPtr((void*)h_data, resolution * sizeof(float), resolution, resolution);
  copyParams.extent = make_cudaExtent(resolution, resolution, resolution);
  copyParams.kind = cudaMemcpyDeviceToHost;
  checkCudaError(cudaMemcpy3D(&copyParams));
}

__global__ void computeVoxelField(cudaSurfaceObject_t surf, int resolution, float bounds, float3 center, float power, float bailout, int maxIter, int startZ, int endZ) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z + startZ;

  if (x >= resolution || y >= resolution || z >= endZ || z >= resolution) return;

  // Sample the center of each voxel, mapped to [center - bounds, center + bounds]
  float fx = map((float)x + 0.5f, 0.0f, (float)resolution, center.x - bounds, center.x + bounds);
  float fy = map((float)y + 0.5f, 0.0f, (float)resolution, center.y - bounds, center.y + bounds);
  float fz = map((float)z + 0.5f, 0.0f, (float)resolution, center.z - bounds, center.z + bounds);

  float3 p = make_float3(fx, fy, fz);
  float dist = mandelbulbDE(p, maxIter, power, bailout, 1.0f, make_float3(0,0,0), nullptr);

  surf3Dwrite(dist, surf, x * sizeof(float), y, z);
}

void voxelizeMandelbulbChunk(int resolution, float bounds, glm::vec3 center, float power, float bailout, int maxIter, int startZ, int endZ) {
  dim3 block(8, 8, 4);
  dim3 grid((resolution + block.x - 1) / block.x, (resolution + block.y - 1) / block.y, (endZ - startZ + block.z - 1) / block.z);
  float3 c = make_float3(center.x, center.y, center.z);
  computeVoxelField<<<grid, block>>>(volumeSurface, resolution, bounds, c, power, bailout, maxIter, startZ, endZ);
  checkCudaError(cudaDeviceSynchronize());
}

void freeExportedVulkanImage() {
  checkCudaError(cudaDestroySurfaceObject(surfaceObject));
  checkCudaError(cudaFreeMipmappedArray(cudaMipmappedImageArray));
  checkCudaError(cudaDestroyExternalMemory(cudaExtMemImageBuffer));
}

void exportVulkanImageToCuda_R8G8B8A8Unorm(NativeHandle mem, VkDeviceSize size,
                                           VkDeviceSize offset, uint32_t width,
                                           uint32_t height) {
  imageWidth = width;
  imageHeight = height;

  // import memory into cuda through native handle (win32)
  cudaExternalMemoryHandleDesc cudaExtMemHandleDesc;
  memset(&cudaExtMemHandleDesc, 0, sizeof(cudaExtMemHandleDesc));
#ifdef _WIN64
  cudaExtMemHandleDesc.type =
      cudaExternalMemoryHandleTypeOpaqueWin32;          // after win8
  cudaExtMemHandleDesc.handle.win32.handle = mem.win32; // allocation handle
#else
  cudaExtMemHandleDesc.type = cudaExternalMemoryHandleTypeOpaqueFd;
  cudaExtMemHandleDesc.handle.fd = mem.fd; // allocation handle
#endif                              /* _WIN64 */
  cudaExtMemHandleDesc.size = size; // allocation size

  checkCudaError(
      cudaImportExternalMemory(&cudaExtMemImageBuffer, &cudaExtMemHandleDesc));

  // extract mipmapped array from memory
  cudaExternalMemoryMipmappedArrayDesc externalMemoryMipmappedArrayDesc;
  memset(&externalMemoryMipmappedArrayDesc, 0,
         sizeof(externalMemoryMipmappedArrayDesc));

  // we want to interpret the raw memory as an image so we need to specify its
  // format and layout
  cudaExtent extent = make_cudaExtent(width, height, 0);
  cudaChannelFormatDesc formatDesc; // 4 channel, 8 bit per channel, unsigned
  formatDesc.x = 8;
  formatDesc.y = 8;
  formatDesc.z = 8;
  formatDesc.w = 8;
  formatDesc.f = cudaChannelFormatKindUnsigned;

  externalMemoryMipmappedArrayDesc.offset = offset; // the image starts here
  externalMemoryMipmappedArrayDesc.formatDesc = formatDesc;
  externalMemoryMipmappedArrayDesc.extent = extent;
  externalMemoryMipmappedArrayDesc.flags = 0;
  externalMemoryMipmappedArrayDesc.numLevels = 1; // no mipmapping
  checkCudaError(cudaExternalMemoryGetMappedMipmappedArray(
      &cudaMipmappedImageArray, cudaExtMemImageBuffer,
      &externalMemoryMipmappedArrayDesc));

  // extract first level
  cudaArray_t cudaMipLevelArray;
  checkCudaError(cudaGetMipmappedArrayLevel(&cudaMipLevelArray,
                                            cudaMipmappedImageArray, 0));

  // create surface object for writing
  cudaResourceDesc resourceDesc;
  memset(&resourceDesc, 0, sizeof(resourceDesc));
  resourceDesc.resType = cudaResourceTypeArray;
  resourceDesc.res.array.array = cudaMipLevelArray;

  checkCudaError(cudaCreateSurfaceObject(&surfaceObject, &resourceDesc));
}

cudaExternalSemaphore_t cudaWaitsForVulkanSemaphore,
    vulkanWaitsForCudaSemaphore;

void freeExportedSemaphores() {
  checkCudaError(cudaDestroyExternalSemaphore(cudaWaitsForVulkanSemaphore));
  checkCudaError(cudaDestroyExternalSemaphore(vulkanWaitsForCudaSemaphore));
}

void exportSemaphoresToCuda(NativeHandle cudaWaitsForVulkanSemaphoreHandle,
                            NativeHandle vulkanWaitsForCudaSemaphoreHandle) {
  cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc;
  memset(&externalSemaphoreHandleDesc, 0, sizeof(externalSemaphoreHandleDesc));
  externalSemaphoreHandleDesc.flags = 0;
#ifdef _WIN64
  externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueWin32;
#else
  externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeOpaqueFd;
#endif /* _WIN64 */

#ifdef _WIN64
  externalSemaphoreHandleDesc.handle.win32.handle =
      cudaWaitsForVulkanSemaphoreHandle.win32;
#else
  externalSemaphoreHandleDesc.handle.fd = cudaWaitsForVulkanSemaphoreHandle.fd;
#endif /* _WIN64 */
  checkCudaError(cudaImportExternalSemaphore(&cudaWaitsForVulkanSemaphore,
                                             &externalSemaphoreHandleDesc));

#ifdef _WIN64
  externalSemaphoreHandleDesc.handle.win32.handle =
      vulkanWaitsForCudaSemaphoreHandle.win32;
#else
  externalSemaphoreHandleDesc.handle.fd = vulkanWaitsForCudaSemaphoreHandle.fd;
#endif /* _WIN64 */
  checkCudaError(cudaImportExternalSemaphore(&vulkanWaitsForCudaSemaphore,
                                             &externalSemaphoreHandleDesc));
}

// compresses 4 32bit floats into a 32bit uint
__device__ unsigned int rgbaFloatToInt(float4 rgba) {
  rgba.x = __saturatef(rgba.x); // clamp to [0.0, 1.0]
  rgba.y = __saturatef(rgba.y);
  rgba.z = __saturatef(rgba.z);
  rgba.w = __saturatef(rgba.w);
  return ((unsigned int)(rgba.w * 255.0f) << 24) |
         ((unsigned int)(rgba.z * 255.0f) << 16) |
         ((unsigned int)(rgba.y * 255.0f) << 8) |
         ((unsigned int)(rgba.x * 255.0f));
}

__device__ float3 estimateNormal(const float3 &p,
                                 const MandelbulbRenderParams &params) {
  float e = params.normalEpsilon;
  float bounds = params.voxelBounds;
  float3 center = make_float3(params.voxelCenter.x, params.voxelCenter.y, params.voxelCenter.z);

  auto getDist = [&](float3 pos) {
    float3 texCoord = make_float3((pos.x - (center.x - bounds)) / (2.0f * bounds),
                                 (pos.y - (center.y - bounds)) / (2.0f * bounds),
                                 (pos.z - (center.z - bounds)) / (2.0f * bounds));
    return tex3D<float>(volumeTexture, texCoord.x, texCoord.y, texCoord.z);
  };

  float nx = getDist(f3_add(p, make_float3(e, 0, 0))) - getDist(f3_sub(p, make_float3(e, 0, 0)));
  float ny = getDist(f3_add(p, make_float3(0, e, 0))) - getDist(f3_sub(p, make_float3(0, e, 0)));
  float nz = getDist(f3_add(p, make_float3(0, 0, e))) - getDist(f3_sub(p, make_float3(0, 0, e)));

  return f3_normalize(make_float3(nx, ny, nz));
}

__device__ float softShadow(const float3 &origin, const float3 &lightDir,
                            const MandelbulbRenderParams &params) {
  if (!params.enableShadows || params.shadowSteps <= 0)
    return 1.0f;

  float bounds = params.voxelBounds;
  float3 center = make_float3(params.voxelCenter.x, params.voxelCenter.y, params.voxelCenter.z);
  float shadow = 1.0f;
  float t = params.hitEpsilon * 4.0f;
  for (int i = 0; i < params.shadowSteps && t < params.maxDistance; ++i) {
    float3 p = f3_add(origin, f3_mul(lightDir, t));

    if (p.x < center.x - bounds || p.x > center.x + bounds || 
        p.y < center.y - bounds || p.y > center.y + bounds || 
        p.z < center.z - bounds || p.z > center.z + bounds) {
        t += 0.1f;
        continue;
    }

    float3 texCoord = make_float3((p.x - (center.x - bounds)) / (2.0f * bounds),
                                 (p.y - (center.y - bounds)) / (2.0f * bounds),
                                 (p.z - (center.z - bounds)) / (2.0f * bounds));
    float h = tex3D<float>(volumeTexture, texCoord.x, texCoord.y, texCoord.z);

    if (h < params.hitEpsilon)
      return 0.0f;
    shadow = fminf(shadow, params.shadowStrength * h / t);
    t += fmaxf(h, 0.01f);
  }

  return saturate(shadow);
}

__global__ void renderToSurface(cudaSurfaceObject_t dstSurface, size_t width,
                                size_t height, MandelbulbRenderParams params) {
  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= width || y >= height)
    return;

  float px = ((static_cast<float>(x) + 0.5f) / static_cast<float>(width)) *
                 2.0f -
             1.0f;
  float py = ((static_cast<float>(y) + 0.5f) / static_cast<float>(height)) *
                 2.0f -
             1.0f;

  float3 ro = make_float3(params.cameraPosition.x, params.cameraPosition.y,
                          params.cameraPosition.z);
  float3 forward =
      f3_normalize(make_float3(params.cameraForward.x, params.cameraForward.y,
                               params.cameraForward.z));
  float3 right = f3_normalize(
      make_float3(params.cameraRight.x, params.cameraRight.y, params.cameraRight.z));
  float3 up =
      f3_normalize(make_float3(params.cameraUp.x, params.cameraUp.y, params.cameraUp.z));

    float zoom = fmaxf(params.zoom, 0.01f);
    float invZoom = 1.0f / zoom;

  float3 rd = f3_add(forward,
             f3_add(f3_mul(right,
                     px * params.aspect * params.tanHalfFovY *
                       invZoom),
                f3_mul(up, py * params.tanHalfFovY * invZoom)));
  rd = f3_normalize(rd);

  float t = 0.0f;
  bool hit = false;
  float3 p = ro;
  float bounds = params.voxelBounds;
  float3 center = make_float3(params.voxelCenter.x, params.voxelCenter.y, params.voxelCenter.z);

  for (int step = 0; step < params.maxMarchSteps; ++step) {
    p = f3_add(ro, f3_mul(rd, t));

    float dist;
    if (p.x < center.x - bounds || p.x > center.x + bounds || 
        p.y < center.y - bounds || p.y > center.y + bounds || 
        p.z < center.z - bounds || p.z > center.z + bounds) {
        
        float3 tmin_v = make_float3((center.x - bounds - ro.x) / rd.x, (center.y - bounds - ro.y) / rd.y, (center.z - bounds - ro.z) / rd.z);
        float3 tmax_v = make_float3((center.x + bounds - ro.x) / rd.x, (center.y + bounds - ro.y) / rd.y, (center.z + bounds - ro.z) / rd.z);
        
        float t_near = fmaxf(fmaxf(fminf(tmin_v.x, tmax_v.x), fminf(tmin_v.y, tmax_v.y)), fminf(tmin_v.z, tmax_v.z));
        float t_far = fminf(fminf(fmaxf(tmin_v.x, tmax_v.x), fmaxf(tmin_v.y, tmax_v.y)), fmaxf(tmin_v.z, tmax_v.z));

        if (t_far < t_near || t_far < 0) {
            t = params.maxDistance + 1.0f; // Missed AABB entirely
            break;
        }
        dist = fmaxf(t_near - t, 0.01f);
    } else {
        float3 texCoord = make_float3((p.x - (center.x - bounds)) / (2.0f * bounds),
                                     (p.y - (center.y - bounds)) / (2.0f * bounds),
                                     (p.z - (center.z - bounds)) / (2.0f * bounds));
        dist = tex3D<float>(volumeTexture, texCoord.x, texCoord.y, texCoord.z);
    }

    if (dist < params.hitEpsilon) {
      hit = true;
      break;
    }
    if (t > params.maxDistance)
      break;
    t += fmaxf(dist * 0.95f, params.hitEpsilon * 0.5f);
  }

  float3 color = make_float3(0.16f + 0.12f * py, 0.12f + 0.09f * py,
                             0.10f + 0.06f * py);

  if (hit) {
    float fog = expf(-0.02f * t);
    float3 base = make_float3(0.8f, 0.6f, 0.4f); // Simpler base color for voxels

    if (params.enableLighting) {
      float3 n = estimateNormal(p, params);
      float3 l = f3_normalize(make_float3(0.7f, 1.25f, 0.4f));
      float diff = fmaxf(0.0f, f3_dot(n, l));
      float shadow = softShadow(f3_add(p, f3_mul(n, params.hitEpsilon * 3.0f)), l, params);
      float3 reflected = f3_reflect(f3_mul(l, -1.0f), n);
      float spec = powf(fmaxf(0.0f, f3_dot(reflected, f3_mul(rd, -1.0f))), 22.0f);

      float ambient = 0.34f;
      float lighting = ambient + 1.25f * diff * shadow;
      color = f3_add(f3_mul(base, lighting),
                     f3_mul(make_float3(1.35f, 1.18f, 0.95f),
                            spec * (0.7f + 0.3f * shadow)));
    } else {
      color = base;
    }

    color = f3_add(f3_mul(color, fog),
                   f3_mul(make_float3(0.18f, 0.12f, 0.09f), 1.0f - fog));
  }

  float4 dataOut = float4{color.x, color.y, color.z, 1.0f};
  surf2Dwrite(rgbaFloatToInt(dataOut), dstSurface, x * 4,
              y); // expects byte coordinates. 1 pixel = 4 byte
}

void renderCuda(const MandelbulbRenderParams &params) {
  // Skip waiting on the very first frame to avoid deadlock
  static bool firstFrame = true;

  // LABTODO: synchronization
  if (!firstFrame) {
    cudaExternalSemaphoreWaitParams extSemaphoreWaitParams;
    memset(&extSemaphoreWaitParams, 0, sizeof(extSemaphoreWaitParams));
    extSemaphoreWaitParams.params.fence.value = 0;
    extSemaphoreWaitParams.flags = 0;
    checkCudaError(cudaWaitExternalSemaphoresAsync(&cudaWaitsForVulkanSemaphore,
                                                   &extSemaphoreWaitParams, 1));
  }
  firstFrame = false;

  uint32_t nthreads = 32;
  dim3 dimBlock{nthreads, nthreads};
  dim3 dimGrid{imageWidth / nthreads + 1, imageHeight / nthreads + 1};
  renderToSurface<<<dimGrid, dimBlock>>>(surfaceObject, imageWidth, imageHeight,
                                         params);

  // LABTODO: synchronization
  cudaExternalSemaphoreSignalParams extSemaphoreSignalParams;
  memset(&extSemaphoreSignalParams, 0, sizeof(extSemaphoreSignalParams));
  extSemaphoreSignalParams.params.fence.value = 0;
  extSemaphoreSignalParams.flags = 0;
  checkCudaError(cudaSignalExternalSemaphoresAsync(
      &vulkanWaitsForCudaSemaphore, &extSemaphoreSignalParams, 1));

  checkCudaError(cudaGetLastError());
}
