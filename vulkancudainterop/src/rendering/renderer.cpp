#include "rendering/renderer.h"
#include "asset_managment/asset_manager.h"
#include "camera/perspective_camera.h"
#include "cuda/interop.cuh"
#include "gui/gui_manager.h"
#include "rendering/common_rendering.h"
#include "rendering/renderer_dependency_provider.h"
#include "scene/game_object.h"
#include "scene/game_scene.h"
#include "scene/lights.h"
#include "shader_manager.h"
#include "utility/utility.hpp"
#include "vulkan_layer.h"
#include "window_manager.h"
#include <filesystem>
#include <fstream>
#ifdef _WIN64
#include "dxgi1_2.h"
#endif /* _WIN64 */

namespace {
bool approxEq(float a, float b, float eps = 1e-5f) {
  return fabsf(a - b) <= eps;
}

bool sameVec3(const glm::vec3 &a, const glm::vec3 &b, float eps = 1e-5f) {
  return approxEq(a.x, b.x, eps) && approxEq(a.y, b.y, eps) &&
         approxEq(a.z, b.z, eps);
}

bool sameRenderParams(const MandelbulbRenderParams &a,
                      const MandelbulbRenderParams &b) {
  return sameVec3(a.cameraPosition, b.cameraPosition) &&
         sameVec3(a.cameraForward, b.cameraForward) &&
         sameVec3(a.cameraRight, b.cameraRight) &&
         sameVec3(a.cameraUp, b.cameraUp) && approxEq(a.tanHalfFovY, b.tanHalfFovY) &&
         approxEq(a.aspect, b.aspect) && approxEq(a.zoom, b.zoom) &&
         approxEq(a.maxDistance, b.maxDistance) &&
         approxEq(a.hitEpsilon, b.hitEpsilon) &&
         a.maxMarchSteps == b.maxMarchSteps &&
         a.maxFractalIterations == b.maxFractalIterations &&
         approxEq(a.power, b.power) && approxEq(a.bailout, b.bailout) &&
         approxEq(a.normalEpsilon, b.normalEpsilon) &&
         a.enableLighting == b.enableLighting &&
         a.enableShadows == b.enableShadows && a.shadowSteps == b.shadowSteps &&
         approxEq(a.shadowStrength, b.shadowStrength);
}
} // namespace

void Renderer::update(double dt) {
  bool orbitChanged = false;
  if (fabsf(velocityRightAxis) > 0.001f || fabsf(velocityUpAxis) > 0.001f) {
    orbitYaw += velocityRightAxis * movementSpeed * 60.0f * static_cast<float>(dt);
    orbitPitch = glm::clamp(
        orbitPitch + velocityUpAxis * movementSpeed * 60.0f * static_cast<float>(dt),
        -89.0f, 89.0f);
    orbitChanged = true;
  }

  if (orbitChanged) {
    float pitchRad = glm::radians(orbitPitch);
    float yawRad = glm::radians(orbitYaw);
    glm::vec3 dir = {
        cosf(yawRad) * cosf(pitchRad),
        sinf(pitchRad),
        sinf(yawRad) * cosf(pitchRad),
    };
    fractalCameraPosition = fractalTarget + glm::normalize(dir) * orbitRadius;
    updateFractalCameraVectors();
    forceCacheRefresh = true;
    cacheValid = false;
  }
}

void Renderer::initialize() {
  query.create();
  createResources();
  createPipelineLayout();
  createFsPipeline();
  createSyncObjects();

  updateFractalCameraVectors();

  {
    glm::vec3 fromTarget = fractalCameraPosition - fractalTarget;
    orbitRadius = glm::max(glm::length(fromTarget), 0.35f);
    glm::vec3 dir = glm::normalize(fromTarget);
    orbitPitch = glm::degrees(asinf(glm::clamp(dir.y, -1.0f, 1.0f)));
    orbitYaw = glm::degrees(atan2f(dir.z, dir.x));
  }

  theInputManager.registerMouseScrollHandler([this](double /*x*/, double y) {
    orbitRadius = glm::clamp(orbitRadius - static_cast<float>(y) * scrollZoomSpeed,
                             0.35f, 200.0f);
    float pitchRad = glm::radians(orbitPitch);
    float yawRad = glm::radians(orbitYaw);
    glm::vec3 dir = {
        cosf(yawRad) * cosf(pitchRad),
        sinf(pitchRad),
        sinf(yawRad) * cosf(pitchRad),
    };
    fractalCameraPosition = fractalTarget + glm::normalize(dir) * orbitRadius;
    updateFractalCameraVectors();
    forceCacheRefresh = true;
    cacheValid = false;
  });

  theInputManager.registerUtf8KeyHandler(
      "w", [this](auto /* mod */, auto action) {
        if (action == glfwim::Action::Press)
          velocityUpAxis += 1.0f;
        else if (action == glfwim::Action::Release)
          velocityUpAxis -= 1.0f;
      });
  theInputManager.registerUtf8KeyHandler(
      "s", [this](auto /* mod */, auto action) {
        if (action == glfwim::Action::Press)
          velocityUpAxis -= 1.0f;
        else if (action == glfwim::Action::Release)
          velocityUpAxis += 1.0f;
      });
  theInputManager.registerUtf8KeyHandler(
      "a", [this](auto /* mod */, auto action) {
        if (action == glfwim::Action::Press)
          velocityRightAxis -= 1.0f;
        else if (action == glfwim::Action::Release)
          velocityRightAxis += 1.0f;
      });
  theInputManager.registerUtf8KeyHandler(
      "d", [this](auto /* mod */, auto action) {
        if (action == glfwim::Action::Press)
          velocityRightAxis += 1.0f;
        else if (action == glfwim::Action::Release)
          velocityRightAxis -= 1.0f;
      });

  // Camera direction is locked to fractalTarget; no free-look cursor handlers.
}

void Renderer::destroy() {
  query.destroy();
  descriptorPoolBuilder.destroy();
  perFrameDscSetBuilder.destroyLayout();
  for (auto &it : perFrameBuffers)
    it.destroy();
  theVulkanLayer.device.destroy(fsSampler);
  theVulkanLayer.device.destroy(fsPipelineLayout);
  fsPipeline.destroy();
  vmaDestroyPool(theVulkanLayer.allocator, externalPool);

  freeExportedSemaphores();
  theVulkanLayer.device.destroy(cudaWaitsForVulkanSemaphore);
  theVulkanLayer.device.destroy(vulkanWaitsForCudaSemaphore);
}

void Renderer::initializeSwapChainDependentComponents() {
  createImages();
  updateDescriptorSets();
  if (!theVulkanLayer.dynamicRenderingAvailable) {
    if (firstSwapChainCreate) {
      createRenderPass();
      firstSwapChainCreate = false;
    }
    createFramebuffers();
  }
}

void Renderer::destroySwapChainDependentComponents() {
  freeExportedVulkanImage();
  colorImage.destroy();
  if (!theVulkanLayer.dynamicRenderingAvailable) {
    for (auto &it : framebuffer) {
      theVulkanLayer.device.destroy(it);
    }
  }
}

void Renderer::render(const RenderContext &ctx) {
  auto &vl = theVulkanLayer;
  auto &wm = theWindowManager;

  // timestamp query
  query.query();
  // not representative with cuda!
  theGUIManager.addStatistic(
      "Renderer", std::make_tuple("Frame time: ",
                                  (query.results[1] - query.results[0]) *
                                      vl.timestampPeriod / 1e6,
                                  " ms"));

  if (ImGui::Begin("Mandelbulb Voxelizer")) {
    bool lightingEnabled = enableLighting != 0;
    bool shadowsEnabled = enableShadows != 0;
    bool autoDetailEnabled = autoDetailFromZoom != 0;

    ImGui::DragFloat("Orbit speed", &movementSpeed, 0.05f, 0.2f, 40.0f);
    ImGui::DragFloat("Scroll zoom speed", &scrollZoomSpeed, 0.01f, 0.05f,
             2.0f);
    ImGui::Separator();
    
    ImGui::Text("Voxel Settings (Requires Voxelize)");
    bool dirty = false;
    if (ImGui::DragInt("Resolution", &voxelResolution, 8, 32, 2048)) dirty = true;
    if (ImGui::DragFloat("Half-Extent", &voxelBounds, 0.05f, 0.01f, 10.0f)) dirty = true;
    if (ImGui::DragFloat3("Voxel Center", &voxelCenter.x, 0.05f)) dirty = true;
    
    if (ImGui::Button("Align Voxel Cube to Camera Target")) {
        voxelCenter = fractalTarget;
        voxelBounds = orbitRadius * 0.5f;
        dirty = true;
    }

    if (ImGui::DragFloat("Power", &fractalPower, 0.05f, 2.0f, 16.0f)) dirty = true;
    if (ImGui::DragFloat("Bailout", &fractalBailout, 0.1f, 2.0f, 50.0f)) dirty = true;
    if (ImGui::DragInt("Fractal Iterations", &maxFractalIterations, 1.0f, 4, 128)) dirty = true;

    if (dirty) {
        voxelDataValid = false;
    }

    if (ImGui::Button("Voxelize & Save") && !voxelizing) {
      forceVoxelRegeneration = true;
      voxelErrorMessage = "";
    }
    ImGui::SameLine();
    if (voxelizing) {
        ImGui::ProgressBar(voxelizationProgress, ImVec2(-1, 0));
    } else if (!voxelErrorMessage.empty()) {
        ImGui::TextColored(ImVec4(1, 0, 0, 1), "Error: %s", voxelErrorMessage.c_str());
    } else if (forceVoxelRegeneration || !voxelDataValid) {
        ImGui::TextColored(ImVec4(1, 1, 0, 1), "Status: DIRTY (Click Voxelize)");
    } else {
        ImGui::TextColored(ImVec4(0, 1, 0, 1), "Status: Up to date");
    }

    ImGui::Separator();
    ImGui::Text("Render Settings (Real-time)");
    ImGui::DragFloat("Zoom", &fractalZoom, 0.02f, 0.25f, 32.0f, "%.2f");
    ImGui::DragFloat("Hit epsilon", &hitEpsilon, 0.0001f,
                     0.00005f, 0.02f, "%.5f");
    ImGui::DragFloat("Normal epsilon", &normalEpsilon,
                     0.0001f, 0.00005f, 0.01f, "%.5f");
    ImGui::DragFloat("Max distance", &maxDistance, 0.5f, 5.0f, 200.0f);
    ImGui::DragInt("Max March Steps", &maxMarchSteps, 1.0f, 16, 1024);
    
    ImGui::Checkbox("Enable lighting", &lightingEnabled);
    ImGui::Checkbox("Enable shadows", &shadowsEnabled);
    ImGui::DragInt("Shadow steps", &shadowSteps, 1.0f, 0, 64);
    ImGui::DragFloat("Shadow strength", &shadowStrength, 0.1f, 0.0f, 24.0f);

    ImGui::Separator();
    ImGui::Checkbox("Auto detail from zoom", &autoDetailEnabled);

    if (ImGui::Button("Reset fractal camera")) {
      fractalTarget = {0.0f, 0.0f, 0.0f};
      fractalCameraPosition = {3.5f, 2.0f, 3.5f};
      glm::vec3 fromTarget = fractalCameraPosition - fractalTarget;
      orbitRadius = glm::max(glm::length(fromTarget), 0.35f);
      glm::vec3 dir = glm::normalize(fromTarget);
      orbitPitch = glm::degrees(asinf(glm::clamp(dir.y, -1.0f, 1.0f)));
      orbitYaw = glm::degrees(atan2f(dir.z, dir.x));
      updateFractalCameraVectors();
    }

    enableLighting = lightingEnabled ? 1 : 0;
    enableShadows = shadowsEnabled ? 1 : 0;
    autoDetailFromZoom = autoDetailEnabled ? 1 : 0;
  }
  ImGui::End();

  auto vp = vk::Viewport{0,
                         0,
                         (float)wm.swapChainExtent.width,
                         (float)wm.swapChainExtent.height,
                         0.0,
                         1.0};

  // Update uniforms
  PerFrameUniformData fud = {};
  fud.v = ctx.cam->V();
  fud.ambientLight = glm::vec4{ctx.pGameScene->ambientLight.color,
                               ctx.pGameScene->ambientLight.power};
  memcpy(perFrameBuffers[ctx.frameID].allocationInfo.pMappedData, &fud,
         sizeof(PerFrameUniformData));
  perFrameBuffers[ctx.frameID].flush();

  query.beginFrame(ctx.cmd);
  query.write(ctx.cmd, vk::PipelineStageFlagBits::eTopOfPipe);

  vk::RenderingAttachmentInfo colorAttachment = {
      .imageView = wm.swapChainImageViews[ctx.imageID],
      .imageLayout = vk::ImageLayout::eColorAttachmentOptimal,
      .loadOp = vk::AttachmentLoadOp::eClear,
      .storeOp = vk::AttachmentStoreOp::eStore,
      .clearValue = vk::ClearColorValue{0.0f, 0.0f, 0.0f, 1.0f}};

  vk::RenderingInfo renderingInfo = {.renderArea = {{0, 0}, wm.swapChainExtent},
                                     .layerCount = 1,
                                     .colorAttachmentCount = 1,
                                     .pColorAttachments = &colorAttachment};

  if (vl.dynamicRenderingAvailable) {
    ctx.cmd.beginRendering(renderingInfo);
  } else {
    startRenderpass(ctx);
  }

  float aspect = static_cast<float>(wm.swapChainExtent.width) /
                 static_cast<float>(glm::max(1u, wm.swapChainExtent.height));

  MandelbulbRenderParams params{};
  params.cameraPosition = fractalCameraPosition;
  params.cameraForward = fractalCameraForward;
  params.cameraRight = fractalCameraRight;
  params.cameraUp = fractalCameraUp;
  params.aspect = aspect;
  params.zoom = fractalZoom;
  params.tanHalfFovY = tanf(glm::radians(ctx.cam->fov() * 0.5f));
  params.maxDistance = maxDistance;
  
  float zoomSafe = glm::max(0.01f, fractalZoom);
  if (autoDetailFromZoom != 0) {
    params.hitEpsilon = hitEpsilon / zoomSafe;
    params.normalEpsilon = normalEpsilon / zoomSafe;
  } else {
    params.hitEpsilon = hitEpsilon;
    params.normalEpsilon = normalEpsilon;
  }
  params.maxMarchSteps = glm::clamp(maxMarchSteps, 8, 4096);
  params.maxFractalIterations = glm::clamp(maxFractalIterations, 2, 128);
  params.power = fractalPower;
  params.bailout = fractalBailout;
  params.enableLighting = enableLighting;
  params.enableShadows = enableShadows;
  params.shadowSteps = glm::clamp(shadowSteps, 0, 128);
  params.shadowStrength = shadowStrength;
  params.voxelBounds = voxelBounds;

  // Voxelization Logic
  std::string modelName = "mandelbulb_r" + std::to_string(voxelResolution) + 
                          "_p" + std::to_string((int)fractalPower) + 
                          "_c" + std::to_string((int)(voxelCenter.x * 100)) + "_" + 
                                 std::to_string((int)(voxelCenter.y * 100)) + "_" + 
                                 std::to_string((int)(voxelCenter.z * 100)) + ".bin";
  std::filesystem::path modelPath = "models" / std::filesystem::path(modelName);

  if (forceVoxelRegeneration || (!voxelDataValid && !voxelizing)) {
      if (initVolumeTexture(voxelResolution)) {
          if (!forceVoxelRegeneration && std::filesystem::exists(modelPath)) {
              // Try loading synchronously
              size_t size = (size_t)voxelResolution * voxelResolution * voxelResolution;
              std::vector<float> h_data(size);
              std::ifstream is(modelPath, std::ios::binary);
              is.read(reinterpret_cast<char*>(h_data.data()), size * sizeof(float));
              copyVolumeHostToDevice(voxelResolution, h_data.data());
              voxelDataValid = true;
          } else {
              // Start async voxelization
              voxelizing = true;
              currentVoxelZ = 0;
              voxelizationProgress = 0.0f;
              forceVoxelRegeneration = false;
          }
      } else {
          voxelErrorMessage = "Out of GPU Memory! Try lower resolution.";
          voxelDataValid = false;
          forceVoxelRegeneration = false;
      }
  }

  if (voxelizing) {
      int chunkSize = glm::max(1, voxelResolution / 64); // Voxelize in 64 steps
      int nextZ = glm::min(currentVoxelZ + chunkSize, voxelResolution);
      voxelizeMandelbulbChunk(voxelResolution, voxelBounds, voxelCenter, fractalPower, fractalBailout, maxFractalIterations, currentVoxelZ, nextZ);
      currentVoxelZ = nextZ;
      voxelizationProgress = (float)currentVoxelZ / (float)voxelResolution;

      if (currentVoxelZ >= voxelResolution) {
          voxelizing = false;
          voxelDataValid = true;
          
          // Save model
          size_t size = (size_t)voxelResolution * voxelResolution * voxelResolution;
          std::vector<float> h_data(size);
          copyVolumeDeviceToHost(voxelResolution, h_data.data());
          if (!std::filesystem::exists("models")) {
              std::filesystem::create_directories("models");
          }
          std::ofstream os(modelPath, std::ios::binary);
          os.write(reinterpret_cast<const char*>(h_data.data()), size * sizeof(float));
      }
  }

  params.voxelCenter = voxelCenter;
  if (voxelDataValid) {
    renderCuda(params);
  }

  ctx.cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, fsPipelineLayout,
                             0, 1, &perFrameDescriptorSets[ctx.frameID], 0,
                             nullptr);
  ctx.cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, fsPipeline);
  ctx.cmd.setViewport(0, 1, &vp);
  ctx.cmd.setScissor(0, 1, &renderingInfo.renderArea);
  ctx.cmd.draw(3, 1, 0, 0);

  if (vl.dynamicRenderingAvailable) {
    ctx.cmd.endRendering();
  } else {
    ctx.cmd.endRenderPass();
  }

  query.write(ctx.cmd, vk::PipelineStageFlagBits::eBottomOfPipe);
  query.endFrame();
}

void Renderer::createPipelineLayout() {
  descriptorPoolBuilder.initialize();

  perFrameDscSetBuilder.setMaximumSetCount(MAX_FRAMES_IN_FLIGHT)
      .addBinding(0, vk::DescriptorType::eUniformBuffer, // perFrameUniformData
                  vk::ShaderStageFlagBits::eVertex |
                      vk::ShaderStageFlagBits::eFragment)
      .addBinding(1, vk::DescriptorType::eCombinedImageSampler, // cuda image
                  vk::ShaderStageFlagBits::eFragment, &fsSampler)
      .createLayout();

  descriptorPoolBuilder.create();

  perFrameDescriptorSets =
      perFrameDscSetBuilder.allocateSets<MAX_FRAMES_IN_FLIGHT>();

  fsPipelineLayout = theVulkanLayer.name(
      theVulkanLayer.createPipelineLayout({perFrameDscSetBuilder.layout()}),
      "Renderer_FSPipelineLayout");
}

void Renderer::updateFractalCameraVectors() {
  glm::vec3 toTarget = fractalTarget - fractalCameraPosition;
  if (glm::length(toTarget) < 1e-5f) {
    toTarget = glm::vec3{0.0f, 0.0f, -1.0f};
  }

  fractalCameraForward = glm::normalize(toTarget);
  glm::vec3 worldUp{0.0f, 1.0f, 0.0f};
  fractalCameraRight = glm::cross(fractalCameraForward, worldUp);
  if (glm::length(fractalCameraRight) < 1e-5f) {
    fractalCameraRight = glm::vec3{1.0f, 0.0f, 0.0f};
  } else {
    fractalCameraRight = glm::normalize(fractalCameraRight);
  }
  fractalCameraUp = glm::normalize(glm::cross(fractalCameraRight, fractalCameraForward));
}

void Renderer::updateDescriptorSets() {
  perFrameDscSetBuilder.update(perFrameDescriptorSets)
      .writeBuffer(0, perFrameBuffers)
      .writeImage(1, colorImage.view, vk::ImageLayout::eGeneral);
}

void Renderer::createImages() {
  auto &vl = theVulkanLayer;
  auto &wm = theWindowManager;

  // we are using the custom pool
  VmaAllocationCreateInfo alloc = {
      .usage = (VmaMemoryUsage)vma::MemoryUsage::eAutoPreferDevice,
      .pool = externalPool};

  // and we want to export the image
  vk::ExternalMemoryImageCreateInfo extInfo = {
#ifdef _WIN64
      .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueWin32
#else
      .handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd
#endif /* _WIN64 */
  };

  colorImage = vl.name(
      vl.createImage(
          wm.swapChainExtent.width, wm.swapChainExtent.height, 1,
          vk::SampleCountFlagBits::e1, vk::Format::eR8G8B8A8Unorm,
          vk::ImageLayout::eGeneral, // general layout so it is compatible
                                     // (doesnt really matter as CUDA is nvidia
                                     // specific)
          vk::ImageAspectFlagBits::eColor, vk::ImageUsageFlagBits::eSampled,
          alloc, vk::ImageTiling::eOptimal,
          &extInfo // extension struct passed here
          ),
      "Renderer_ColorImage");

  colorImageNativeHandle =
      colorImage.getNativeHandle(); // receive a native win32 handle
  // and finally export the image to CUDA
  exportVulkanImageToCuda_R8G8B8A8Unorm(
      colorImageNativeHandle, colorImage.allocationInfo.size,
      colorImage.allocationInfo.offset, colorImage.width, colorImage.height);
}

void Renderer::createResources() {
  auto &vl = theVulkanLayer;

  for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
    perFrameBuffers[i] = vl.createBuffer(
        sizeof(PerFrameUniformData), vk::BufferUsageFlagBits::eUniformBuffer,
        vma::AllocationCreateFlagBits::eHostAccessSequentialWriteBit |
            vma::AllocationCreateFlagBits::eCreateMappedBit);
  }

  fsSampler = vl.device.createSampler(
      vk::SamplerCreateInfo{.magFilter = vk::Filter::eLinear,
                            .minFilter = vk::Filter::eLinear,
                            .mipmapMode = vk::SamplerMipmapMode::eLinear,
                            .addressModeU = vk::SamplerAddressMode::eRepeat,
                            .addressModeV = vk::SamplerAddressMode::eRepeat,
                            .addressModeW = vk::SamplerAddressMode::eRepeat,
                            .maxAnisotropy = 1.0f,
                            .minLod = -1000,
                            .maxLod = 1000});

  // we create the vma pool. Most parameters are default
  VmaPoolCreateInfo poolInfo = {.flags = {},
                                .blockSize = 0,
                                .minBlockCount = 0,
                                .maxBlockCount = 0,
                                .priority = 1.0,
                                .minAllocationAlignment = 0};

  // example image similar to those that we will allocate from the pool
  // this way the pool can select a correct memory type
  vk::ImageCreateInfo example = {.flags = {},
                                 .imageType = vk::ImageType::e2D,
                                 .format = vk::Format::eR8G8B8A8Unorm,
                                 .extent = {100, 100, 1},
                                 .mipLevels = 1,
                                 .arrayLayers = 1,
                                 .samples = vk::SampleCountFlagBits::e1,
                                 .tiling = vk::ImageTiling::eOptimal,
                                 .usage = vk::ImageUsageFlagBits::eSampled,
                                 .sharingMode = vk::SharingMode::eExclusive,
                                 .initialLayout = vk::ImageLayout::eUndefined};

  VmaAllocationCreateInfo allocInfo = {
      .usage = (VmaMemoryUsage)vma::MemoryUsage::eAutoPreferDevice,
      .requiredFlags =
          (VkMemoryPropertyFlags)vk::MemoryPropertyFlagBits::eDeviceLocal};

  // find the memory type
  vmaFindMemoryTypeIndexForImageInfo(
      vl.allocator, reinterpret_cast<VkImageCreateInfo *>(&example), &allocInfo,
      &poolInfo.memoryTypeIndex);
#ifdef _WIN64
  // define windows security attributes and settings for the allocations
  vulkanExportMemoryWin32HandleInfoKHR.sType =
      VK_STRUCTURE_TYPE_EXPORT_MEMORY_WIN32_HANDLE_INFO_KHR;
  vulkanExportMemoryWin32HandleInfoKHR.pNext = nullptr;
  vulkanExportMemoryWin32HandleInfoKHR.pAttributes = &winSecurityAttributes;
  vulkanExportMemoryWin32HandleInfoKHR.dwAccess =
      DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE;
  vulkanExportMemoryWin32HandleInfoKHR.name = (LPCWSTR) nullptr;
#endif /* _WIN64 */
  exportMemoryInfo.sType = VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO;
#ifdef _WIN64
  // allocation can be exported as a native win32 handle
  exportMemoryInfo.handleTypes =
      VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
  exportMemoryInfo.pNext = &vulkanExportMemoryWin32HandleInfoKHR;
#else
  exportMemoryInfo.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
  exportMemoryInfo.pNext = nullptr;
#endif /* _WIN64 */

  poolInfo.pMemoryAllocateNext = &exportMemoryInfo;
  vmaCreatePool(vl.allocator, &poolInfo, &externalPool); // create the pool
}

void Renderer::createFsPipeline() {
  // simple fullscreen pipeline to copy a texture to the swap chain image
  theShaderManager.addAndExecuteShaderDependency(
      fsPipeline, {"copy.frag"}, [this](auto shaders) {
        vk::PipelineRenderingCreateInfo rendering = {
            .colorAttachmentCount = 1,
            .pColorAttachmentFormats = &theWindowManager.swapChainImageFormat};
        return theVulkanLayer.name(
            thePipelineBuildHelpers.createFullscreenPipeline(
                fsPipelineLayout, rendering, shaders[0]),
            "FSCopyPipeline");
      });
}

void Renderer::createSyncObjects() {
#ifdef _WIN64
  WindowsSecurityAttributes winSecurityAttributes;
  vk::ExportSemaphoreWin32HandleInfoKHR vulkanExportSemaphoreWin32HandleInfoKHR{
      .pAttributes = &winSecurityAttributes,
      .dwAccess = DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
      .name = (LPCWSTR)NULL};
#endif /* _WIN64 */

  vk::ExportSemaphoreCreateInfoKHR vulkanExportSemaphoreCreateInfo{
#ifdef _WIN64
      .pNext = &vulkanExportSemaphoreWin32HandleInfoKHR,
      .handleTypes = vk::ExternalSemaphoreHandleTypeFlagBitsKHR::eOpaqueWin32
#else
      .handleTypes = vk::ExternalSemaphoreHandleTypeFlagBitsKHR::eOpaqueFd
#endif /* _WIN64 */
  };

  vk::SemaphoreCreateInfo semaphoreInfo{.pNext =
                                            &vulkanExportSemaphoreCreateInfo};

  cudaWaitsForVulkanSemaphore =
      theVulkanLayer.device.createSemaphore(semaphoreInfo);
  vulkanWaitsForCudaSemaphore =
      theVulkanLayer.device.createSemaphore(semaphoreInfo);

  exportSemaphoresToCuda(
      theVulkanLayer.getNativeHandle(cudaWaitsForVulkanSemaphore),
      theVulkanLayer.getNativeHandle(vulkanWaitsForCudaSemaphore));
}
