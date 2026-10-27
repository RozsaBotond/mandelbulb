#include "asset_managment/mesh_traits.h"
#include "constants.h"
#include "rendering/common_rendering.h"
#include "rendering/render_context.h"
#include "shader_manager.h"
#include "utility/utility.hpp"

struct RendererDependency;

class Renderer {
public:
  Renderer() : perFrameDscSetBuilder{descriptorPoolBuilder} {}
  void initialize();
  void destroy();
  void initializeSwapChainDependentComponents();
  void destroySwapChainDependentComponents();
  void render(const RenderContext &ctx);
  void update(double dt);

private:
  struct PerFrameUniformData {
    glm::mat4 v;
    glm::vec4 ambientLight;
  };

  struct alignas(16) PerObjectUniformData {
    glm::mat4 mv, mvp, mvInvT;
    int materialIndex;
  };

  struct alignas(16) PerPointLightUniformData {
    glm::vec4 position, power;
  };

  static const size_t MAX_POINTLIGHT_COUNT = 50;
  struct PointLightsUBO {
    glm::uvec4 lightCount_pad3;
    PerPointLightUniformData pointLights[MAX_POINTLIGHT_COUNT];
  };

  void createPipelineLayout();
  void createFsPipeline();
  void createSyncObjects();
  void createResources();
  void createImages();
  void updateDescriptorSets();
  void updateFractalCameraVectors();

private:
  VulkanPipeline fsPipeline;
  vk::PipelineLayout fsPipelineLayout;
  VulkanImage colorImage;
  NativeHandle colorImageNativeHandle;
  QueryInfo<2> query;
  vk::Sampler fsSampler;
  std::array<VulkanBuffer, MAX_FRAMES_IN_FLIGHT> perFrameBuffers;
  DescriptorPoolBuilder descriptorPoolBuilder;
  DescriptorSetBuilder perFrameDscSetBuilder;
  std::array<vk::DescriptorSet, MAX_FRAMES_IN_FLIGHT> perFrameDescriptorSets;

  // for vma pool
  // vma allocates memory by default from a standard pool
  // to assign extension structures describing an external allocation we need to
  // create a custom pool the custom pool uses these extension structures for
  // each allocation so they need to be alive as long as the pool is alive
#ifdef _WIN64
  WindowsSecurityAttributes winSecurityAttributes;
  VkExportMemoryWin32HandleInfoKHR vulkanExportMemoryWin32HandleInfoKHR;
#endif /* _WIN64 */
  VkExportMemoryAllocateInfoKHR exportMemoryInfo;
  VmaPool externalPool;

  // Dedicated camera/navigation state for Mandelbulb rendering.
  glm::vec3 fractalCameraPosition{3.5f, 2.0f, 3.5f};
  glm::vec3 fractalTarget{0.0f, 0.0f, 0.0f};
  glm::vec3 fractalCameraForward{-0.6f, -0.3f, -0.7f};
  glm::vec3 fractalCameraRight{1.0f, 0.0f, 0.0f};
  glm::vec3 fractalCameraUp{0.0f, 1.0f, 0.0f};
  float orbitYaw = -135.0f;
  float orbitPitch = -22.0f;
  float orbitRadius = 5.3f;
  float movementSpeed = 2.8f;
  float scrollZoomSpeed = 0.5f;
  float velocityRightAxis = 0.0f;
  float velocityUpAxis = 0.0f;

  float fractalPower = 8.0f;
  float fractalZoom = 1.0f;
  float fractalBailout = 8.0f;
  float hitEpsilon = 0.0015f;
  float normalEpsilon = 0.0008f;
  float maxDistance = 50.0f;
  int maxMarchSteps = 180;
  int maxFractalIterations = 12;
  int enableLighting = 1;
  int enableShadows = 1;
  int shadowSteps = 20;
  float shadowStrength = 8.0f;

  int voxelResolution = 256;
  float voxelBounds = 1.25f;
  glm::vec3 voxelCenter{0.0f, 0.0f, 0.0f};
  bool voxelDataValid = false;
  bool forceVoxelRegeneration = false;
  bool voxelizing = false;
  int currentVoxelZ = 0;
  float voxelizationProgress = 0.0f;
  std::string voxelErrorMessage = "";

  int autoDetailFromZoom = 1;

  int useFrameCache = 0;
  bool cacheValid = false;
  bool forceCacheRefresh = true;

public:
  // cuda - vulkan synchronization
  // LABTODO: these must be used in VulkanEngine
  vk::Semaphore cudaWaitsForVulkanSemaphore, vulkanWaitsForCudaSemaphore;

private:
  // no dynamic rendering
  void createRenderPass();
  void createFramebuffers();
  void startRenderpass(const RenderContext &ctx);
  bool firstSwapChainCreate = true;
  vk::RenderPass renderPass;
  std::vector<vk::Framebuffer> framebuffer;
};
