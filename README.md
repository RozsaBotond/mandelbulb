# Mandelbulb

The Mandelbulb ray marching with cuda project.

## Theory

[blog about mandelbulb](https://www.skytopia.com/project/fractal/2mandelbulb.html)

## Implementation

Implementation starts out of the Vulkan Cuda interoperability laboratory.

The visuals are done via ray marching, which is a technique to render implicit surfaces. The main idea is to march along the ray and check for intersection with the surface. The distance to the surface is calculated using a distance estimator, which is a function that returns the distance from a point to the surface. The ray is marched until it hits the surface or goes too far away.

## Video of running it

https://github.com/user-attachments/assets/e53dd7e2-bd65-4791-84ac-9f532cae4853

- [Demo video](media/mandelbulb_demo.mp4)
- [Screenshot](media/mandelbulb_screenshot.png)

## Important files

Files marked ⭐ are where the Mandelbulb and the Vulkan–CUDA interop live; everything else is framework code from the lab.

```text
mandelbulb/
├── README.md
├── media/                          # demo video + screenshot
└── vulkancudainterop/
    ├── CMakeLists.txt              # build setup (CUDA files listed in CUDA_FILES)
    ├── wd/                         # working directory – run the program from here
    │   └── models/                 # cached voxelized Mandelbulb models
    ├── thirdparty/                 # glfw, glm, imgui, vma, tinygltf, ... (not ours)
    └── src/
        ├── cuda/
        │   ├── interop.cuh         ⭐ CUDA API used by the C++ side (MandelbulbRenderParams,
        │   │                          export*/renderCuda/voxelize* declarations)
        │   └── interop.cu          ⭐ ALL CUDA code:
        │                              - mandelbulbDE        distance estimator
        │                              - computeVoxelField   voxelizes the fractal into a 3D texture
        │                              - renderToSurface     ray marching kernel (normals, soft shadows)
        │                              - import of Vulkan memory/semaphores (cudaImportExternal*)
        │                              - renderCuda          wait → launch kernel → signal
        ├── rendering/
        │   ├── renderer.cpp        ⭐ interop setup + per-frame driver:
        │   │                          createImages      → exportVulkanImageToCuda_R8G8B8A8Unorm
        │   │                          createSyncObjects → exportSemaphoresToCuda
        │   │                          render            → voxelize / renderCuda
        │   ├── renderer.h          ⭐ fractal params, shared image + semaphores
        │   └── ...                 # render pass / common helpers
        ├── shaders/
        │   ├── copy.frag           ⭐ samples the image CUDA wrote and puts it on screen
        │   ├── full_screen_quad.vert  # vertex shader for copy.frag
        │   └── ...                 # forward shading shaders (lab framework)
        ├── vulkan_layer.cpp        ⭐ exports Vulkan memory / semaphores as FD (Linux) or Win32 handles
        ├── vulkan_engine.cpp       ⭐ main loop; queue submit waits on CUDA, signals CUDA
        ├── main.cpp                # entry point
        ├── application.cpp         # app setup
        ├── window_manager.cpp      # GLFW window
        ├── shader_manager.cpp      # shader compilation
        ├── gui/                    # ImGui panels
        ├── camera/                 # perspective camera
        ├── asset_managment/        # mesh/texture loading (lab framework)
        ├── resources/              # GPU resource helpers
        ├── scene/                  # scene objects, lights (lab framework)
        └── utility/                # math + misc helpers
```

### Interoperability flow in one frame

1. Vulkan submit signals `cudaWaitsForVulkanSemaphore` (`vulkan_engine.cpp`)
2. `renderCuda` waits on it, runs `renderToSurface` into the shared image, signals `vulkanWaitsForCudaSemaphore` (`interop.cu`)
3. Next Vulkan submit waits on `vulkanWaitsForCudaSemaphore`, then `copy.frag` draws the image to the swapchain

## Building
with cmake and make, you can build the project by running the following commands:

```bash
mkdir vulkancudainterop/build
cd vulkancudainterop/build
cmake ..
make -j $(nproc)
```

## Running
After building, you can run the project by executing the following command (from the root of the project):
```bash
cd vulkancudainterop/wd
../build/vulkanCudaInterop
```
