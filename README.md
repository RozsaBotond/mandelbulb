* Mandelbulb

The Mandelbulb ray marching with cuda project.

** Theory

[blog about mandelbulb](<<https://www.skytopia.com/project/fractal/2mandelbulb.html>)

** Implementation

Implementation starts out of the Vulkan Cuda interopt labritory.

The visuals are done via ray marching, which is a technique to render implicit surfaces. The main idea is to march along the ray and check for intersection with the surface. The distance to the surface is calculated using a distance estimator, which is a function that returns the distance from a point to the surface. The ray is marched until it hits the surface or goes too far away.

** Video of running it

[![Mandelbulb](media/mandelbulb_screenshot.png)](media/mandelbulb_demo.mp4)

- [Demo video](media/mandelbulb_demo.mp4)
- [Screenshot](media/mandelbulb_screenshot.png)

** Building
with cmake and make, you can build the project by running the following commands:

```bash
mkdir vulkancudainterop/build
cd vulkancudainterop/build
cmake ..
make -j {nproc}
```

** Running
After building, you can run the project by executing the following command (from the root of the project):
```bash
cd vulkancudainterop/wd
../build/vulkanCudaInterop
```
