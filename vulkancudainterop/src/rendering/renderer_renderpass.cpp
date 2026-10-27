#include "rendering/renderer.h"
#include "window_manager.h"

void Renderer::createFramebuffers()
{
    auto& wm = theWindowManager;
    for (int i = 0; i < wm.swapChainImageViews.size(); i++) {
        std::array<vk::ImageView, 1> views = {
            wm.swapChainImageViews[i]
        };

        vk::FramebufferCreateInfo framebufferInfo = {}; framebufferInfo
            .setRenderPass(renderPass)
            .setAttachments(views)
            .setWidth(wm.swapChainExtent.width)
            .setHeight(wm.swapChainExtent.height)
            .setLayers(1);
        framebuffer.push_back(theVulkanLayer.device.createFramebuffer(framebufferInfo));
    }
}

void Renderer::createRenderPass()
{
    auto& vl = theVulkanLayer;
    auto& wm = theWindowManager;

    vk::AttachmentDescription attachments[1] = {};
    attachments[0]
        .setFormat(wm.swapChainImageFormat)
        .setSamples(vk::SampleCountFlagBits::e1)
        .setLoadOp(vk::AttachmentLoadOp::eClear)
        .setStoreOp(vk::AttachmentStoreOp::eStore)
        .setStencilLoadOp(vk::AttachmentLoadOp::eDontCare)
        .setStencilStoreOp(vk::AttachmentStoreOp::eDontCare)
        .setInitialLayout(vk::ImageLayout::eColorAttachmentOptimal)
        .setFinalLayout(vk::ImageLayout::eColorAttachmentOptimal);

    vk::AttachmentReference color_attachment = {}; color_attachment
        .setAttachment(0)
        .setLayout(vk::ImageLayout::eColorAttachmentOptimal);


    vk::SubpassDescription subpass = {}; subpass
        .setPipelineBindPoint(vk::PipelineBindPoint::eGraphics)
        .setColorAttachments(color_attachment);

    vk::RenderPassCreateInfo info = {}; info
        .setAttachments(attachments)
        .setSubpasses(subpass)
        .setDependencies(nullptr);
    renderPass = vl.name(vl.device.createRenderPass(info), "MainRenderPass");
}

void Renderer::startRenderpass(const RenderContext& ctx)
{
    vk::ClearValue clearValues[1] = {
        vk::ClearColorValue{ 0, 0, 0, 0 }
    };

    vk::RenderPassBeginInfo renderPassBeginInfo = {}; renderPassBeginInfo
        .setFramebuffer(framebuffer[ctx.imageID])
        .setRenderPass(renderPass)
        .setRenderArea({ { 0, 0 } /* offset */, theWindowManager.swapChainExtent /* extent */ })
        .setClearValues(clearValues);
    ctx.cmd.beginRenderPass(renderPassBeginInfo, vk::SubpassContents::eInline);
}
