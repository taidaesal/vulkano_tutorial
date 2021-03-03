// Copyright (c) 2021 taidaesal
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>
//
// This file contains code copied and/or adapted
// from code provided by the Vulkano project under
// the MIT license

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState, SubpassContents};
use vulkano::descriptor::{descriptor_set::PersistentDescriptorSet, pipeline_layout::PipelineLayoutAbstract};
use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::{AttachmentImage, Dimensions, MipmapsCount, immutable::ImmutableImage, SwapchainImage};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::{GraphicsPipeline, viewport::Viewport};
use vulkano::sampler::{MipmapMode, Filter, Sampler, SamplerAddressMode};
use vulkano::swapchain;
use vulkano::swapchain::{AcquireError, ColorSpace, FullscreenExclusive, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError};
use vulkano::sync;
use vulkano::sync::{GpuFuture, FlushError};

use vulkano_win::VkSurfaceBuild;

use winit::window::{WindowBuilder, Window};
use winit::event_loop::{EventLoop, ControlFlow};
use winit::event::{Event, WindowEvent};

use nalgebra_glm::{identity, look_at, perspective, TMat4, vec3};

use png;

use std::io::Cursor;
use std::sync::Arc;

#[derive(Default, Debug, Clone)]
struct Vertex {
    position: [f32; 3],
    uv: [f32; 2]
}
vulkano::impl_vertex!(Vertex, position, uv);

#[derive(Debug, Clone)]
struct MVP {
    model: TMat4<f32>,
    view: TMat4<f32>,
    projection: TMat4<f32>
}

impl MVP {
    fn new() -> MVP {
        MVP {
            model: identity(),
            view: identity(),
            projection: identity()
        }
    }
}

fn main() {
    let mut mvp = MVP::new();
    mvp.view = look_at(&vec3(0.0, 0.0, 0.01), &vec3(0.0, 0.0, 0.0), &vec3(0.0, -1.0, 0.0));

    let instance = {
        let extensions = vulkano_win::required_extensions();
        Instance::new(None, &extensions, None).unwrap()
    };

    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new().build_vk_surface(&event_loop, instance.clone()).unwrap();

    let queue_family = physical.queue_families().find(|&q| {
        q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
    }).unwrap();

    let device_ext = DeviceExtensions { khr_swapchain: true, .. DeviceExtensions::none() };
    let (device, mut queues) = Device::new(physical, physical.supported_features(), &device_ext,
                                           [(queue_family, 0.5)].iter().cloned()).unwrap();

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical).unwrap();
        let usage = caps.supported_usage_flags;
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
        let dimensions: [u32; 2] = surface.window().inner_size().into();
        mvp.projection = perspective(dimensions[0] as f32 / dimensions[1] as f32, 180.0, 0.01, 100.0);

        Swapchain::new(device.clone(), surface.clone(), caps.min_image_count, format,
                       dimensions, 1, usage, &queue, SurfaceTransform::Identity, alpha,
                       PresentMode::Fifo, FullscreenExclusive::Default, true, ColorSpace::SrgbNonLinear).unwrap()
    };


    mod vs {
        vulkano_shaders::shader!{
        ty: "vertex",
        src: "
#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec2 uv;

layout(location = 0) out vec2 tex_coords;

layout(set = 0, binding = 0) uniform MVP_Data {
    mat4 model;
    mat4 view;
    mat4 projection;
} uniforms;

void main() {
    mat4 worldview = uniforms.view * uniforms.model;
    gl_Position = uniforms.projection * worldview * vec4(position, 1.0);
    tex_coords = uv;
}"
        }
    }

    mod fs {
        vulkano_shaders::shader!{
        ty: "fragment",
        src: "
#version 450
layout(location = 0) in vec2 tex_coords;

layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 1) uniform sampler2D tex;

void main() {
    f_color = texture(tex, tex_coords);
}
"
        }
    }

    let vs = vs::Shader::load(device.clone()).unwrap();
    let fs = fs::Shader::load(device.clone()).unwrap();

    let uniform_buffer = CpuBufferPool::<vs::ty::MVP_Data>::uniform_buffer(device.clone());

    let render_pass = Arc::new(vulkano::single_pass_renderpass!(device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.format(),
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: Format::D16Unorm,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {depth}
        }
    ).unwrap());

    let pipeline = Arc::new(GraphicsPipeline::start()
        .vertex_input_single_buffer()
        .vertex_shader(vs.main_entry_point(), ())
        .triangle_list()
        .viewports_dynamic_scissors_irrelevant(1)
        .fragment_shader(fs.main_entry_point(), ())
        .blend_alpha_blending()
        .depth_stencil_simple_depth()
        .front_face_counter_clockwise()
        .cull_mode_back()
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap());

    let vertex_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), false, [
        Vertex { position: [-0.5, -0.5, -0.5], uv: [0.0, 0.0] }, // top left corner
        Vertex { position: [-0.5, 0.5, -0.5], uv: [0.0, 1.0] }, // bottom left corner
        Vertex { position: [0.5, -0.5, -0.5], uv: [1.0, 0.0] }, // top right corner

        Vertex { position: [0.5, -0.5, -0.5], uv: [1.0, 0.0] }, // top right corner
        Vertex { position: [-0.5, 0.5, -0.5], uv: [0.0, 1.0] }, // bottom left corner
        Vertex { position: [0.5, 0.5, -0.5], uv: [1.0, 1.0] }, // bottom right corner
    ].iter().cloned()).unwrap();

    let mut dynamic_state = DynamicState { line_width: None, viewports: None, scissors: None, compare_mask: None, write_mask: None, reference: None };

    let mut framebuffers = window_size_dependent_setup(device.clone(), &images, render_pass.clone(), &mut dynamic_state);

    let mut recreate_swapchain = false;

    let (texture, tex_future) = {
        let png_bytes = include_bytes!("./textures/diamond.png").to_vec();
        let cursor = Cursor::new(png_bytes);
        let decoder = png::Decoder::new(cursor);
        let (info, mut reader) = decoder.read_info().unwrap();
        let mut image_data = Vec::new();
        let depth: u32 = match info.bit_depth {
            png::BitDepth::One => 1,
            png::BitDepth::Two => 2,
            png::BitDepth::Four => 4,
            png::BitDepth::Eight => 8,
            png::BitDepth::Sixteen => 16
        };
        image_data.resize((info.width * info.height * depth) as usize, 0);
        reader.next_frame(&mut image_data).unwrap();
        let dimensions = Dimensions::Dim2d {
            width: info.width,
            height: info.height,
        };
        ImmutableImage::from_iter(
            image_data.iter().cloned(),
            dimensions,
            MipmapsCount::One,
            Format::R8G8B8A8Srgb,
            queue.clone(),
        ).unwrap()
    };


    let sampler = Sampler::new(
        device.clone(),
        Filter::Linear,
        Filter::Linear,
        MipmapMode::Nearest,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        SamplerAddressMode::Repeat,
        0.0,
        1.0,
        0.0,
        0.0,
    ).unwrap();

    let mut previous_frame_end = Some(tex_future.boxed());

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit;
            },
            Event::WindowEvent { event: WindowEvent::Resized(_), .. } => {
                recreate_swapchain = true;
            },
            Event::RedrawEventsCleared => {
                previous_frame_end.as_mut().take().unwrap().cleanup_finished();

                if recreate_swapchain {
                    let dimensions: [u32; 2] = surface.window().inner_size().into();
                    let (new_swapchain, new_images) = match swapchain.recreate_with_dimensions(dimensions) {
                        Ok(r) => r,
                        Err(SwapchainCreationError::UnsupportedDimensions) => return,
                        Err(e) => panic!("Failed to recreate swapchain: {:?}", e)
                    };
                    mvp.projection = perspective(dimensions[0] as f32 / dimensions[1] as f32, 180.0, 0.01, 100.0);

                    swapchain = new_swapchain;
                    framebuffers = window_size_dependent_setup(device.clone(), &new_images, render_pass.clone(), &mut dynamic_state);
                    recreate_swapchain = false;
                }

                let (image_num, suboptimal, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    },
                    Err(e) => panic!("Failed to acquire next image: {:?}", e)
                };

                if suboptimal {
                    recreate_swapchain = true;
                }

                let clear_values = vec![[0.0, 0.68, 1.0, 1.0].into(), 1f32.into()];

                let uniform_buffer_subbuffer = {
                    let uniform_data = vs::ty::MVP_Data {
                        model: mvp.model.into(),
                        view: mvp.view.into(),
                        projection: mvp.projection.into(),
                    };

                    uniform_buffer.next(uniform_data).unwrap()
                };

                let layout = pipeline.descriptor_set_layout(0).unwrap();

                let set = Arc::new(PersistentDescriptorSet::start(layout.clone())
                    .add_buffer(uniform_buffer_subbuffer).unwrap()
                    .add_sampled_image(texture.clone(), sampler.clone()).unwrap()
                    .build().unwrap()
                );

                let mut cmd_buffer_builder = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap();
                cmd_buffer_builder
                    .begin_render_pass(framebuffers[image_num].clone(), SubpassContents::Inline, clear_values)
                    .unwrap()
                    .draw(pipeline.clone(), &dynamic_state, vertex_buffer.clone(), set.clone(), (),)
                    .unwrap()
                    .end_render_pass()
                    .unwrap();
                let command_buffer = cmd_buffer_builder.build().unwrap();

                let future = previous_frame_end.take().unwrap()
                    .join(acquire_future)
                    .then_execute(queue.clone(), command_buffer).unwrap()
                    .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                    .then_signal_fence_and_flush();

                match future {
                    Ok(future) => {
                        previous_frame_end = Some(future.boxed());
                    }
                    Err(FlushError::OutOfDate) => {
                        recreate_swapchain = true;
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                    Err(e) => {
                        println!("Failed to flush future: {:?}", e);
                        previous_frame_end = Some(sync::now(device.clone()).boxed());
                    }
                }
            },
            _ => ()
        }
    });
}

/// This method is called once during initialization, then again whenever the window is resized
/// stolen from the vulkano example
fn window_size_dependent_setup(
    device: Arc<Device>,
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dynamic_state: &mut DynamicState
) -> Vec<Arc<dyn FramebufferAbstract + Send + Sync>> {
    let dimensions = images[0].dimensions();

    let viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [dimensions[0] as f32, dimensions[1] as f32],
        depth_range: 0.0 .. 1.0,
    };
    dynamic_state.viewports = Some(vec!(viewport));

    let depth_buffer = AttachmentImage::transient(device.clone(), dimensions, Format::D16Unorm).unwrap();

    images.iter().map(|image| {
        Arc::new(
            Framebuffer::start(render_pass.clone())
                .add(image.clone()).unwrap()
                .add(depth_buffer.clone()).unwrap()
                .build().unwrap()
        ) as Arc<dyn FramebufferAbstract + Send + Sync>
    }).collect::<Vec<_>>()
}