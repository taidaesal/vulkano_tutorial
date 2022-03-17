// Copyright (c) 2022 taidaesal
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>
//
// This file contains code copied and/or adapted
// from code provided by the Vulkano project under
// the MIT license

use bytemuck::{Pod, Zeroable};
use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents};
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::{PhysicalDevice, PhysicalDeviceType};
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{AttachmentImage, ImageAccess, SwapchainImage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::Pipeline;
use vulkano::pipeline::{GraphicsPipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::swapchain::{
    self, AcquireError, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
};
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano::Version;

use vulkano_win::VkSurfaceBuild;

use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use nalgebra_glm::{
    identity, look_at, perspective, pi, rotate_normalized_axis, translate, vec3, TMat4,
};

use std::sync::Arc;
use std::time::Instant;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, Zeroable, Pod)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
}
vulkano::impl_vertex!(Vertex, position, normal, color);

#[derive(Default, Debug, Clone)]
struct AmbientLight {
    color: [f32; 3],
    intensity: f32,
}

#[derive(Default, Debug, Clone)]
struct DirectionalLight {
    position: [f32; 4],
    color: [f32; 3],
}

#[derive(Debug, Clone)]
struct MVP {
    model: TMat4<f32>,
    view: TMat4<f32>,
    projection: TMat4<f32>,
}

impl MVP {
    fn new() -> MVP {
        MVP {
            model: identity(),
            view: identity(),
            projection: identity(),
        }
    }
}

fn main() {
    let mut mvp = MVP::new();
    mvp.view = look_at(
        &vec3(0.0, 0.0, 0.01),
        &vec3(0.0, 0.0, 0.0),
        &vec3(0.0, -1.0, 0.0),
    );
    mvp.model = translate(&identity(), &vec3(0.0, 0.0, -2.5));

    let ambient_light = AmbientLight {
        color: [1.0, 1.0, 1.0],
        intensity: 0.2,
    };
    let directional_light = DirectionalLight {
        position: [-4.0, -4.0, 0.0, 1.0],
        color: [1.0, 1.0, 1.0],
    };

    let instance = {
        let extensions = vulkano_win::required_extensions();
        Instance::new(InstanceCreateInfo {
            enabled_extensions: extensions,
            max_api_version: Some(Version::V1_1),
            ..Default::default()
        })
        .unwrap()
    };

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let device_ext = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };

    let (physical_device, queue_family) = PhysicalDevice::enumerate(&instance)
        .filter(|&p| p.supported_extensions().is_superset_of(&device_ext))
        .filter_map(|p| {
            p.queue_families()
                .find(|&q| q.supports_graphics() && q.supports_surface(&surface).unwrap_or(false))
                .map(|q| (p, q))
        })
        .min_by_key(|(p, _)| match p.properties().device_type {
            PhysicalDeviceType::DiscreteGpu => 0,
            PhysicalDeviceType::IntegratedGpu => 1,
            PhysicalDeviceType::VirtualGpu => 2,
            PhysicalDeviceType::Cpu => 3,
            PhysicalDeviceType::Other => 4,
        })
        .unwrap();

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: physical_device.required_extensions().union(&device_ext),

            queue_create_infos: vec![QueueCreateInfo::family(queue_family)],
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let dim: [u32; 2] = surface.window().inner_size().into();
        let caps = physical_device
            .surface_capabilities(&surface, Default::default())
            .unwrap();
        let usage = caps.supported_usage_flags;
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let image_format = Some(
            physical_device
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0,
        );
        mvp.projection = perspective(dim[0] as f32 / dim[1] as f32, 180.0, 0.01, 100.0);

        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count,
                image_format,
                image_extent: surface.window().inner_size().into(),
                image_usage: usage,
                composite_alpha: alpha,
                ..Default::default()
            },
        )
        .unwrap()
    };

    mod deferred_vert {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "src/shaders/deferred.vert",
            types_meta: {
                use bytemuck::{Pod, Zeroable};

                #[derive(Clone, Copy, Zeroable, Pod)]
            },
        }
    }

    mod deferred_frag {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "src/shaders/deferred.frag"
        }
    }

    mod lighting_vert {
        vulkano_shaders::shader! {
            ty: "vertex",
            path: "src/shaders/lighting.vert",
            types_meta: {
                use bytemuck::{Pod, Zeroable};

                #[derive(Clone, Copy, Zeroable, Pod)]
            },
        }
    }

    mod lighting_frag {
        vulkano_shaders::shader! {
            ty: "fragment",
            path: "src/shaders/lighting.frag",
            types_meta: {
                use bytemuck::{Pod, Zeroable};

                #[derive(Clone, Copy, Zeroable, Pod)]
            },
        }
    }

    let deferred_vert = deferred_vert::load(device.clone()).unwrap();
    let deferred_frag = deferred_frag::load(device.clone()).unwrap();
    let lighting_vert = lighting_vert::load(device.clone()).unwrap();
    let lighting_frag = lighting_frag::load(device.clone()).unwrap();

    let uniform_buffer =
        CpuBufferPool::<deferred_vert::ty::MVP_Data>::uniform_buffer(device.clone());
    let ambient_buffer =
        CpuBufferPool::<lighting_frag::ty::Ambient_Data>::uniform_buffer(device.clone());
    let directional_buffer =
        CpuBufferPool::<lighting_frag::ty::Directional_Light_Data>::uniform_buffer(device.clone());

    let render_pass = vulkano::ordered_passes_renderpass!(device.clone(),
        attachments: {
            final_color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            },
            color: {
                load: Clear,
                store: DontCare,
                format: Format::A2B10G10R10_UNORM_PACK32,
                samples: 1,
            },
            normals: {
                load: Clear,
                store: DontCare,
                format: Format::R16G16B16A16_SFLOAT,
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: Format::D16_UNORM,
                samples: 1,
            }
        },
        passes: [
            {
                color: [color, normals],
                depth_stencil: {depth},
                input: []
            },
            {
                color: [final_color],
                depth_stencil: {},
                input: [color, normals]
            }
        ]
    )
    .unwrap();

    let deferred_pass = Subpass::from(render_pass.clone(), 0).unwrap();
    let lighting_pass = Subpass::from(render_pass.clone(), 1).unwrap();

    let deferred_pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(deferred_vert.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(deferred_frag.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
        .render_pass(deferred_pass)
        .build(device.clone())
        .unwrap();

    let lighting_pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(lighting_vert.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(lighting_frag.entry_point("main").unwrap(), ())
        .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
        .render_pass(lighting_pass)
        .build(device.clone())
        .unwrap();

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        [
            // front face
            Vertex {
                position: [-1.000000, -1.000000, 1.000000],
                normal: [0.0000, 0.0000, 1.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, 1.000000, 1.000000],
                normal: [0.0000, 0.0000, 1.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, 1.000000, 1.000000],
                normal: [0.0000, 0.0000, 1.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, -1.000000, 1.000000],
                normal: [0.0000, 0.0000, 1.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, 1.000000, 1.000000],
                normal: [0.0000, 0.0000, 1.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, -1.000000, 1.000000],
                normal: [0.0000, 0.0000, 1.0000],
                color: [1.0, 0.35, 0.137],
            },
            // back face
            Vertex {
                position: [1.000000, -1.000000, -1.000000],
                normal: [0.0000, 0.0000, -1.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, 1.000000, -1.000000],
                normal: [0.0000, 0.0000, -1.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, 1.000000, -1.000000],
                normal: [0.0000, 0.0000, -1.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, -1.000000, -1.000000],
                normal: [0.0000, 0.0000, -1.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, 1.000000, -1.000000],
                normal: [0.0000, 0.0000, -1.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, -1.000000, -1.000000],
                normal: [0.0000, 0.0000, -1.0000],
                color: [1.0, 0.35, 0.137],
            },
            // top face
            Vertex {
                position: [-1.000000, -1.000000, 1.000000],
                normal: [0.0000, -1.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, -1.000000, 1.000000],
                normal: [0.0000, -1.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, -1.000000, -1.000000],
                normal: [0.0000, -1.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, -1.000000, 1.000000],
                normal: [0.0000, -1.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, -1.000000, -1.000000],
                normal: [0.0000, -1.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, -1.000000, -1.000000],
                normal: [0.0000, -1.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            // bottom face
            Vertex {
                position: [1.000000, 1.000000, 1.000000],
                normal: [0.0000, 1.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, 1.000000, 1.000000],
                normal: [0.0000, 1.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, 1.000000, -1.000000],
                normal: [0.0000, 1.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, 1.000000, 1.000000],
                normal: [0.0000, 1.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, 1.000000, -1.000000],
                normal: [0.0000, 1.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, 1.000000, -1.000000],
                normal: [0.0000, 1.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            // left face
            Vertex {
                position: [-1.000000, -1.000000, -1.000000],
                normal: [-1.0000, 0.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, 1.000000, -1.000000],
                normal: [-1.0000, 0.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, 1.000000, 1.000000],
                normal: [-1.0000, 0.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, -1.000000, -1.000000],
                normal: [-1.0000, 0.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, 1.000000, 1.000000],
                normal: [-1.0000, 0.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [-1.000000, -1.000000, 1.000000],
                normal: [-1.0000, 0.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            // right face
            Vertex {
                position: [1.000000, -1.000000, 1.000000],
                normal: [1.0000, 0.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, 1.000000, 1.000000],
                normal: [1.0000, 0.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, 1.000000, -1.000000],
                normal: [1.0000, 0.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, -1.000000, 1.000000],
                normal: [1.0000, 0.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, 1.000000, -1.000000],
                normal: [1.0000, 0.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
            Vertex {
                position: [1.000000, -1.000000, -1.000000],
                normal: [1.0000, 0.0000, 0.0000],
                color: [1.0, 0.35, 0.137],
            },
        ]
        .iter()
        .cloned(),
    )
    .unwrap();

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    let (mut framebuffers, mut color_buffer, mut normal_buffer) =
        window_size_dependent_setup(device.clone(), &images, render_pass.clone(), &mut viewport);

    let mut recreate_swapchain = false;

    let mut previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>);

    let rotation_start = Instant::now();

    event_loop.run(move |event, _, control_flow| match event {
        Event::WindowEvent {
            event: WindowEvent::CloseRequested,
            ..
        } => {
            *control_flow = ControlFlow::Exit;
        }
        Event::WindowEvent {
            event: WindowEvent::Resized(_),
            ..
        } => {
            recreate_swapchain = true;
        }
        Event::RedrawEventsCleared => {
            previous_frame_end
                .as_mut()
                .take()
                .unwrap()
                .cleanup_finished();

            if recreate_swapchain {
                let dimensions: [u32; 2] = surface.window().inner_size().into();
                let (new_swapchain, new_images) = match swapchain.recreate(SwapchainCreateInfo {
                    image_extent: surface.window().inner_size().into(),
                    ..swapchain.create_info()
                }) {
                    Ok(r) => r,
                    Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                    Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                };
                mvp.projection = perspective(
                    dimensions[0] as f32 / dimensions[1] as f32,
                    180.0,
                    0.01,
                    100.0,
                );

                swapchain = new_swapchain;
                let (new_framebuffers, new_color_buffer, new_normal_buffer) =
                    window_size_dependent_setup(
                        device.clone(),
                        &new_images,
                        render_pass.clone(),
                        &mut viewport,
                    );
                framebuffers = new_framebuffers;
                color_buffer = new_color_buffer;
                normal_buffer = new_normal_buffer;
                recreate_swapchain = false;
            }

            let (image_num, suboptimal, acquire_future) =
                match swapchain::acquire_next_image(swapchain.clone(), None) {
                    Ok(r) => r,
                    Err(AcquireError::OutOfDate) => {
                        recreate_swapchain = true;
                        return;
                    }
                    Err(e) => panic!("Failed to acquire next image: {:?}", e),
                };

            if suboptimal {
                recreate_swapchain = true;
            }

            let clear_values = vec![
                [0.0, 0.0, 0.0, 1.0].into(),
                [0.0, 0.0, 0.0, 1.0].into(),
                [0.0, 0.0, 0.0, 1.0].into(),
                1f32.into(),
            ];

            let uniform_buffer_subbuffer = {
                let elapsed = rotation_start.elapsed().as_secs() as f64
                    + rotation_start.elapsed().subsec_nanos() as f64 / 1_000_000_000.0;
                let elapsed_as_radians = elapsed * pi::<f64>() / 180.0;
                let mut model: TMat4<f32> = rotate_normalized_axis(
                    &identity(),
                    elapsed_as_radians as f32 * 50.0,
                    &vec3(0.0, 0.0, 1.0),
                );
                model = rotate_normalized_axis(
                    &model,
                    elapsed_as_radians as f32 * 30.0,
                    &vec3(0.0, 1.0, 0.0),
                );
                model = rotate_normalized_axis(
                    &model,
                    elapsed_as_radians as f32 * 20.0,
                    &vec3(1.0, 0.0, 0.0),
                );
                model = mvp.model * model;

                let uniform_data = deferred_vert::ty::MVP_Data {
                    model: model.into(),
                    view: mvp.view.into(),
                    projection: mvp.projection.into(),
                };

                uniform_buffer.next(uniform_data).unwrap()
            };

            let ambient_uniform_subbuffer = {
                let uniform_data = lighting_frag::ty::Ambient_Data {
                    color: ambient_light.color.into(),
                    intensity: ambient_light.intensity.into(),
                };

                ambient_buffer.next(uniform_data).unwrap()
            };

            let directional_uniform_subbuffer = {
                let uniform_data = lighting_frag::ty::Directional_Light_Data {
                    position: directional_light.position.into(),
                    color: directional_light.color.into(),
                };

                directional_buffer.next(uniform_data).unwrap()
            };

            let deferred_layout = deferred_pipeline.layout().set_layouts().get(0).unwrap();
            let deferred_set = PersistentDescriptorSet::new(
                deferred_layout.clone(),
                [WriteDescriptorSet::buffer(
                    0,
                    uniform_buffer_subbuffer.clone(),
                )],
            )
            .unwrap();

            let lighting_layout = lighting_pipeline.layout().set_layouts().get(0).unwrap();
            let lighting_set = PersistentDescriptorSet::new(
                lighting_layout.clone(),
                [
                    WriteDescriptorSet::image_view(0, color_buffer.clone()),
                    WriteDescriptorSet::image_view(1, normal_buffer.clone()),
                    WriteDescriptorSet::buffer(2, uniform_buffer_subbuffer),
                    WriteDescriptorSet::buffer(3, ambient_uniform_subbuffer),
                    WriteDescriptorSet::buffer(4, directional_uniform_subbuffer),
                ],
            )
            .unwrap();

            let mut cmd_buffer_builder = AutoCommandBufferBuilder::primary(
                device.clone(),
                queue.family(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();
            cmd_buffer_builder
                .begin_render_pass(
                    framebuffers[image_num].clone(),
                    SubpassContents::Inline,
                    clear_values,
                )
                .unwrap()
                .set_viewport(0, [viewport.clone()])
                .bind_pipeline_graphics(deferred_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    deferred_pipeline.layout().clone(),
                    0,
                    deferred_set.clone(),
                )
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .draw(vertex_buffer.len() as u32, 1, 0, 0)
                .unwrap()
                .next_subpass(SubpassContents::Inline)
                .unwrap()
                .bind_pipeline_graphics(lighting_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    lighting_pipeline.layout().clone(),
                    0,
                    lighting_set.clone(),
                )
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .draw(vertex_buffer.len() as u32, 1, 0, 0)
                .unwrap()
                .end_render_pass()
                .unwrap();
            let command_buffer = cmd_buffer_builder.build().unwrap();

            let future = previous_frame_end
                .take()
                .unwrap()
                .join(acquire_future)
                .then_execute(queue.clone(), command_buffer)
                .unwrap()
                .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
                .then_signal_fence_and_flush();

            match future {
                Ok(future) => {
                    previous_frame_end = Some(Box::new(future) as Box<_>);
                }
                Err(FlushError::OutOfDate) => {
                    recreate_swapchain = true;
                    previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
                }
                Err(e) => {
                    println!("Failed to flush future: {:?}", e);
                    previous_frame_end = Some(Box::new(sync::now(device.clone())) as Box<_>);
                }
            }
        }
        _ => (),
    });
}

/// This method is called once during initialization, then again whenever the window is resized
/// stolen from the vulkano example
fn window_size_dependent_setup(
    device: Arc<Device>,
    images: &[Arc<SwapchainImage<Window>>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> (
    Vec<Arc<Framebuffer>>,
    Arc<ImageView<AttachmentImage>>,
    Arc<ImageView<AttachmentImage>>,
) {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];
    let depth_buffer = ImageView::new_default(
        AttachmentImage::transient(device.clone(), dimensions, Format::D16_UNORM).unwrap(),
    )
    .unwrap();
    let color_buffer = ImageView::new_default(
        AttachmentImage::transient_input_attachment(
            device.clone(),
            dimensions,
            Format::A2B10G10R10_UNORM_PACK32,
        )
        .unwrap(),
    )
    .unwrap();
    let normal_buffer = ImageView::new_default(
        AttachmentImage::transient_input_attachment(
            device.clone(),
            dimensions,
            Format::R16G16B16A16_SFLOAT,
        )
        .unwrap(),
    )
    .unwrap();

    (
        images
            .iter()
            .map(|image| {
                let view = ImageView::new_default(image.clone()).unwrap();
                Framebuffer::new(
                    render_pass.clone(),
                    FramebufferCreateInfo {
                        attachments: vec![
                            view,
                            color_buffer.clone(),
                            normal_buffer.clone(),
                            depth_buffer.clone(),
                        ],
                        ..Default::default()
                    },
                )
                .unwrap()
            })
            .collect::<Vec<_>>(),
        color_buffer.clone(),
        normal_buffer.clone(),
    )
}
