// Copyright (c) 2022 taidaesal
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>
//
// This file contains code copied and/or adapted
// from code provided by the Vulkano project under
// the MIT license

use bytemuck::{Pod, Zeroable};

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess};
use vulkano::command_buffer::allocator::StandardCommandBufferAllocator;
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, RenderPassBeginInfo, SubpassContents,
};
use vulkano::descriptor_set::allocator::StandardDescriptorSetAllocator;
use vulkano::descriptor_set::{PersistentDescriptorSet, WriteDescriptorSet};
use vulkano::device::physical::PhysicalDeviceType;
use vulkano::device::{Device, DeviceCreateInfo, DeviceExtensions, QueueCreateInfo};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{AttachmentImage, ImageAccess, SwapchainImage};
use vulkano::instance::{Instance, InstanceCreateInfo};
use vulkano::memory::allocator::StandardMemoryAllocator;
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, FramebufferCreateInfo, RenderPass, Subpass};
use vulkano::swapchain::{
    self, AcquireError, Swapchain, SwapchainCreateInfo, SwapchainCreationError,
    SwapchainPresentInfo,
};
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano::{Version, VulkanLibrary};

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
        let library = VulkanLibrary::new().unwrap();
        let extensions = vulkano_win::required_extensions(&library);

        Instance::new(
            library,
            InstanceCreateInfo {
                enabled_extensions: extensions,
                enumerate_portability: true, // required for MoltenVK on macOS
                max_api_version: Some(Version::V1_1),
                ..Default::default()
            },
        )
        .unwrap()
    };

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let device_extensions = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::empty()
    };

    let (physical_device, queue_family_index) = instance
        .enumerate_physical_devices()
        .unwrap()
        .filter(|p| p.supported_extensions().contains(&device_extensions))
        .filter_map(|p| {
            p.queue_family_properties()
                .iter()
                .enumerate()
                .position(|(i, q)| {
                    // pick first queue_familiy_index that handles graphics and can draw on the surface created by winit
                    q.queue_flags.graphics && p.surface_support(i as u32, &surface).unwrap_or(false)
                })
                .map(|i| (p, i as u32))
        })
        .min_by_key(|(p, _)| {
            // lower score for preferred device types
            match p.properties().device_type {
                PhysicalDeviceType::DiscreteGpu => 0,
                PhysicalDeviceType::IntegratedGpu => 1,
                PhysicalDeviceType::VirtualGpu => 2,
                PhysicalDeviceType::Cpu => 3,
                PhysicalDeviceType::Other => 4,
                _ => 5,
            }
        })
        .expect("No suitable physical device found");

    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            enabled_extensions: device_extensions,
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = device
            .physical_device()
            .surface_capabilities(&surface, Default::default())
            .unwrap();

        let usage = caps.supported_usage_flags;
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        let image_format = Some(
            device
                .physical_device()
                .surface_formats(&surface, Default::default())
                .unwrap()[0]
                .0,
        );

        let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
        let image_extent: [u32; 2] = window.inner_size().into();

        let aspect_ratio = image_extent[0] as f32 / image_extent[1] as f32;
        mvp.projection = perspective(aspect_ratio, 180.0, 0.01, 100.0);

        Swapchain::new(
            device.clone(),
            surface.clone(),
            SwapchainCreateInfo {
                min_image_count: caps.min_image_count,
                image_format,
                image_extent,
                image_usage: usage,
                composite_alpha: alpha,
                ..Default::default()
            },
        )
        .unwrap()
    };

    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));
    let descriptor_set_allocator = StandardDescriptorSetAllocator::new(device.clone());
    let command_buffer_allocator =
        StandardCommandBufferAllocator::new(device.clone(), Default::default());

    mod vs {
        vulkano_shaders::shader! {
            ty: "vertex",
            src: "
                #version 450
                layout(location = 0) in vec3 position;
                layout(location = 1) in vec3 normal;
                layout(location = 2) in vec3 color;

                layout(location = 0) out vec3 out_color;
                layout(location = 1) out vec3 out_normal;
                layout(location = 2) out vec3 frag_pos;

                layout(set = 0, binding = 0) uniform MVP_Data {
                    mat4 model;
                    mat4 view;
                    mat4 projection;
                } uniforms;

                void main() {
                    mat4 worldview = uniforms.view * uniforms.model;
                    gl_Position = uniforms.projection * worldview * vec4(position, 1.0);
                    out_color = color;
                    out_normal = mat3(uniforms.model) * normal;
                    frag_pos = vec3(uniforms.model * vec4(position, 1.0));
                }
            ",
            types_meta: {
                use bytemuck::{Pod, Zeroable};

                #[derive(Clone, Copy, Zeroable, Pod)]
            },
        }
    }

    mod fs {
        vulkano_shaders::shader! {
            ty: "fragment",
            src: "
                #version 450
                layout(location = 0) in vec3 in_color;
                layout(location = 1) in vec3 in_normal;
                layout(location = 2) in vec3 frag_pos;

                layout(location = 0) out vec4 f_color;

                layout(set = 0, binding = 1) uniform Ambient_Data {
                    vec3 color;
                    float intensity;
                } ambient;

                layout(set = 0, binding = 2) uniform Directional_Light_Data {
                    vec4 position;
                    vec3 color;
                } directional;

                void main() {
                    vec3 ambient_color = ambient.intensity * ambient.color;
                    vec3 light_direction = normalize(directional.position.xyz - frag_pos);
                    float directional_intensity = max(dot(in_normal, light_direction), 0.0);
                    vec3 directional_color = directional_intensity * directional.color;
                    vec3 combined_color = (ambient_color + directional_color) * in_color;
                    f_color = vec4(combined_color, 1.0);
                }
            ",
            types_meta: {
                use bytemuck::{Pod, Zeroable};

                #[derive(Clone, Copy, Zeroable, Pod)]
            }
        }
    }

    let vs = vs::load(device.clone()).unwrap();
    let fs = fs::load(device.clone()).unwrap();

    let render_pass = vulkano::single_pass_renderpass!(device.clone(),
        attachments: {
            color: {
                load: Clear,
                store: Store,
                format: swapchain.image_format(),
                samples: 1,
            },
            depth: {
                load: Clear,
                store: DontCare,
                format: Format::D16_UNORM,
                samples: 1,
            }
        },
        pass: {
            color: [color],
            depth_stencil: {depth}
        }
    )
    .unwrap();

    let pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<Vertex>())
        .vertex_shader(vs.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(fs.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
        .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
        .build(device.clone())
        .unwrap();

    let uniform_buffer: CpuBufferPool<vs::ty::MVP_Data> =
        CpuBufferPool::uniform_buffer(memory_allocator.clone());
    let ambient_buffer: CpuBufferPool<fs::ty::Ambient_Data> =
        CpuBufferPool::uniform_buffer(memory_allocator.clone());
    let directional_buffer: CpuBufferPool<fs::ty::Directional_Light_Data> =
        CpuBufferPool::uniform_buffer(memory_allocator.clone());

    let vertices = [
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
    ];

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        &memory_allocator,
        BufferUsage {
            vertex_buffer: true,
            ..BufferUsage::empty()
        },
        false,
        vertices,
    )
    .unwrap();

    let mut viewport = Viewport {
        origin: [0.0, 0.0],
        dimensions: [0.0, 0.0],
        depth_range: 0.0..1.0,
    };

    let mut framebuffers = window_size_dependent_setup(
        &memory_allocator,
        &images,
        render_pass.clone(),
        &mut viewport,
    );

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
                let window = surface.object().unwrap().downcast_ref::<Window>().unwrap();
                let image_extent: [u32; 2] = window.inner_size().into();

                let aspect_ratio = image_extent[0] as f32 / image_extent[1] as f32;
                mvp.projection = perspective(aspect_ratio, 180.0, 0.01, 100.0);

                let (new_swapchain, new_images) = match swapchain.recreate(SwapchainCreateInfo {
                    image_extent,
                    ..swapchain.create_info()
                }) {
                    Ok(r) => r,
                    Err(SwapchainCreationError::ImageExtentNotSupported { .. }) => return,
                    Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                };

                swapchain = new_swapchain;
                framebuffers = window_size_dependent_setup(
                    &memory_allocator,
                    &new_images,
                    render_pass.clone(),
                    &mut viewport,
                );
                recreate_swapchain = false;
            }

            let (image_index, suboptimal, acquire_future) =
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

            let clear_values = vec![Some([0.0, 0.68, 1.0, 1.0].into()), Some(1.0.into())];

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

                let uniform_data = vs::ty::MVP_Data {
                    model: model.into(),
                    view: mvp.view.into(),
                    projection: mvp.projection.into(),
                };

                uniform_buffer.from_data(uniform_data).unwrap()
            };

            let ambient_uniform_subbuffer = {
                let uniform_data = fs::ty::Ambient_Data {
                    color: ambient_light.color.into(),
                    intensity: ambient_light.intensity.into(),
                };

                ambient_buffer.from_data(uniform_data).unwrap()
            };

            let directional_uniform_subbuffer = {
                let uniform_data = fs::ty::Directional_Light_Data {
                    position: directional_light.position.into(),
                    color: directional_light.color.into(),
                };

                directional_buffer.from_data(uniform_data).unwrap()
            };

            let layout = pipeline.layout().set_layouts().get(0).unwrap();
            let set = PersistentDescriptorSet::new(
                &descriptor_set_allocator,
                layout.clone(),
                [
                    WriteDescriptorSet::buffer(0, uniform_buffer_subbuffer),
                    WriteDescriptorSet::buffer(1, ambient_uniform_subbuffer),
                    WriteDescriptorSet::buffer(2, directional_uniform_subbuffer),
                ],
            )
            .unwrap();

            let mut cmd_buffer_builder = AutoCommandBufferBuilder::primary(
                &command_buffer_allocator,
                queue.queue_family_index(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();

            cmd_buffer_builder
                .begin_render_pass(
                    RenderPassBeginInfo {
                        clear_values,
                        ..RenderPassBeginInfo::framebuffer(
                            framebuffers[image_index as usize].clone(),
                        )
                    },
                    SubpassContents::Inline,
                )
                .unwrap()
                .set_viewport(0, [viewport.clone()])
                .bind_pipeline_graphics(pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    pipeline.layout().clone(),
                    0,
                    set.clone(),
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
                .then_swapchain_present(
                    queue.clone(),
                    SwapchainPresentInfo::swapchain_image_index(swapchain.clone(), image_index),
                )
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
    allocator: &StandardMemoryAllocator,
    images: &[Arc<SwapchainImage>],
    render_pass: Arc<RenderPass>,
    viewport: &mut Viewport,
) -> Vec<Arc<Framebuffer>> {
    let dimensions = images[0].dimensions().width_height();
    viewport.dimensions = [dimensions[0] as f32, dimensions[1] as f32];
    let depth_buffer = ImageView::new_default(
        AttachmentImage::transient(allocator, dimensions, Format::D16_UNORM).unwrap(),
    )
    .unwrap();

    images
        .iter()
        .map(|image| {
            let view = ImageView::new_default(image.clone()).unwrap();
            Framebuffer::new(
                render_pass.clone(),
                FramebufferCreateInfo {
                    attachments: vec![view, depth_buffer.clone()],
                    ..Default::default()
                },
            )
            .unwrap()
        })
        .collect::<Vec<_>>()
}
