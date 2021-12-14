// Copyright (c) 2021 taidaesal
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>
//
// This file contains code copied and/or adapted
// from code provided by the Vulkano project under
// the MIT license

mod model;
mod obj_loader;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, TypedBufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, CommandBufferUsage, SubpassContents};
use vulkano::descriptor_set::PersistentDescriptorSet;
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::Format;
use vulkano::image::attachment::AttachmentImage;
use vulkano::image::view::ImageView;
use vulkano::image::{ImageAccess, SwapchainImage};
use vulkano::instance::Instance;
use vulkano::memory::pool::StdMemoryPool;
use vulkano::pipeline::graphics::color_blend::{
    AttachmentBlend, BlendFactor, BlendOp, ColorBlendState,
};
use vulkano::pipeline::graphics::depth_stencil::DepthStencilState;
use vulkano::pipeline::graphics::input_assembly::InputAssemblyState;
use vulkano::pipeline::graphics::rasterization::{CullMode, RasterizationState};
use vulkano::pipeline::graphics::vertex_input::BuffersDefinition;
use vulkano::pipeline::graphics::viewport::{Viewport, ViewportState};
use vulkano::pipeline::{GraphicsPipeline, Pipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, RenderPass, Subpass};
use vulkano::swapchain::{self, AcquireError, Swapchain, SwapchainCreationError};
use vulkano::sync::{self, FlushError, GpuFuture};
use vulkano::Version;

use vulkano_win::VkSurfaceBuild;

use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};
use winit::window::{Window, WindowBuilder};

use nalgebra_glm::{identity, look_at, perspective, pi, vec3, TMat4};

use model::Model;
use obj_loader::{DummyVertex, NormalVertex};

use std::sync::Arc;
use std::time::Instant;

vulkano::impl_vertex!(NormalVertex, position, normal, color);
vulkano::impl_vertex!(DummyVertex, position);

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
struct VP {
    view: TMat4<f32>,
    projection: TMat4<f32>,
}

impl VP {
    fn new() -> VP {
        VP {
            view: identity(),
            projection: identity(),
        }
    }
}

mod deferred_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/deferred.vert"
    }
}

mod deferred_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/deferred.frag"
    }
}

mod directional_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/directional.vert"
    }
}

mod directional_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/directional.frag"
    }
}

mod ambient_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/shaders/ambient.vert"
    }
}

mod ambient_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/shaders/ambient.frag"
    }
}

fn main() {
    let mut model = Model::new("data/models/teapot.obj").build();
    let mut vp = VP::new();
    vp.view = look_at(
        &vec3(0.0, 0.0, 0.01),
        &vec3(0.0, 0.0, 0.0),
        &vec3(0.0, -1.0, 0.0),
    );
    model.translate(vec3(0.0, 0.0, -3.5));

    let ambient_light = AmbientLight {
        color: [1.0, 1.0, 1.0],
        intensity: 0.1,
    };
    let directional_light = DirectionalLight {
        position: [-4.0, -4.0, 0.0, 1.0],
        color: [1.0, 1.0, 1.0],
    };

    let instance = {
        let extensions = vulkano_win::required_extensions();
        Instance::new(None, Version::V1_1, &extensions, None).unwrap()
    };

    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

    let event_loop = EventLoop::new();
    let surface = WindowBuilder::new()
        .build_vk_surface(&event_loop, instance.clone())
        .unwrap();

    let queue_family = physical
        .queue_families()
        .find(|&q| q.supports_graphics() && surface.is_supported(q).unwrap_or(false))
        .unwrap();

    let device_ext = DeviceExtensions {
        khr_swapchain: true,
        ..DeviceExtensions::none()
    };
    let (device, mut queues) = Device::new(
        physical,
        physical.supported_features(),
        &device_ext,
        [(queue_family, 0.5)].iter().cloned(),
    )
    .unwrap();

    let queue = queues.next().unwrap();

    let (mut swapchain, images) = {
        let caps = surface.capabilities(physical).unwrap();
        let usage = caps.supported_usage_flags;
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();
        let format = caps.supported_formats[0].0;
        let dimensions: [u32; 2] = surface.window().inner_size().into();
        vp.projection = perspective(
            dimensions[0] as f32 / dimensions[1] as f32,
            180.0,
            0.01,
            100.0,
        );

        Swapchain::start(device.clone(), surface.clone())
            .num_images(caps.min_image_count)
            .format(format)
            .dimensions(dimensions)
            .usage(usage)
            .sharing_mode(&queue)
            .composite_alpha(alpha)
            .build()
            .unwrap()
    };

    let deferred_vert = deferred_vert::load(device.clone()).unwrap();
    let deferred_frag = deferred_frag::load(device.clone()).unwrap();
    let directional_vert = directional_vert::load(device.clone()).unwrap();
    let directional_frag = directional_frag::load(device.clone()).unwrap();
    let ambient_vert = ambient_vert::load(device.clone()).unwrap();
    let ambient_frag = ambient_frag::load(device.clone()).unwrap();

    let vp_buffer = CpuAccessibleBuffer::from_data(
        device.clone(),
        BufferUsage::all(),
        false,
        deferred_vert::ty::VP_Data {
            view: vp.view.into(),
            projection: vp.projection.into(),
        },
    )
    .unwrap();

    let model_uniform_buffer =
        CpuBufferPool::<deferred_vert::ty::Model_Data>::uniform_buffer(device.clone());
    let ambient_buffer =
        CpuBufferPool::<ambient_frag::ty::Ambient_Data>::uniform_buffer(device.clone());
    let directional_buffer =
        CpuBufferPool::<directional_frag::ty::Directional_Light_Data>::uniform_buffer(
            device.clone(),
        );

    let render_pass = vulkano::ordered_passes_renderpass!(device.clone(),
        attachments: {
            final_color: {
                load: Clear,
                store: Store,
                format: swapchain.format(),
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
        .vertex_input_state(BuffersDefinition::new().vertex::<NormalVertex>())
        .vertex_shader(deferred_vert.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(deferred_frag.entry_point("main").unwrap(), ())
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
        .render_pass(deferred_pass.clone())
        .build(device.clone())
        .unwrap();

    let directional_pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<DummyVertex>())
        .vertex_shader(directional_vert.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(directional_frag.entry_point("main").unwrap(), ())
        .color_blend_state(
            ColorBlendState::new(lighting_pass.num_color_attachments()).blend(AttachmentBlend {
                color_op: BlendOp::Add,
                color_source: BlendFactor::One,
                color_destination: BlendFactor::One,
                alpha_op: BlendOp::Max,
                alpha_source: BlendFactor::One,
                alpha_destination: BlendFactor::One,
            }),
        )
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
        .render_pass(lighting_pass.clone())
        .build(device.clone())
        .unwrap();

    let ambient_pipeline = GraphicsPipeline::start()
        .vertex_input_state(BuffersDefinition::new().vertex::<DummyVertex>())
        .vertex_shader(ambient_vert.entry_point("main").unwrap(), ())
        .input_assembly_state(InputAssemblyState::new())
        .viewport_state(ViewportState::viewport_dynamic_scissor_irrelevant())
        .fragment_shader(ambient_frag.entry_point("main").unwrap(), ())
        .color_blend_state(
            ColorBlendState::new(lighting_pass.num_color_attachments()).blend(AttachmentBlend {
                color_op: BlendOp::Add,
                color_source: BlendFactor::One,
                color_destination: BlendFactor::One,
                alpha_op: BlendOp::Max,
                alpha_source: BlendFactor::One,
                alpha_destination: BlendFactor::One,
            }),
        )
        .depth_stencil_state(DepthStencilState::simple_depth_test())
        .rasterization_state(RasterizationState::new().cull_mode(CullMode::Back))
        .render_pass(lighting_pass.clone())
        .build(device.clone())
        .unwrap();

    let vertex_buffer = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        model.data().iter().cloned(),
    )
    .unwrap();

    let dummy_verts = CpuAccessibleBuffer::from_iter(
        device.clone(),
        BufferUsage::all(),
        false,
        DummyVertex::list().iter().cloned(),
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

    let deferred_layout = deferred_pipeline
        .layout()
        .descriptor_set_layouts()
        .get(0)
        .unwrap();
    let mut vp_set_builder = PersistentDescriptorSet::start(deferred_layout.clone());
    vp_set_builder.add_buffer(vp_buffer.clone()).unwrap();
    let mut vp_set = vp_set_builder.build().unwrap();

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
                let (new_swapchain, new_images) =
                    match swapchain.recreate().dimensions(dimensions).build() {
                        Ok(r) => r,
                        Err(SwapchainCreationError::UnsupportedDimensions) => return,
                        Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
                    };
                vp.projection = perspective(
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

                let new_vp_buffer = CpuAccessibleBuffer::from_data(
                    device.clone(),
                    BufferUsage::all(),
                    false,
                    deferred_vert::ty::VP_Data {
                        view: vp.view.into(),
                        projection: vp.projection.into(),
                    },
                )
                .unwrap();

                let deferred_layout = deferred_pipeline
                    .layout()
                    .descriptor_set_layouts()
                    .get(0)
                    .unwrap();
                let mut vp_set_builder = PersistentDescriptorSet::start(deferred_layout.clone());
                vp_set_builder.add_buffer(new_vp_buffer.clone()).unwrap();
                vp_set = vp_set_builder.build().unwrap();

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

            let model_uniform_subbuffer = {
                let elapsed = rotation_start.elapsed().as_secs() as f64
                    + rotation_start.elapsed().subsec_nanos() as f64 / 1_000_000_000.0;
                let elapsed_as_radians = elapsed * pi::<f64>() / 180.0;
                model.zero_rotation();
                model.rotate(elapsed_as_radians as f32 * 50.0, vec3(0.0, 0.0, 1.0));
                model.rotate(elapsed_as_radians as f32 * 30.0, vec3(0.0, 1.0, 0.0));
                model.rotate(elapsed_as_radians as f32 * 20.0, vec3(1.0, 0.0, 0.0));

                let (model_mat, normal_mat) = model.model_matrices();

                let uniform_data = deferred_vert::ty::Model_Data {
                    model: model_mat.into(),
                    normals: normal_mat.into(),
                };

                model_uniform_buffer.next(uniform_data).unwrap()
            };

            let ambient_uniform_subbuffer = {
                let uniform_data = ambient_frag::ty::Ambient_Data {
                    color: ambient_light.color.into(),
                    intensity: ambient_light.intensity.into(),
                };

                ambient_buffer.next(uniform_data).unwrap()
            };

            let deferred_layout_model = deferred_pipeline
                .layout()
                .descriptor_set_layouts()
                .get(1)
                .unwrap();
            let mut model_set_builder =
                PersistentDescriptorSet::start(deferred_layout_model.clone());
            model_set_builder
                .add_buffer(model_uniform_subbuffer.clone())
                .unwrap();
            let model_set = model_set_builder.build().unwrap();

            let ambient_layout = ambient_pipeline
                .layout()
                .descriptor_set_layouts()
                .get(0)
                .unwrap();
            let mut ambient_set_builder = PersistentDescriptorSet::start(ambient_layout.clone());
            ambient_set_builder
                .add_image(color_buffer.clone())
                .unwrap()
                .add_image(normal_buffer.clone())
                .unwrap()
                .add_buffer(ambient_uniform_subbuffer.clone())
                .unwrap();
            let ambient_set = ambient_set_builder.build().unwrap();

            let directional_uniform_subbuffer =
                generate_directional_buffer(&directional_buffer, &directional_light);
            let directional_layout = directional_pipeline
                .layout()
                .descriptor_set_layouts()
                .get(0)
                .unwrap();
            let mut directional_set_builder =
                PersistentDescriptorSet::start(directional_layout.clone());
            directional_set_builder
                .add_image(color_buffer.clone())
                .unwrap()
                .add_image(normal_buffer.clone())
                .unwrap()
                .add_buffer(directional_uniform_subbuffer.clone())
                .unwrap();
            let directional_set = directional_set_builder.build().unwrap();

            let mut commands = AutoCommandBufferBuilder::primary(
                device.clone(),
                queue.family(),
                CommandBufferUsage::OneTimeSubmit,
            )
            .unwrap();
            commands
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
                    (vp_set.clone(), model_set.clone()),
                )
                .bind_vertex_buffers(0, vertex_buffer.clone())
                .draw(vertex_buffer.len() as u32, 1, 0, 0)
                .unwrap()
                .next_subpass(SubpassContents::Inline)
                .unwrap()
                .bind_pipeline_graphics(directional_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    directional_pipeline.layout().clone(),
                    0,
                    directional_set.clone(),
                )
                .bind_vertex_buffers(0, dummy_verts.clone())
                .draw(dummy_verts.len() as u32, 1, 0, 0)
                .unwrap()
                .bind_pipeline_graphics(ambient_pipeline.clone())
                .bind_descriptor_sets(
                    PipelineBindPoint::Graphics,
                    ambient_pipeline.layout().clone(),
                    0,
                    ambient_set.clone(),
                )
                .draw(dummy_verts.len() as u32, 1, 0, 0)
                .unwrap()
                .end_render_pass()
                .unwrap();
            let command_buffer = commands.build().unwrap();

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

fn generate_directional_buffer(
    pool: &vulkano::buffer::cpu_pool::CpuBufferPool<directional_frag::ty::Directional_Light_Data>,
    light: &DirectionalLight,
) -> Arc<
    vulkano::buffer::cpu_pool::CpuBufferPoolSubbuffer<
        directional_frag::ty::Directional_Light_Data,
        Arc<StdMemoryPool>,
    >,
> {
    let uniform_data = directional_frag::ty::Directional_Light_Data {
        position: light.position.into(),
        color: light.color.into(),
    };

    pool.next(uniform_data).unwrap()
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

    let color_buffer = ImageView::new(
        AttachmentImage::transient_input_attachment(
            device.clone(),
            dimensions,
            Format::A2B10G10R10_UNORM_PACK32,
        )
        .unwrap(),
    )
    .unwrap();
    let normal_buffer = ImageView::new(
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
                let view = ImageView::new(image.clone()).unwrap();
                let depth_buffer = ImageView::new(
                    AttachmentImage::transient(device.clone(), dimensions, Format::D16_UNORM)
                        .unwrap(),
                )
                .unwrap();
                Framebuffer::start(render_pass.clone())
                    .add(view)
                    .unwrap()
                    .add(color_buffer.clone())
                    .unwrap()
                    .add(normal_buffer.clone())
                    .unwrap()
                    .add(depth_buffer.clone())
                    .unwrap()
                    .build()
                    .unwrap()
            })
            .collect::<Vec<_>>(),
        color_buffer.clone(),
        normal_buffer.clone(),
    )
}
