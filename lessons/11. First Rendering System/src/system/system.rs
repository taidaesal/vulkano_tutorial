// Copyright (c) 2021 taidaesal
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>
//
// This file contains code copied and/or adapted
// from code provided by the Vulkano project under
// the MIT license

use crate::model::Model;
use crate::obj_loader::{DummyVertex, NormalVertex};
use crate::system::DirectionalLight;

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer};
use vulkano::buffer::{CpuBufferPool, TypedBufferAccess};
use vulkano::command_buffer::pool::standard::{
    StandardCommandPoolAlloc, StandardCommandPoolBuilder,
};
use vulkano::command_buffer::{
    AutoCommandBufferBuilder, CommandBufferUsage, PrimaryAutoCommandBuffer, SubpassContents,
};
use vulkano::descriptor_set::{DescriptorSet, PersistentDescriptorSet};
use vulkano::device::physical::PhysicalDevice;
use vulkano::device::{Device, DeviceExtensions, Queue};
use vulkano::format::Format;
use vulkano::image::view::ImageView;
use vulkano::image::{AttachmentImage, SwapchainImage};
use vulkano::instance::Instance;
use vulkano::memory::pool::StdMemoryPool;
use vulkano::pipeline::blend::{AttachmentBlend, BlendFactor, BlendOp};
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, PipelineBindPoint};
use vulkano::render_pass::{Framebuffer, FramebufferAbstract, RenderPass, Subpass};
use vulkano::swapchain::{self, Surface, SwapchainAcquireFuture};
use vulkano::swapchain::{AcquireError, Swapchain, SwapchainCreationError};
use vulkano::sync;
use vulkano::sync::{FlushError, GpuFuture};
use vulkano::Version;

use vulkano_win::VkSurfaceBuild;

use winit::event_loop::EventLoop;
use winit::window::{Window, WindowBuilder};

use nalgebra_glm::{identity, perspective, TMat4};

use std::mem;
use std::sync::Arc;

vulkano::impl_vertex!(DummyVertex, position);
vulkano::impl_vertex!(NormalVertex, position, normal, color);

mod deferred_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/system/shaders/deferred.vert"
    }
}

mod deferred_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/system/shaders/deferred.frag"
    }
}

mod directional_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/system/shaders/directional.vert"
    }
}

mod directional_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/system/shaders/directional.frag"
    }
}

mod ambient_vert {
    vulkano_shaders::shader! {
        ty: "vertex",
        path: "src/system/shaders/ambient.vert"
    }
}

mod ambient_frag {
    vulkano_shaders::shader! {
        ty: "fragment",
        path: "src/system/shaders/ambient.frag"
    }
}

#[derive(Debug, Clone)]
enum RenderStage {
    Stopped,
    Deferred,
    Ambient,
    Directional,
    NeedsRedraw,
}

pub struct System {
    surface: Arc<Surface<Window>>,
    pub device: Arc<Device>,
    queue: Arc<Queue>,
    vp: VP,
    swapchain: Arc<Swapchain<Window>>,
    vp_buffer: Arc<CpuAccessibleBuffer<deferred_vert::ty::VP_Data>>,
    model_uniform_buffer: CpuBufferPool<deferred_vert::ty::Model_Data>,
    ambient_buffer: Arc<CpuAccessibleBuffer<ambient_frag::ty::Ambient_Data>>,
    directional_buffer: CpuBufferPool<directional_frag::ty::Directional_Light_Data>,
    render_pass: Arc<RenderPass>,
    deferred_pipeline: Arc<GraphicsPipeline>,
    directional_pipeline: Arc<GraphicsPipeline>,
    ambient_pipeline: Arc<GraphicsPipeline>,
    dummy_verts: Arc<CpuAccessibleBuffer<[DummyVertex]>>,
    framebuffers: Vec<Arc<dyn FramebufferAbstract>>,
    color_buffer: Arc<ImageView<Arc<AttachmentImage>>>,
    normal_buffer: Arc<ImageView<Arc<AttachmentImage>>>,
    vp_set: Arc<dyn DescriptorSet + Send + Sync>,
    render_stage: RenderStage,
    commands: Option<
        AutoCommandBufferBuilder<
            PrimaryAutoCommandBuffer<StandardCommandPoolAlloc>,
            StandardCommandPoolBuilder,
        >,
    >,
    img_index: usize,
    acquire_future: Option<SwapchainAcquireFuture<Window>>,
    viewport: Viewport,
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

impl System {
    pub fn new(event_loop: &EventLoop<()>) -> System {
        let instance = {
            let extensions = vulkano_win::required_extensions();
            Instance::new(None, Version::V1_1, &extensions, None).unwrap()
        };

        let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

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

        let mut vp = VP::new();

        let (swapchain, images) = {
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

        let deferred_vert = deferred_vert::Shader::load(device.clone()).unwrap();
        let deferred_frag = deferred_frag::Shader::load(device.clone()).unwrap();
        let directional_vert = directional_vert::Shader::load(device.clone()).unwrap();
        let directional_frag = directional_frag::Shader::load(device.clone()).unwrap();
        let ambient_vert = ambient_vert::Shader::load(device.clone()).unwrap();
        let ambient_frag = ambient_frag::Shader::load(device.clone()).unwrap();

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

        let ambient_buffer = CpuAccessibleBuffer::from_data(
            device.clone(),
            BufferUsage::all(),
            false,
            ambient_frag::ty::Ambient_Data {
                color: [1.0, 1.0, 1.0],
                intensity: 0.1,
            },
        )
        .unwrap();

        let model_uniform_buffer =
            CpuBufferPool::<deferred_vert::ty::Model_Data>::uniform_buffer(device.clone());
        let directional_buffer =
            CpuBufferPool::<directional_frag::ty::Directional_Light_Data>::uniform_buffer(
                device.clone(),
            );

        let render_pass = Arc::new(
            vulkano::ordered_passes_renderpass!(device.clone(),
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
            .unwrap(),
        );

        let deferred_pass = Subpass::from(render_pass.clone(), 0).unwrap();
        let lighting_pass = Subpass::from(render_pass.clone(), 1).unwrap();

        let deferred_pipeline = Arc::new(
            GraphicsPipeline::start()
                .vertex_input_single_buffer::<NormalVertex>()
                .vertex_shader(deferred_vert.main_entry_point(), ())
                .triangle_list()
                .viewports_dynamic_scissors_irrelevant(1)
                .fragment_shader(deferred_frag.main_entry_point(), ())
                .depth_stencil_simple_depth()
                .front_face_counter_clockwise()
                .cull_mode_back()
                .render_pass(deferred_pass.clone())
                .build(device.clone())
                .unwrap(),
        );

        let directional_pipeline = Arc::new(
            GraphicsPipeline::start()
                .vertex_input_single_buffer::<DummyVertex>()
                .vertex_shader(directional_vert.main_entry_point(), ())
                .triangle_list()
                .viewports_dynamic_scissors_irrelevant(1)
                .fragment_shader(directional_frag.main_entry_point(), ())
                .blend_collective(AttachmentBlend {
                    enabled: true,
                    color_op: BlendOp::Add,
                    color_source: BlendFactor::One,
                    color_destination: BlendFactor::One,
                    alpha_op: BlendOp::Max,
                    alpha_source: BlendFactor::One,
                    alpha_destination: BlendFactor::One,
                    mask_red: true,
                    mask_green: true,
                    mask_blue: true,
                    mask_alpha: true,
                })
                .front_face_counter_clockwise()
                .cull_mode_back()
                .render_pass(lighting_pass.clone())
                .build(device.clone())
                .unwrap(),
        );

        let ambient_pipeline = Arc::new(
            GraphicsPipeline::start()
                .vertex_input_single_buffer::<DummyVertex>()
                .vertex_shader(ambient_vert.main_entry_point(), ())
                .triangle_list()
                .viewports_dynamic_scissors_irrelevant(1)
                .fragment_shader(ambient_frag.main_entry_point(), ())
                .blend_collective(AttachmentBlend {
                    enabled: true,
                    color_op: BlendOp::Add,
                    color_source: BlendFactor::One,
                    color_destination: BlendFactor::One,
                    alpha_op: BlendOp::Max,
                    alpha_source: BlendFactor::One,
                    alpha_destination: BlendFactor::One,
                    mask_red: true,
                    mask_green: true,
                    mask_blue: true,
                    mask_alpha: true,
                })
                .front_face_counter_clockwise()
                .cull_mode_back()
                .render_pass(lighting_pass.clone())
                .build(device.clone())
                .unwrap(),
        );

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

        let (framebuffers, color_buffer, normal_buffer) = System::window_size_dependent_setup(
            device.clone(),
            &images,
            render_pass.clone(),
            &mut viewport,
        );

        let vp_layout = deferred_pipeline
            .layout()
            .descriptor_set_layouts()
            .get(0)
            .unwrap();
        let mut vp_set_builder = PersistentDescriptorSet::start(vp_layout.clone());
        vp_set_builder.add_buffer(vp_buffer.clone()).unwrap();
        let vp_set = Arc::new(vp_set_builder.build().unwrap());

        let render_stage = RenderStage::Stopped;

        let commands = None;
        let img_index = 0;
        let acquire_future = None;

        System {
            surface,
            device,
            queue,
            vp,
            swapchain,
            vp_buffer,
            model_uniform_buffer,
            ambient_buffer,
            directional_buffer,
            render_pass,
            deferred_pipeline,
            directional_pipeline,
            ambient_pipeline,
            dummy_verts,
            framebuffers,
            color_buffer,
            normal_buffer,
            vp_set,
            render_stage,
            commands,
            img_index,
            acquire_future,
            viewport,
        }
    }

    pub fn ambient(&mut self) {
        match self.render_stage {
            RenderStage::Deferred => {
                self.render_stage = RenderStage::Ambient;
            }
            RenderStage::Ambient => {
                return;
            }
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
            _ => {
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
        }

        let ambient_layout = self
            .ambient_pipeline
            .layout()
            .descriptor_set_layouts()
            .get(0)
            .unwrap();
        let mut ambient_set_builder = PersistentDescriptorSet::start(ambient_layout.clone());
        ambient_set_builder
            .add_image(self.color_buffer.clone())
            .unwrap()
            .add_image(self.normal_buffer.clone())
            .unwrap()
            .add_buffer(self.ambient_buffer.clone())
            .unwrap();
        let ambient_set = Arc::new(ambient_set_builder.build().unwrap());

        let mut commands = self.commands.take().unwrap();
        commands
            .next_subpass(SubpassContents::Inline)
            .unwrap()
            .bind_pipeline_graphics(self.ambient_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.ambient_pipeline.layout().clone(),
                0,
                ambient_set.clone(),
            )
            .set_viewport(0, [self.viewport.clone()])
            .bind_vertex_buffers(0, self.dummy_verts.clone())
            .draw(self.dummy_verts.len() as u32, 1, 0, 0)
            .unwrap();
        self.commands = Some(commands);
    }

    pub fn directional(&mut self, directional_light: &DirectionalLight) {
        match self.render_stage {
            RenderStage::Ambient => {
                self.render_stage = RenderStage::Directional;
            }
            RenderStage::Directional => {}
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
            _ => {
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
        }

        let directional_uniform_subbuffer =
            self.generate_directional_buffer(&self.directional_buffer, &directional_light);

        let directional_layout = self
            .directional_pipeline
            .layout()
            .descriptor_set_layouts()
            .get(0)
            .unwrap();
        let mut directional_set_builder =
            PersistentDescriptorSet::start(directional_layout.clone());
        directional_set_builder
            .add_image(self.color_buffer.clone())
            .unwrap()
            .add_image(self.normal_buffer.clone())
            .unwrap()
            .add_buffer(directional_uniform_subbuffer.clone())
            .unwrap();
        let directional_set = Arc::new(directional_set_builder.build().unwrap());

        let mut commands = self.commands.take().unwrap();
        commands
            .set_viewport(0, [self.viewport.clone()])
            .bind_pipeline_graphics(self.directional_pipeline.clone())
            .bind_vertex_buffers(0, self.dummy_verts.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.directional_pipeline.layout().clone(),
                0,
                directional_set.clone(),
            )
            .draw(self.dummy_verts.len() as u32, 1, 0, 0)
            .unwrap();

        self.commands = Some(commands);
    }

    pub fn finish(&mut self, previous_frame_end: &mut Option<Box<dyn GpuFuture>>) {
        match self.render_stage {
            RenderStage::Directional => {}
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
            _ => {
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
        }

        let mut commands = self.commands.take().unwrap();
        commands.end_render_pass().unwrap();
        let command_buffer = commands.build().unwrap();

        let af = self.acquire_future.take().unwrap();

        let mut local_future: Option<Box<dyn GpuFuture>> =
            Some(Box::new(sync::now(self.device.clone())) as Box<dyn GpuFuture>);

        mem::swap(&mut local_future, previous_frame_end);

        let future = local_future
            .take()
            .unwrap()
            .join(af)
            .then_execute(self.queue.clone(), command_buffer)
            .unwrap()
            .then_swapchain_present(self.queue.clone(), self.swapchain.clone(), self.img_index)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                *previous_frame_end = Some(Box::new(future) as Box<_>);
            }
            Err(FlushError::OutOfDate) => {
                self.recreate_swapchain();
                *previous_frame_end = Some(Box::new(sync::now(self.device.clone())) as Box<_>);
            }
            Err(e) => {
                println!("Failed to flush future: {:?}", e);
                *previous_frame_end = Some(Box::new(sync::now(self.device.clone())) as Box<_>);
            }
        }

        self.commands = None;
        self.render_stage = RenderStage::Stopped;
    }

    fn generate_directional_buffer(
        &self,
        pool: &vulkano::buffer::cpu_pool::CpuBufferPool<
            directional_frag::ty::Directional_Light_Data,
        >,
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

        Arc::new(pool.next(uniform_data).unwrap())
    }

    pub fn geometry(&mut self, model: &mut Model) {
        match self.render_stage {
            RenderStage::Deferred => {}
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
            _ => {
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
        }

        let model_uniform_subbuffer = Arc::new({
            let (model_mat, normal_mat) = model.model_matrices();

            let uniform_data = deferred_vert::ty::Model_Data {
                model: model_mat.into(),
                normals: normal_mat.into(),
            };

            self.model_uniform_buffer.next(uniform_data).unwrap()
        });

        let deferred_layout_model = self
            .deferred_pipeline
            .layout()
            .descriptor_set_layouts()
            .get(1)
            .unwrap();
        let mut model_set_builder = PersistentDescriptorSet::start(deferred_layout_model.clone());
        model_set_builder
            .add_buffer(model_uniform_subbuffer.clone())
            .unwrap();
        let model_set = Arc::new(model_set_builder.build().unwrap());

        let vertex_buffer = CpuAccessibleBuffer::from_iter(
            self.device.clone(),
            BufferUsage::all(),
            false,
            model.data().iter().cloned(),
        )
        .unwrap();

        let mut commands = self.commands.take().unwrap();
        commands
            .set_viewport(0, [self.viewport.clone()])
            .bind_pipeline_graphics(self.deferred_pipeline.clone())
            .bind_descriptor_sets(
                PipelineBindPoint::Graphics,
                self.deferred_pipeline.layout().clone(),
                0,
                (self.vp_set.clone(), model_set.clone()),
            )
            .bind_vertex_buffers(0, vertex_buffer.clone())
            .draw(vertex_buffer.len() as u32, 1, 0, 0)
            .unwrap();
        self.commands = Some(commands);
    }

    #[allow(dead_code)]
    pub fn set_ambient(&mut self, color: [f32; 3], intensity: f32) {
        self.ambient_buffer = CpuAccessibleBuffer::from_data(
            self.device.clone(),
            BufferUsage::all(),
            false,
            ambient_frag::ty::Ambient_Data { color, intensity },
        )
        .unwrap();
    }

    pub fn set_view(&mut self, view: &TMat4<f32>) {
        self.vp.view = view.clone();
        self.vp_buffer = CpuAccessibleBuffer::from_data(
            self.device.clone(),
            BufferUsage::all(),
            false,
            deferred_vert::ty::VP_Data {
                view: self.vp.view.into(),
                projection: self.vp.projection.into(),
            },
        )
        .unwrap();

        let vp_layout = self
            .deferred_pipeline
            .layout()
            .descriptor_set_layouts()
            .get(0)
            .unwrap();
        let mut vp_set_builder = PersistentDescriptorSet::start(vp_layout.clone());
        vp_set_builder.add_buffer(self.vp_buffer.clone()).unwrap();
        self.vp_set = Arc::new(vp_set_builder.build().unwrap());

        self.render_stage = RenderStage::Stopped;
    }

    pub fn start(&mut self) {
        match self.render_stage {
            RenderStage::Stopped => {
                self.render_stage = RenderStage::Deferred;
            }
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
            _ => {
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
        }

        let (img_index, suboptimal, acquire_future) =
            match swapchain::acquire_next_image(self.swapchain.clone(), None) {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    self.recreate_swapchain();
                    return;
                }
                Err(err) => panic!("{:?}", err),
            };

        if suboptimal {
            self.recreate_swapchain();
            return;
        }

        let clear_values = vec![
            [0.0, 0.0, 0.0, 1.0].into(),
            [0.0, 0.0, 0.0, 1.0].into(),
            [0.0, 0.0, 0.0, 1.0].into(),
            1f32.into(),
        ];

        let mut commands = AutoCommandBufferBuilder::primary(
            self.device.clone(),
            self.queue.family(),
            CommandBufferUsage::OneTimeSubmit,
        )
        .unwrap();
        commands
            .begin_render_pass(
                self.framebuffers[img_index].clone(),
                SubpassContents::Inline,
                clear_values,
            )
            .unwrap();
        self.commands = Some(commands);

        self.img_index = img_index;

        self.acquire_future = Some(acquire_future);
    }

    pub fn recreate_swapchain(&mut self) {
        self.render_stage = RenderStage::NeedsRedraw;
        self.commands = None;

        let dimensions: [u32; 2] = self.surface.window().inner_size().into();
        let (new_swapchain, new_images) =
            match self.swapchain.recreate().dimensions(dimensions).build() {
                Ok(r) => r,
                Err(SwapchainCreationError::UnsupportedDimensions) => return,
                Err(e) => panic!("Failed to recreate swapchain: {:?}", e),
            };
        self.vp.projection = perspective(
            dimensions[0] as f32 / dimensions[1] as f32,
            180.0,
            0.01,
            100.0,
        );

        self.swapchain = new_swapchain;
        let (new_framebuffers, new_color_buffer, new_normal_buffer) =
            System::window_size_dependent_setup(
                self.device.clone(),
                &new_images,
                self.render_pass.clone(),
                &mut self.viewport,
            );
        self.framebuffers = new_framebuffers;
        self.color_buffer = new_color_buffer;
        self.normal_buffer = new_normal_buffer;

        self.vp_buffer = CpuAccessibleBuffer::from_data(
            self.device.clone(),
            BufferUsage::all(),
            false,
            deferred_vert::ty::VP_Data {
                view: self.vp.view.into(),
                projection: self.vp.projection.into(),
            },
        )
        .unwrap();

        let vp_layout = self
            .deferred_pipeline
            .layout()
            .descriptor_set_layouts()
            .get(0)
            .unwrap();
        let mut vp_set_builder = PersistentDescriptorSet::start(vp_layout.clone());
        vp_set_builder.add_buffer(self.vp_buffer.clone()).unwrap();
        self.vp_set = Arc::new(vp_set_builder.build().unwrap());

        self.render_stage = RenderStage::Stopped;
    }

    fn window_size_dependent_setup(
        device: Arc<Device>,
        images: &[Arc<SwapchainImage<Window>>],
        render_pass: Arc<RenderPass>,
        viewport: &mut Viewport,
    ) -> (
        Vec<Arc<dyn FramebufferAbstract>>,
        Arc<ImageView<Arc<AttachmentImage>>>,
        Arc<ImageView<Arc<AttachmentImage>>>,
    ) {
        let dimensions = images[0].dimensions();
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
                    Arc::new(
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
                            .unwrap(),
                    ) as Arc<dyn FramebufferAbstract>
                })
                .collect::<Vec<_>>(),
            color_buffer.clone(),
            normal_buffer.clone(),
        )
    }
}
