use crate::obj_loader::{DummyVertex, NormalVertex};
use crate::model::Model;
use crate::system::DirectionalLight;
use super::system_errors::{SystemError, SystemErrorType};

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool, BufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::DescriptorSet;
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions, Queue};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::{AttachmentImage, SwapchainImage};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::memory::pool::StdMemoryPool;
use vulkano::pipeline::blend::{BlendOp, AttachmentBlend, BlendFactor};
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain::{AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError, Surface, SwapchainAcquireFuture};
use vulkano::swapchain;
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::sync;

use vulkano_win::VkSurfaceBuild;

use winit::{EventsLoop, Window, WindowBuilder};

use nalgebra_glm::{identity, perspective, TMat4};

use std::sync::Arc;
use std::mem;

vulkano::impl_vertex!(DummyVertex, position);
vulkano::impl_vertex!(NormalVertex, position, normal, color);

mod deferred_vert {
    vulkano_shaders::shader!{
        ty: "vertex",
        path: "src/system/shaders/deferred.vert"
    }
}

mod deferred_frag {
    vulkano_shaders::shader!{
        ty: "fragment",
        path: "src/system/shaders/deferred.frag"
    }
}

mod directional_vert {
    vulkano_shaders::shader!{
        ty: "vertex",
        path: "src/system/shaders/directional.vert"
    }
}

mod directional_frag {
    vulkano_shaders::shader!{
        ty: "fragment",
        path: "src/system/shaders/directional.frag"
    }
}

mod ambient_vert {
    vulkano_shaders::shader!{
        ty: "vertex",
        path: "src/system/shaders/ambient.vert"
    }
}

mod ambient_frag {
    vulkano_shaders::shader!{
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
    vp_buffer:Arc<CpuAccessibleBuffer<deferred_vert::ty::VP_Data>>,
    model_uniform_buffer:CpuBufferPool<deferred_vert::ty::Model_Data>,
    ambient_buffer:Arc<CpuAccessibleBuffer<ambient_frag::ty::Ambient_Data>>,
    directional_buffer:CpuBufferPool<directional_frag::ty::Directional_Light_Data>,
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    deferred_pipeline: Arc<dyn GraphicsPipelineAbstract+ Send + Sync>,
    directional_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    ambient_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    dummy_verts: Arc<CpuAccessibleBuffer<[DummyVertex]>>,
    dynamic_state: DynamicState,
    framebuffers: Vec<Arc<dyn FramebufferAbstract + Send + Sync>>,
    color_buffer: Arc<AttachmentImage>,
    normal_buffer: Arc<AttachmentImage>,
    vp_set: Arc<dyn DescriptorSet + Send + Sync>,
    render_stage: RenderStage,
    commands: Option<AutoCommandBufferBuilder>,
    img_index: usize,
    acquire_future: Option<SwapchainAcquireFuture<Window>>,
}

#[derive(Debug, Clone)]
struct VP {
    view: TMat4<f32>,
    projection: TMat4<f32>
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
    pub fn new(events_loop: &mut EventsLoop) -> Result<System, SystemError>  {
        let instance = {
            let extensions = vulkano_win::required_extensions();
            Instance::new(None, &extensions, None).unwrap()
        };

        let physical = PhysicalDevice::enumerate(&instance).next().unwrap();

        let surface = WindowBuilder::new().build_vk_surface(&events_loop, instance.clone()).unwrap();
        let window = surface.window();

        let queue_family = physical.queue_families().find(|&q| {
            q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
        }).unwrap();

        let device_ext = DeviceExtensions { khr_swapchain: true, .. DeviceExtensions::none() };
        let (device, mut queues) = Device::new(physical, physical.supported_features(), &device_ext,
                                               [(queue_family, 0.5)].iter().cloned()).unwrap();

        let queue = queues.next().unwrap();

        let mut vp = VP::new();

        let (swapchain, images) = {
            let caps = surface.capabilities(physical).unwrap();
            let usage = caps.supported_usage_flags;
            let alpha = caps.supported_composite_alpha.iter().next().unwrap();
            let format = caps.supported_formats[0].0;

            let initial_dimensions = if let Some(dimensions) = window.get_inner_size() {
                let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
                vp.projection = perspective(dimensions.0 as f32 / dimensions.1 as f32, 180.0, 0.01, 100.0);
                [dimensions.0, dimensions.1]
            } else {
                return Err(SystemError::new(SystemErrorType::ResizeFailure));
            };

            Swapchain::new(device.clone(), surface.clone(), caps.min_image_count, format,
                           initial_dimensions, 1, usage, &queue, SurfaceTransform::Identity, alpha,
                           PresentMode::Fifo, true, None).unwrap()
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
            deferred_vert::ty::VP_Data {
                view: vp.view.into(),
                projection: vp.projection.into(),
            }
        ).unwrap();

        let ambient_buffer = CpuAccessibleBuffer::from_data(
            device.clone(),
            BufferUsage::all(),
            ambient_frag::ty::Ambient_Data {
                color: [1.0, 1.0, 1.0],
                intensity: 0.1
            }
        ).unwrap();

        let model_uniform_buffer = CpuBufferPool::<deferred_vert::ty::Model_Data>::uniform_buffer(device.clone());
        let directional_buffer = CpuBufferPool::<directional_frag::ty::Directional_Light_Data>::uniform_buffer(device.clone());

        let render_pass = Arc::new(vulkano::ordered_passes_renderpass!(device.clone(),
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
                    format: Format::A2B10G10R10UnormPack32,
                    samples: 1,
                },
                normals: {
                    load: Clear,
                    store: DontCare,
                    format: Format::R16G16B16A16Sfloat,
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: Format::D16Unorm,
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
                    input: [color, normals, depth]
                }
            ]
        ).unwrap());

        let deferred_pass = Subpass::from(render_pass.clone(), 0).unwrap();
        let lighting_pass = Subpass::from(render_pass.clone(), 1).unwrap();

        let deferred_pipeline = Arc::new(GraphicsPipeline::start()
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
            .unwrap());

        let directional_pipeline = Arc::new(GraphicsPipeline::start()
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
            .unwrap());

        let ambient_pipeline = Arc::new(GraphicsPipeline::start()
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
            .unwrap());


        let dummy_verts = CpuAccessibleBuffer::from_iter(
            device.clone(),
            BufferUsage::all(),
            DummyVertex::list().iter().cloned()
        ).unwrap();

        let mut dynamic_state = DynamicState { line_width: None, viewports: None, scissors: None, compare_mask: None, write_mask: None, reference: None };

        let (framebuffers, color_buffer, normal_buffer) = System::window_size_dependent_setup(device.clone(), &images, render_pass.clone(), &mut dynamic_state);

        let vp_set = Arc::new(PersistentDescriptorSet::start(deferred_pipeline.clone(), 0)
            .add_buffer(vp_buffer.clone()).unwrap()
            .build().unwrap());

        let render_stage = RenderStage::Stopped;

        let commands = None;
        let img_index = 0;
        let acquire_future = None;

        Ok(System{
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
            dynamic_state,
            framebuffers,
            color_buffer,
            normal_buffer,
            vp_set,
            render_stage,
            commands,
            img_index,
            acquire_future,
        })
    }

    pub fn ambient(&mut self) {
        match self.render_stage {
            RenderStage::Deferred => {
                self.render_stage = RenderStage::Ambient;
            },
            RenderStage::Ambient => {
                return;
            }
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            },
            _ => {
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
        }

        let ambient_set = Arc::new(PersistentDescriptorSet::start(self.ambient_pipeline.clone(), 0)
            .add_image(self.color_buffer.clone()).unwrap()
            .add_image(self.normal_buffer.clone()).unwrap()
            .add_buffer(self.ambient_buffer.clone()).unwrap()
            .build().unwrap());

        self.commands = Some(self.commands.take()
            .unwrap()
            .next_subpass(false)
            .unwrap()
            .draw(self.ambient_pipeline.clone(),
                  &self.dynamic_state,
                  vec![self.dummy_verts.clone()],
                  ambient_set.clone(),
                  ())
            .unwrap());
    }

    pub fn directional(&mut self, directional_light: &DirectionalLight) {
        match self.render_stage {
            RenderStage::Ambient => {
                self.render_stage = RenderStage::Directional;
            },
            RenderStage::Directional => {
            }
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            },
            _ => {
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
        }

        let directional_uniform_subbuffer = self.generate_directional_buffer(&self.directional_buffer, &directional_light);
        let directional_set = Arc::new(
            PersistentDescriptorSet::start(self.directional_pipeline.clone(), 0)
            .add_image(self.color_buffer.clone()).unwrap()
            .add_image(self.normal_buffer.clone()).unwrap()
            .add_buffer(directional_uniform_subbuffer.clone()).unwrap()
            .build().unwrap());
        self.commands = Some(self.commands.take().unwrap()
            .draw(
                self.directional_pipeline.clone(),
                &self.dynamic_state,
                vec![self.dummy_verts.clone()],
                directional_set.clone(),
                ())
            .unwrap());
    }

    pub fn finish(&mut self, previous_frame_end: &mut Box<dyn GpuFuture>) {
        match self.render_stage {
            RenderStage::Directional => {
            },
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            },
            _ => {
                self.commands = None;
                self.render_stage = RenderStage::Stopped;
                return;
            }
        }

        let command_buffer = self.commands.take()
            .unwrap()
            .end_render_pass()
            .unwrap()
            .build()
            .unwrap();

        let af = self.acquire_future.take().unwrap();

        let mut local_future:Box<dyn GpuFuture> = Box::new(sync::now(self.device.clone()));

        mem::swap(&mut local_future, previous_frame_end);

        let future = local_future.join(af)
            .then_execute(self.queue.clone(), command_buffer).unwrap()
            .then_swapchain_present(self.queue.clone(), self.swapchain.clone(), self.img_index)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                // This wait is required when using NVIDIA or running on macOS. See https://github.com/vulkano-rs/vulkano/issues/1247
                future.wait(None).unwrap();
                *previous_frame_end = Box::new(future);
            }
            Err(FlushError::OutOfDate) => {
                self.recreate_swapchain();
                *previous_frame_end = Box::new(sync::now(self.device.clone()));
            }
            Err(e) => {
                println!("{:?}", e);
                *previous_frame_end = Box::new(sync::now(self.device.clone()));
            }
        }

        self.commands = None;
        self.render_stage = RenderStage::Stopped;
    }

    fn generate_directional_buffer(
        &self,
        pool: &vulkano::buffer::cpu_pool::CpuBufferPool<directional_frag::ty::Directional_Light_Data>,
        light: &DirectionalLight
    ) -> vulkano::buffer::cpu_pool::CpuBufferPoolSubbuffer<directional_frag::ty::Directional_Light_Data, Arc<StdMemoryPool>> {
        let uniform_data = directional_frag::ty::Directional_Light_Data {
            position: light.position.into(),
            color: light.color.into()
        };

        pool.next(uniform_data).unwrap()
    }

    pub fn geometry(&mut self, model: &mut Model) {
        match self.render_stage {
            RenderStage::Deferred => {
            },
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            },
            _ => {
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
        }

        let model_uniform_subbuffer = {
            let (model_mat, normal_mat) = model.model_matrices();

            let uniform_data = deferred_vert::ty::Model_Data {
                model: model_mat.into(),
                normals: normal_mat.into(),
            };

            self.model_uniform_buffer.next(uniform_data).unwrap()
        };

        let model_set:Arc<dyn DescriptorSet + Send + Sync> = Arc::new(
            PersistentDescriptorSet::start(self.deferred_pipeline.clone(), 1)
                .add_buffer(model_uniform_subbuffer.clone()).unwrap()
                .build().unwrap());

        let vertex_buffer:Arc<dyn BufferAccess + Send + Sync> = CpuAccessibleBuffer::from_iter(
            self.device.clone(),
            BufferUsage::all(),
            model.data().iter().cloned()).unwrap();

        self.commands = Some(self.commands.take().unwrap()
            .draw(self.deferred_pipeline.clone(),
                  &self.dynamic_state,
                  vec![vertex_buffer.clone()],
                  vec![self.vp_set.clone(), model_set.clone()],
                  ()
            )
            .unwrap());
    }

    #[allow(dead_code)]
    pub fn set_ambient(&mut self, color: [f32;3], intensity: f32) {
        self.ambient_buffer = CpuAccessibleBuffer::from_data(
            self.device.clone(),
            BufferUsage::all(),
            ambient_frag::ty::Ambient_Data {
                color,
                intensity
            }
        ).unwrap();
    }

    pub fn set_view(&mut self, view: &TMat4<f32>) {
        self.vp.view = view.clone();
        self.vp_buffer = CpuAccessibleBuffer::from_data(
            self.device.clone(),
            BufferUsage::all(),
            deferred_vert::ty::VP_Data {
                view: self.vp.view.into(),
                projection: self.vp.projection.into(),
            }
        ).unwrap();

        self.vp_set = Arc::new(PersistentDescriptorSet::start(self.deferred_pipeline.clone(), 0)
            .add_buffer(self.vp_buffer.clone()).unwrap()
            .build().unwrap());
        self.render_stage = RenderStage::Stopped;
    }

    pub fn start(&mut self) {
        match self.render_stage {
            RenderStage::Stopped => {
                self.render_stage = RenderStage::Deferred;
            },
            RenderStage::NeedsRedraw => {
                self.recreate_swapchain();
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            },
            _ => {
                self.render_stage = RenderStage::Stopped;
                self.commands = None;
                return;
            }
        }

        let (img_index, acquire_future) = match swapchain::acquire_next_image(self.swapchain.clone(), None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                self.recreate_swapchain();
                return;
            },
            Err(err) => panic!("{:?}", err)
        };

        let clear_values = vec![[0.0, 0.0, 0.0, 1.0].into(), [0.0, 0.0, 0.0, 1.0].into(), [0.0, 0.0, 0.0, 1.0].into(), 1f32.into()];

        self.commands = Some(AutoCommandBufferBuilder::primary_one_time_submit(self.device.clone(), self.queue.family()).unwrap()
            .begin_render_pass(self.framebuffers[img_index].clone(), false, clear_values)
            .unwrap());

        self.img_index = img_index;

        self.acquire_future = Some(acquire_future);
    }

    pub fn recreate_swapchain(&mut self) {
        self.render_stage = RenderStage::NeedsRedraw;
        self.commands = None;

        let window = self.surface.window();

        let dimensions = if let Some(dimensions) = window.get_inner_size() {
            let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
            self.vp.projection = perspective(dimensions.0 as f32 / dimensions.1 as f32, 180.0, 0.01, 100.0);
            [dimensions.0, dimensions.1]
        } else {
            return;
        };

        let (new_swapchain, new_images) = match self.swapchain.recreate_with_dimension(dimensions) {
            Ok(r) => r,
            Err(SwapchainCreationError::UnsupportedDimensions) => return,
            Err(err) => panic!("{:?}", err)
        };

        self.swapchain = new_swapchain;
        let (new_framebuffers, new_color_buffer, new_normal_buffer) = System::window_size_dependent_setup(self.device.clone(), &new_images, self.render_pass.clone(), &mut self.dynamic_state);
        self.framebuffers = new_framebuffers;
        self.color_buffer = new_color_buffer;
        self.normal_buffer = new_normal_buffer;

        self.vp_buffer = CpuAccessibleBuffer::from_data(
            self.device.clone(),
            BufferUsage::all(),
            deferred_vert::ty::VP_Data {
                view: self.vp.view.into(),
                projection: self.vp.projection.into(),
            }
        ).unwrap();

        self.vp_set = Arc::new(PersistentDescriptorSet::start(self.deferred_pipeline.clone(), 0)
            .add_buffer(self.vp_buffer.clone()).unwrap()
            .build().unwrap());

        self.render_stage = RenderStage::Stopped;
    }

    fn window_size_dependent_setup(
        device: Arc<Device>,
        images: &[Arc<SwapchainImage<Window>>],
        render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
        dynamic_state: &mut DynamicState
    ) -> (Vec<Arc<dyn FramebufferAbstract + Send + Sync>>, Arc<AttachmentImage>, Arc<AttachmentImage>) {
        let dimensions = images[0].dimensions();

        let viewport = Viewport {
            origin: [0.0, 0.0],
            dimensions: [dimensions[0] as f32, dimensions[1] as f32],
            depth_range: 0.0 .. 1.0,
        };
        dynamic_state.viewports = Some(vec!(viewport));

        let color_buffer = AttachmentImage::transient_input_attachment(device.clone(), dimensions, Format::A2B10G10R10UnormPack32).unwrap();
        let normal_buffer = AttachmentImage::transient_input_attachment(device.clone(), dimensions, Format::R16G16B16A16Sfloat).unwrap();
        let depth_buffer = AttachmentImage::transient_input_attachment(device.clone(), dimensions, Format::D16Unorm).unwrap();

        (images.iter().map(|image| {
            Arc::new(
                Framebuffer::start(render_pass.clone())
                    .add(image.clone()).unwrap()
                    .add(color_buffer.clone()).unwrap()
                    .add(normal_buffer.clone()).unwrap()
                    .add(depth_buffer.clone()).unwrap()
                    .build().unwrap()
            ) as Arc<dyn FramebufferAbstract + Send + Sync>
        }).collect::<Vec<_>>(), color_buffer.clone(), normal_buffer.clone())
    }
}