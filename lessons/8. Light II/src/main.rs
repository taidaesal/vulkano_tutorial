use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, CpuBufferPool};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::device::{Device, DeviceExtensions};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, RenderPassAbstract, Subpass};
use vulkano::image::{AttachmentImage, SwapchainImage};
use vulkano::instance::{Instance, PhysicalDevice};
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain::{AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError};
use vulkano::swapchain;
use vulkano::sync::{GpuFuture, FlushError};
use vulkano::sync;

use vulkano_win::VkSurfaceBuild;

use winit::{EventsLoop, Window, WindowBuilder, Event, WindowEvent};

use nalgebra_glm::{identity, look_at, perspective, pi, rotate_normalized_axis, TMat4, translate, vec3};

use std::sync::Arc;
use std::time::Instant;
use vulkano::pipeline::blend::{BlendOp, BlendFactor, AttachmentBlend};
use vulkano::memory::pool::StdMemoryPool;

#[derive(Default, Debug, Clone)]
struct Vertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3]
}
vulkano::impl_vertex!(Vertex, position, normal, color);

#[derive(Default, Debug, Clone)]
struct AmbientLight {
    color: [f32; 3],
    intensity: f32
}

#[derive(Default, Debug, Clone)]
struct DirectionalLight {
    position: [f32; 4],
    color: [f32; 3]
}

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

mod deferred_vert {
    vulkano_shaders::shader!{
        ty: "vertex",
        path: "src/shaders/deferred.vert"
    }
}

mod deferred_frag {
    vulkano_shaders::shader!{
        ty: "fragment",
        path: "src/shaders/deferred.frag"
    }
}

mod directional_vert {
    vulkano_shaders::shader!{
        ty: "vertex",
        path: "src/shaders/directional.vert"
    }
}

mod directional_frag {
    vulkano_shaders::shader!{
        ty: "fragment",
        path: "src/shaders/directional.frag"
    }
}

mod ambient_vert {
    vulkano_shaders::shader!{
        ty: "vertex",
        path: "src/shaders/ambient.vert"
    }
}

mod ambient_frag {
    vulkano_shaders::shader!{
        ty: "fragment",
        path: "src/shaders/ambient.frag"
    }
}

fn main() {
    let mut mvp = MVP::new();
    mvp.view = look_at(&vec3(0.0, 0.0, 0.01), &vec3(0.0, 0.0, 0.0), &vec3(0.0, -1.0, 0.0));
    mvp.model = translate(&identity(), &vec3(0.0, 0.0, -2.5));

    let ambient_light = AmbientLight { color: [1.0, 1.0, 1.0], intensity: 0.1 };
    let directional_light_r = DirectionalLight {position: [-4.0, 0.0, -2.0, 1.0], color: [1.0, 0.0, 0.0]};
    let directional_light_g = DirectionalLight {position: [0.0, -4.0, 1.0, 1.0], color: [0.0, 1.0, 0.0]};
    let directional_light_b = DirectionalLight {position: [4.0, -2.0, -1.0, 1.0], color: [0.0, 0.0, 1.0]};

    let instance = {
        let extensions = vulkano_win::required_extensions();
        Instance::new(None, &extensions, None).unwrap()
    };

    let physical = PhysicalDevice::enumerate(&instance).next().unwrap();
    println!("Using device: {} (type: {:?})", physical.name(), physical.ty());

    let mut events_loop = EventsLoop::new();
    let surface = WindowBuilder::new().build_vk_surface(&events_loop, instance.clone()).unwrap();
    let window = surface.window();

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

        let initial_dimensions = if let Some(dimensions) = window.get_inner_size() {
            let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
            mvp.projection = perspective(dimensions.0 as f32 / dimensions.1 as f32, 180.0, 0.01, 100.0);
            [dimensions.0, dimensions.1]
        } else {
            return;
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

    let uniform_buffer = CpuBufferPool::<deferred_vert::ty::MVP_Data>::uniform_buffer(device.clone());
    let ambient_buffer = CpuBufferPool::<ambient_frag::ty::Ambient_Data>::uniform_buffer(device.clone());
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
        .vertex_input_single_buffer()
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
        .vertex_input_single_buffer::<Vertex>()
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
        .vertex_input_single_buffer::<Vertex>()
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

    let vertex_buffer = CpuAccessibleBuffer::from_iter(device.clone(), BufferUsage::all(), [
        // front face
        Vertex { position: [-1.000000, -1.000000, 1.000000], normal: [0.0000, 0.0000, 1.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, 1.000000, 1.000000], normal: [0.0000, 0.0000, 1.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, 1.000000, 1.000000], normal: [0.0000, 0.0000, 1.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, -1.000000, 1.000000], normal: [0.0000, 0.0000, 1.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, 1.000000, 1.000000], normal: [0.0000, 0.0000, 1.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, -1.000000, 1.000000], normal: [0.0000, 0.0000, 1.0000], color: [1.0, 0.35, 0.137]},

        // back face
        Vertex { position: [1.000000, -1.000000, -1.000000], normal: [0.0000, 0.0000, -1.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, 1.000000, -1.000000], normal: [0.0000, 0.0000, -1.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, 1.000000, -1.000000], normal: [0.0000, 0.0000, -1.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, -1.000000, -1.000000], normal: [0.0000, 0.0000, -1.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, 1.000000, -1.000000], normal: [0.0000, 0.0000, -1.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, -1.000000, -1.000000], normal: [0.0000, 0.0000, -1.0000], color: [1.0, 0.35, 0.137]},

        // top face
        Vertex { position: [-1.000000, -1.000000, 1.000000], normal: [0.0000, -1.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, -1.000000, 1.000000], normal: [0.0000, -1.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, -1.000000, -1.000000], normal: [0.0000, -1.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, -1.000000, 1.000000], normal: [0.0000, -1.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, -1.000000, -1.000000], normal: [0.0000, -1.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, -1.000000, -1.000000], normal: [0.0000, -1.0000, 0.0000], color: [1.0, 0.35, 0.137]},

        // bottom face
        Vertex { position: [1.000000, 1.000000, 1.000000], normal: [0.0000, 1.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, 1.000000, 1.000000], normal: [0.0000, 1.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, 1.000000, -1.000000], normal: [0.0000, 1.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, 1.000000, 1.000000], normal: [0.0000, 1.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, 1.000000, -1.000000], normal: [0.0000, 1.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, 1.000000, -1.000000], normal: [0.0000, 1.0000, 0.0000], color: [1.0, 0.35, 0.137]},

        // left face
        Vertex { position: [-1.000000, -1.000000, -1.000000], normal: [-1.0000, 0.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, 1.000000, -1.000000], normal: [-1.0000, 0.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, 1.000000, 1.000000], normal: [-1.0000, 0.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, -1.000000, -1.000000], normal: [-1.0000, 0.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, 1.000000, 1.000000], normal: [-1.0000, 0.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [-1.000000, -1.000000, 1.000000], normal: [-1.0000, 0.0000, 0.0000], color: [1.0, 0.35, 0.137]},

        // right face
        Vertex { position: [1.000000, -1.000000, 1.000000], normal: [1.0000, 0.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, 1.000000, 1.000000], normal: [1.0000, 0.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, 1.000000, -1.000000], normal: [1.0000, 0.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, -1.000000, 1.000000], normal: [1.0000, 0.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, 1.000000, -1.000000], normal: [1.0000, 0.0000, 0.0000], color: [1.0, 0.35, 0.137]},
        Vertex { position: [1.000000, -1.000000, -1.000000], normal: [1.0000, 0.0000, 0.0000], color: [1.0, 0.35, 0.137]},
    ].iter().cloned()).unwrap();

    let mut dynamic_state = DynamicState { line_width: None, viewports: None, scissors: None, compare_mask: None, write_mask: None, reference: None };

    let (mut framebuffers, mut color_buffer, mut normal_buffer) = window_size_dependent_setup(device.clone(), &images, render_pass.clone(), &mut dynamic_state);

    let mut recreate_swapchain = false;

    let mut previous_frame_end = Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>;

    let rotation_start = Instant::now();

    loop {
        previous_frame_end.cleanup_finished();

        if recreate_swapchain {
            let dimensions = if let Some(dimensions) = window.get_inner_size() {
                let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
                mvp.projection = perspective(dimensions.0 as f32 / dimensions.1 as f32, 180.0, 0.01, 100.0);
                [dimensions.0, dimensions.1]
            } else {
                return;
            };

            let (new_swapchain, new_images) = match swapchain.recreate_with_dimension(dimensions) {
                Ok(r) => r,
                Err(SwapchainCreationError::UnsupportedDimensions) => continue,
                Err(err) => panic!("{:?}", err)
            };

            swapchain = new_swapchain;
            let (new_framebuffers, new_color_buffer, new_normal_buffer) = window_size_dependent_setup(device.clone(), &new_images, render_pass.clone(), &mut dynamic_state);
            framebuffers = new_framebuffers;
            color_buffer = new_color_buffer;
            normal_buffer = new_normal_buffer;

            recreate_swapchain = false;
        }

        let uniform_buffer_subbuffer = {
            let elapsed = rotation_start.elapsed().as_secs() as f64 + rotation_start.elapsed().subsec_nanos() as f64 / 1_000_000_000.0;
            let elapsed_as_radians = elapsed * pi::<f64>() / 180.0;
            let mut model:TMat4<f32> = rotate_normalized_axis(&identity(), elapsed_as_radians as f32 * 50.0, &vec3(0.0, 0.0, 1.0));
            model = rotate_normalized_axis(&model, elapsed_as_radians as f32 * 30.0, &vec3(0.0, 1.0, 0.0));
            model = rotate_normalized_axis(&model, elapsed_as_radians as f32 * 20.0, &vec3(1.0, 0.0, 0.0));
            model = mvp.model * model;

            let uniform_data = deferred_vert::ty::MVP_Data {
                model: model.into(),
                view: mvp.view.into(),
                projection: mvp.projection.into(),
            };

            uniform_buffer.next(uniform_data).unwrap()
        };

        let ambient_uniform_subbuffer = {
            let uniform_data = ambient_frag::ty::Ambient_Data {
                color: ambient_light.color.into(),
                intensity: ambient_light.intensity.into()
            };

            ambient_buffer.next(uniform_data).unwrap()
        };

        let deferred_set = Arc::new(PersistentDescriptorSet::start(deferred_pipeline.clone(), 0)
            .add_buffer(uniform_buffer_subbuffer.clone()).unwrap()
            .build().unwrap());
        let ambient_set = Arc::new(PersistentDescriptorSet::start(ambient_pipeline.clone(), 0)
            .add_image(color_buffer.clone()).unwrap()
            .add_image(normal_buffer.clone()).unwrap()
            .add_buffer(uniform_buffer_subbuffer.clone()).unwrap()
            .add_buffer(ambient_uniform_subbuffer.clone()).unwrap()
            .build().unwrap());

        let (image_num, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                recreate_swapchain = true;
                continue;
            },
            Err(err) => panic!("{:?}", err)
        };

        let clear_values = vec![[0.0, 0.0, 0.0, 1.0].into(), [0.0, 0.0, 0.0, 1.0].into(), [0.0, 0.0, 0.0, 1.0].into(), 1f32.into()];

        let mut commands = AutoCommandBufferBuilder::primary_one_time_submit(device.clone(), queue.family()).unwrap()
            .begin_render_pass(framebuffers[image_num].clone(), false, clear_values)
            .unwrap()
            .draw(deferred_pipeline.clone(), &dynamic_state, vertex_buffer.clone(), deferred_set.clone(), ())
            .unwrap()
            .next_subpass(false)
            .unwrap();

        let mut directional_uniform_subbuffer = generate_directional_buffer(&directional_buffer, &directional_light_r);
        let mut directional_set = Arc::new(PersistentDescriptorSet::start(directional_pipeline.clone(), 0)
            .add_image(color_buffer.clone()).unwrap()
            .add_image(normal_buffer.clone()).unwrap()
            .add_buffer(uniform_buffer_subbuffer.clone()).unwrap()
            .add_buffer(directional_uniform_subbuffer.clone()).unwrap()
            .build().unwrap());
        commands = commands
            .draw(directional_pipeline.clone(), &dynamic_state, vertex_buffer.clone(), directional_set.clone(), ())
            .unwrap();

        directional_uniform_subbuffer = generate_directional_buffer(&directional_buffer, &directional_light_g);
        directional_set = Arc::new(PersistentDescriptorSet::start(directional_pipeline.clone(), 0)
            .add_image(color_buffer.clone()).unwrap()
            .add_image(normal_buffer.clone()).unwrap()
            .add_buffer(uniform_buffer_subbuffer.clone()).unwrap()
            .add_buffer(directional_uniform_subbuffer.clone()).unwrap()
            .build().unwrap());
        commands = commands
            .draw(directional_pipeline.clone(), &dynamic_state, vertex_buffer.clone(), directional_set.clone(), ())
            .unwrap();

        directional_uniform_subbuffer = generate_directional_buffer(&directional_buffer, &directional_light_b);
        directional_set = Arc::new(PersistentDescriptorSet::start(directional_pipeline.clone(), 0)
            .add_image(color_buffer.clone()).unwrap()
            .add_image(normal_buffer.clone()).unwrap()
            .add_buffer(uniform_buffer_subbuffer.clone()).unwrap()
            .add_buffer(directional_uniform_subbuffer.clone()).unwrap()
            .build().unwrap());
        commands = commands
            .draw(directional_pipeline.clone(), &dynamic_state, vertex_buffer.clone(), directional_set.clone(), ())
            .unwrap();

        let command_buffer = commands
            .draw(ambient_pipeline.clone(), &dynamic_state, vertex_buffer.clone(), ambient_set.clone(), ())
            .unwrap()
            .end_render_pass()
            .unwrap()
            .build()
            .unwrap();

        let future = previous_frame_end.join(acquire_future)
            .then_execute(queue.clone(), command_buffer).unwrap()
            .then_swapchain_present(queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                // This wait is required when using NVIDIA or running on macOS. See https://github.com/vulkano-rs/vulkano/issues/1247
                future.wait(None).unwrap();
                previous_frame_end = Box::new(future) as Box<_>;
            }
            Err(FlushError::OutOfDate) => {
                recreate_swapchain = true;
                previous_frame_end = Box::new(sync::now(device.clone())) as Box<_>;
            }
            Err(e) => {
                println!("{:?}", e);
                previous_frame_end = Box::new(sync::now(device.clone())) as Box<_>;
            }
        }

        let mut done = false;
        events_loop.poll_events(|ev| {
            match ev {
                Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => done = true,
                Event::WindowEvent { event: WindowEvent::Resized(_), .. } => recreate_swapchain = true,
                _ => ()
            }
        });
        if done { return; }
    }
}

fn generate_directional_buffer(
    pool: &vulkano::buffer::cpu_pool::CpuBufferPool<directional_frag::ty::Directional_Light_Data>,
    light: &DirectionalLight
) -> vulkano::buffer::cpu_pool::CpuBufferPoolSubbuffer<directional_frag::ty::Directional_Light_Data, Arc<StdMemoryPool>> {
    let uniform_data = directional_frag::ty::Directional_Light_Data {
        position: light.position.into(),
        color: light.color.into()
    };

    pool.next(uniform_data).unwrap()
}

/// This method is called once during initialization, then again whenever the window is resized
/// stolen from the vulkano example
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