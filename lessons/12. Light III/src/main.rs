// Copyright (c) 2022 taidaesal
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>

mod model;
mod obj_loader;
mod system;

use model::Model;
use system::DirectionalLight;
use system::System;

use vulkano::sync;
use vulkano::sync::GpuFuture;

use winit::event::{Event, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use nalgebra_glm::{look_at, pi, vec3};

use std::time::Instant;

fn main() {
    let event_loop = EventLoop::new();
    let mut system = System::new(&event_loop);

    system.set_view(&look_at(
        &vec3(0.0, 0.0, 0.01),
        &vec3(0.0, 0.0, 0.0),
        &vec3(0.0, -1.0, 0.0),
    ));

    let mut previous_frame_end =
        Some(Box::new(sync::now(system.device.clone())) as Box<dyn GpuFuture>);

    let mut cube1 = Model::new("data/models/cube.obj")
        .specular(0.5, 12.0)
        .build();
    cube1.translate(vec3(1.1, 0.0, -2.0));

    let mut cube2 = Model::new("data/models/cube.obj")
        .specular(0.5, 128.0)
        .build();
    cube2.translate(vec3(-1.1, 0.0, -2.0));

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
            system.recreate_swapchain();
        }
        Event::RedrawEventsCleared => {
            previous_frame_end
                .as_mut()
                .take()
                .unwrap()
                .cleanup_finished();

            let elapsed = rotation_start.elapsed().as_secs() as f32
                + rotation_start.elapsed().subsec_nanos() as f32 / 1_000_000_000.0;
            let elapsed_as_radians = elapsed * 30.0 * (pi::<f32>() / 180.0);

            let x: f32 = 3.0 * elapsed_as_radians.cos();
            let z: f32 = -2.0 + (3.0 * elapsed_as_radians.sin());

            let directional_light = DirectionalLight::new([x, 0.0, z, 1.0], [1.0, 1.0, 1.0]);

            system.start();
            system.geometry(&mut cube1);
            system.geometry(&mut cube2);
            system.ambient();
            system.directional(&directional_light);
            system.light_object(&directional_light);
            system.finish(&mut previous_frame_end);
        }
        _ => (),
    });
}
