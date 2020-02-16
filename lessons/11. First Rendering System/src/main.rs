// Copyright (c) 2020 taidaesal
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>

mod model;
mod obj_loader;
mod system;

use model::Model;
use system::System;
use system::DirectionalLight;

use vulkano::sync;
use vulkano::sync::GpuFuture;

use winit::event_loop::{EventLoop, ControlFlow};
use winit::event::{Event, WindowEvent};

use nalgebra_glm::{vec3, look_at, pi};

use std::time::Instant;

fn main() {
    let event_loop = EventLoop::new();
    let mut system = System::new(&event_loop);

    system.set_view(&look_at(&vec3(0.0, 0.0, 0.01), &vec3(0.0, 0.0, 0.0), &vec3(0.0, -1.0, 0.0)));

    let mut previous_frame_end = Some(Box::new(sync::now(system.device.clone())) as Box<dyn GpuFuture>);

    let mut teapot = Model::new("./src/models/teapot.obj").build();
    teapot.translate(vec3(-5.0, 2.0, -5.0));

    let mut suzanne = Model::new("./src/models/suzanne.obj").build();
    suzanne.translate(vec3(5.0, 2.0, -5.0));

    let mut torus = Model::new("./src/models/torus.obj").build();
    torus.translate(vec3(0.0, -2.0, -5.0));

    let directional_light_r = DirectionalLight::new([-4.0, -4.0, 0.0, -2.0], [1.0, 0.0, 0.0]);
    let directional_light_g = DirectionalLight::new([4.0, -4.0, 0.0, -2.0], [0.0, 1.0, 0.0]);
    let directional_light_b = DirectionalLight::new([0.0, 4.0, 0.0, -2.0], [0.0, 0.0, 1.0]);

    let rotation_start = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { event: WindowEvent::CloseRequested, .. } => {
                *control_flow = ControlFlow::Exit;
            },
            Event::WindowEvent { event: WindowEvent::Resized(_), .. } => {
                system.recreate_swapchain();
            },
            Event::RedrawEventsCleared => {
                previous_frame_end.as_mut().take().unwrap().cleanup_finished();

                let elapsed = rotation_start.elapsed().as_secs() as f64 + rotation_start.elapsed().subsec_nanos() as f64 / 1_000_000_000.0;
                let elapsed_as_radians = elapsed * pi::<f64>() / 180.0;

                teapot.zero_rotation();
                teapot.rotate(elapsed_as_radians as f32 * 50.0, vec3(0.0, 0.0, 1.0));
                teapot.rotate(elapsed_as_radians as f32 * 30.0, vec3(0.0, 1.0, 0.0));
                teapot.rotate(elapsed_as_radians as f32 * 20.0, vec3(1.0, 0.0, 0.0));

                suzanne.zero_rotation();
                suzanne.rotate(elapsed_as_radians as f32 * 25.0, vec3(0.0, 0.0, 1.0));
                suzanne.rotate(elapsed_as_radians as f32 * 10.0, vec3(0.0, 1.0, 0.0));
                suzanne.rotate(elapsed_as_radians as f32 * 60.0, vec3(1.0, 0.0, 0.0));

                torus.zero_rotation();
                torus.rotate(elapsed_as_radians as f32 * 5.0, vec3(0.0, 0.0, 1.0));
                torus.rotate(elapsed_as_radians as f32 * 45.0, vec3(0.0, 1.0, 0.0));
                torus.rotate(elapsed_as_radians as f32 * 12.0, vec3(1.0, 0.0, 0.0));

                system.start();
                system.geometry(&mut teapot);
                system.geometry(&mut suzanne);
                system.geometry(&mut torus);
                system.ambient();
                system.directional(&directional_light_r);
                system.directional(&directional_light_g);
                system.directional(&directional_light_b);
                system.finish(&mut previous_frame_end);
            },
            _ => ()
        }
    });
}