mod system;
mod system_errors;

pub use system::System as System;

#[derive(Default, Debug, Clone)]
pub struct AmbientLight {
    color: [f32; 3],
    intensity: f32
}

#[derive(Default, Debug, Clone)]
pub struct DirectionalLight {
    position: [f32; 4],
    color: [f32; 3]
}

impl DirectionalLight {
    pub fn new(position: [f32; 4], color: [f32; 3]) -> DirectionalLight {
        DirectionalLight {
            position,
            color
        }
    }
}