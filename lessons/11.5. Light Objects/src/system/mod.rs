// Copyright (c) 2021 taidaesal
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>

mod system;

use nalgebra_glm::{vec3, TVec3};
pub use system::System;

#[derive(Default, Debug, Clone)]
pub struct AmbientLight {
    color: [f32; 3],
    intensity: f32,
}

#[derive(Default, Debug, Clone)]
pub struct DirectionalLight {
    position: [f32; 4],
    color: [f32; 3],
}

impl DirectionalLight {
    pub fn new(position: [f32; 4], color: [f32; 3]) -> DirectionalLight {
        DirectionalLight { position, color }
    }

    pub fn get_position(&self) -> TVec3<f32> {
        vec3(self.position[0], self.position[1], self.position[2])
    }
}
