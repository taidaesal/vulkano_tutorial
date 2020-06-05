// Copyright (c) 2020 taidaesal
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>

mod system;

pub use system::System as System;
use nalgebra_glm::{TVec3, vec3};

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
    pub fn new(in_position: [f32; 3], color: [f32; 3]) -> DirectionalLight {
        let position = [in_position[0],in_position[1],in_position[2],0.0];
        DirectionalLight {
            position,
            color
        }
    }

    pub fn get_position(&self) -> TVec3<f32> {
        vec3(self.position[0], self.position[1], self.position[2])
    }
}