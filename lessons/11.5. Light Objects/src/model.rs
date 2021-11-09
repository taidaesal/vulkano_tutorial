// Copyright (c) 2020 taidaesal
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>

#![allow(dead_code)]
use super::obj_loader::{ColoredVertex, Loader, NormalVertex};

use nalgebra_glm::{identity, rotate_normalized_axis, TMat4, TVec3, inverse_transpose, scale, translate, vec3};

/// Holds our data for a renderable model, including the model matrix data
///
/// Note: When building an instance of `Model` the loader will assume that
/// the input obj file is in clockwise winding order. If it is already in
/// counter-clockwise winding order, call `.invert_winding_order(false)`
/// when building the `Model`.
pub struct Model {
    data: Vec<NormalVertex>,
    translation: TMat4<f32>,
    rotation: TMat4<f32>,
    uniform_scale: f32,
    model: TMat4<f32>,
    normals: TMat4<f32>,
    // we might call multiple translation/rotation calls
    // in between asking for the model matrix. This lets
    // only recreate the model matrix when needed.
    requires_update: bool
}

pub struct ModelBuilder {
    file_name: String,
    custom_color: [f32; 3],
    invert: bool,
    scale_factor: f32,
}

impl ModelBuilder {
    fn new(file: String) -> ModelBuilder {
        ModelBuilder {
            file_name: file,
            custom_color: [1.0, 0.35, 0.137],
            invert: true,
            scale_factor: 1.0,
        }
    }

    pub fn build(self) -> Model {
        let loader = Loader::new(self.file_name.as_str(), self.custom_color, self.invert);
        Model {
            data: loader.as_normal_vertices(),
            translation: identity(),
            rotation: identity(),
            uniform_scale: self.scale_factor,
            model: identity(),
            normals: identity(),
            requires_update: false,
        }
    }

    pub fn color(mut self, new_color: [f32; 3]) -> ModelBuilder {
        self.custom_color = new_color;
        self
    }

    pub fn file(mut self, file: String) -> ModelBuilder {
        self.file_name = file;
        self
    }

    pub fn invert_winding_order(mut self, invert: bool) -> ModelBuilder {
        self.invert = invert;
        self
    }

    pub fn uniform_scale_factor(mut self, scale: f32) -> ModelBuilder {
        self.scale_factor = scale;
        self
    }
}

impl Model {
    pub fn new(file_name: &str) -> ModelBuilder {
        ModelBuilder::new(file_name.into())
    }

    pub fn color_data(&self) -> Vec<ColoredVertex> {
        let mut ret:Vec<ColoredVertex> = Vec::new();
        for v in &self.data {
            ret.push(
                ColoredVertex {
                    position: v.position,
                    color: v.color
                }
            );
        }
        ret
    }

    pub fn data(&self) -> Vec<NormalVertex> {
        self.data.clone()
    }

    pub fn model_matrices(&mut self) -> (TMat4<f32>, TMat4<f32>) {
        if self.requires_update {
            self.model = self.translation * self.rotation;
            self.model = scale(&self.model, &vec3(self.uniform_scale, self.uniform_scale, self.uniform_scale));
            self.normals = inverse_transpose(self.model);
            self.requires_update = false;
        }
        (self.model, self.normals)
    }

    pub fn rotate(&mut self, radians: f32, v: TVec3<f32>) {
        self.rotation = rotate_normalized_axis(&self.rotation, radians, &v);
        self.requires_update = true;
    }

    pub fn translate(&mut self, v: TVec3<f32>) {
        self.translation = translate(&self.translation, &v);
        self.requires_update = true;
    }

    /// Return the model's rotation to 0
    pub fn zero_rotation(&mut self) {
        self.rotation = identity();
        self.requires_update = true;
    }
}