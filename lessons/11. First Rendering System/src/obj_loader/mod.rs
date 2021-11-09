// Copyright (c) 2020 taidaesal
// Licensed under the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>

mod loader;
mod face;
mod vertex;
pub use self::loader::Loader as Loader;

use std::fmt;

/// A vertex type intended to be used to provide dummy rendering
/// data for rendering passes that do not require geometry data.
/// This is due to a quirk of the Vulkan API in that *all*
/// render passes require some sort of input.
#[derive(Default, Debug, Clone)]
pub struct DummyVertex {
    /// A regular position vector with the z-value shaved off for space.
    /// This assumes the shaders will take a `vec2` and transform it as
    /// needed.
    pub position: [f32; 2]
}

impl DummyVertex {
    /// Creates an array of `DummyVertex` values.
    ///
    /// This is intended to compliment the use of this data type for passing to
    /// deferred rendering passes that do not actually require geometry input.
    /// This list will draw a square across the entire rendering area. This will
    /// cause the fragment shaders to execute on all pixels in the rendering
    /// area.
    ///
    /// # Example
    ///
    /// ```glsl
    /// #version 450
    ///
    ///layout(location = 0) in vec2 position;
    ///
    ///void main() {
    ///    gl_Position = vec4(position, 0.0, 1.0);
    ///}
    /// ```
    pub fn list() -> [DummyVertex; 6] {
        [
            DummyVertex { position: [-1.0, -1.0] },
            DummyVertex { position: [-1.0, 1.0] },
            DummyVertex { position: [1.0, 1.0] },
            DummyVertex { position: [-1.0, -1.0] },
            DummyVertex { position: [1.0, 1.0] },
            DummyVertex { position: [1.0, -1.0] }
        ]
    }
}

/// A structure for the vertex information used in earlier lessons
#[derive(Default, Debug, Clone)]
pub struct ColoredVertex {
    pub position: [f32; 3],
    pub color: [f32; 3]
}

/// A structure used for the vertex information starting
/// from our lesson on lighting
#[derive(Default, Debug, Clone)]
pub struct NormalVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub color: [f32; 3],
}

impl fmt::Display for DummyVertex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pos = format!("[{:.6}, {:.6}]", self.position[0], self.position[1]);
        write!(f, "DummyVertex {{ position: {} }}", pos)
    }
}

impl fmt::Display for ColoredVertex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pos = format!("[{:.6}, {:.6}, {:.6}]", self.position[0], self.position[1], self.position[2]);
        let color = format!("[{:.6}, {:.6}, {:.6}]", self.color[0], self.color[1], self.color[2]);
        write!(f, "ColoredVertex {{ position: {}, color: {} }}", pos, color)
    }
}

impl fmt::Display for NormalVertex {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pos = format!("[{:.6}, {:.6}, {:.6}]", self.position[0], self.position[1], self.position[2]);
        let color = format!("[{:.6}, {:.6}, {:.6}]", self.color[0], self.color[1], self.color[2]);
        let norms = format!("[{:.6}, {:.6}, {:.6}]", self.normal[0], self.normal[1], self.normal[2]);
        write!(f, "NormalVertex {{ position: {}, normal: {}, color: {} }}", pos, norms, color)
    }
}
