use vulkano;
pub mod loader;
mod model;

pub use loader::ColoredVertex;
pub use loader::DummyVertex;
pub use loader::NormalVertex;
pub use model::Model;

vulkano::impl_vertex!(NormalVertex, position, normal, color);
vulkano::impl_vertex!(ColoredVertex, position, color);
vulkano::impl_vertex!(DummyVertex, position);
