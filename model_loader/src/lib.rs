use vulkano;
pub mod loader;
mod model;

pub use model::Model as Model;
pub use loader::ColoredVertex as ColoredVertex;
pub use loader::DummyVertex as DummyVertex;
pub use loader::NormalVertex as NormalVertex;

vulkano::impl_vertex!(NormalVertex, position, normal, color);
vulkano::impl_vertex!(ColoredVertex, position, color);
vulkano::impl_vertex!(DummyVertex, position);