pub mod neighbor;
pub use neighbor::Neighbor;

pub const GRAPH_SLACK_FACTOR: f64 = 1.3_f64;

pub mod data_store;
pub use data_store::InmemDataset;

pub mod graph;
pub use graph::{InMemoryGraph, VertexAndNeighbors};

pub mod vertex;
pub use vertex::Vertex;
