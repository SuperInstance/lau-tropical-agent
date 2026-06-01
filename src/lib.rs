//! # lau-tropical-agent
//!
//! Tropical geometry for agent decisions — the ℏ→0 computational limit where
//! everything becomes piecewise-linear.
//!
//! In the tropical limit, neural network activations become max-plus operations,
//! softmax becomes hardmax, and optimal decisions are tropical rational functions.

pub mod semiring;
pub mod matrix;
pub mod polynomial;
pub mod rational_map;
pub mod attention;
pub mod decision;
pub mod eigen;
pub mod shortest_path;
pub mod spectral;

pub use semiring::Tropical;
pub use matrix::TropicalMatrix;
pub use polynomial::TropicalPoly;
pub use rational_map::TropicalRationalMap;
pub use attention::TropicalAttention;
pub use decision::TropicalDecision;
pub use eigen::tropical_eigen;
pub use shortest_path::tropical_shortest_paths;
