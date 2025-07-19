//! This module provides the Transformer model implementation.
mod config;
pub mod quantize;
pub mod transformer;
mod weights;

/// The Transformer trait defines the interface for a Transformer model.
pub trait Transformer {
    fn forward(&mut self, token: u32, pos: usize) -> &[f32];
}
