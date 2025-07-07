use crate::model::config::Config;
use crate::model::weights::TransformerWeights;

pub struct Transformer {
    config: Config,
    weights: TransformerWeights,
}

impl Transformer {
    pub fn forward(&self, token: u32, pos: usize) -> Vec<f32> {
        // a few convenience variables
        let config = &self.config;
        let dim = config.dim as usize;
        let weights = &self.weights;

        let token = token as usize;
        let x = weights.token_embedding_table[token * dim..(token + 1) * dim].as_ref();

        for layer in 0..config.n_layers as usize {}
        vec![]
    }
}
