use crate::model::config::Config;
use crate::model::weights::TransformerWeights;
use crate::nn;

pub struct Transformer {
    config: Config,
    weights: TransformerWeights,
    state: RunState,
}

struct RunState {
    xb: Vec<f32>, // activation inside a residual branch. (dim, )
}

const EPS: f32 = 1e-5;

impl Transformer {
    pub fn forward(&mut self, token: u32, pos: usize) -> Vec<f32> {
        // a few convenience variables
        let config = &self.config;
        let dim = config.dim as usize;
        let weights = &self.weights;

        let token = token as usize;
        let x = weights.token_embedding_table[token * dim..(token + 1) * dim].as_ref();

        for layer in 0..config.n_layers as usize {
            // attention rmsnorm
            nn::rmsnorm(
                &mut self.state.xb,
                x,
                &weights.rms_att_weight[layer * dim..(layer + 1) * dim],
                EPS,
            );
        }
        vec![]
    }
}
