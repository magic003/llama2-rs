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
    q: Vec<f32>,  // query (dim, )

    // kv cache
    key_cache: Vec<f32>,   // (layer, seq_len, kv_dim)
    value_cache: Vec<f32>, // (layer, seq_len, kv_dim)
}

const EPS: f32 = 1e-5;

impl Transformer {
    pub fn forward(&mut self, token: u32, pos: usize) -> Vec<f32> {
        // a few convenience variables
        let config = &self.config;
        let dim = config.dim as usize;
        let head_size = dim / config.n_heads as usize;
        let weights = &self.weights;
        let kv_dim = ((config.dim / config.n_heads) * config.n_kv_heads) as usize;

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

            // key and value from the kv cache
            let kv_size = config.seq_len as usize * kv_dim;
            let offset = layer * kv_size + pos * kv_dim;
            let k = &mut self.state.key_cache[offset..(offset + kv_dim)];
            let v = &mut self.state.value_cache[offset..(offset + kv_dim)];

            // qkv for this position
            let q = &mut self.state.q;
            let qw_size = dim * dim;
            let kvw_size = dim * kv_dim;
            let wq = &weights.wq[layer * qw_size..(layer + 1) * qw_size];
            let wk = &weights.wk[layer * kvw_size..(layer + 1) * kvw_size];
            let wv = &weights.wv[layer * kvw_size..(layer + 1) * kvw_size];
            nn::matmul(q, x, wq, dim, dim);
            nn::matmul(k, x, wk, kv_dim, dim);
            nn::matmul(v, x, wv, kv_dim, dim);

            // RoPE relative positional encoding: complex-valued rotation q and k in each head
            for i in (0..dim).step_by(2) {
                // RoPE is applied within each head.
                let index_within_head = i % head_size;
                let freq = 1.0 / 10000.0f32.powf(index_within_head as f32 / head_size as f32);
                let val = pos as f32 * freq;
                let fcr = val.cos();
                let fci = val.sin();
                // when using grouped query attention, kv_dim is less than dim.
                let rotn = if i < kv_dim { 2 } else { 1 };
                for v in 0..rotn {
                    let vec = if v == 0 {
                        &mut q[i..i + 2]
                    } else {
                        &mut k[i..i + 2]
                    };
                    let (v0, v1) = (vec[i], vec[i + 1]);
                    vec[i] = v0 * fcr - v1 * fci;
                    vec[i + 1] = v0 * fci + v1 * fcr;
                }
            }
        }
        vec![]
    }
}
