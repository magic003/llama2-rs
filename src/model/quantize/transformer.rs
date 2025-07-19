use std::fs::File;
use std::io::BufReader;
use std::{io, thread};

use crate::nn;

use super::QuantizedTensor;
use super::config::Config;
use super::weights::TransformerWeights;

/// Transformer model for Llama-2.
pub struct Transformer {
    pub config: Config,
    weights: TransformerWeights,
    state: RunState,
}

/// State of the Transformer model during inference.
struct RunState {
    x: Vec<f32>,         // activation at the current time stamp. (dim, )
    xb: Vec<f32>,        // activation inside a residual branch. (dim, )
    xb2: Vec<f32>,       // another buffer for convenience. (dim, )
    hb: Vec<f32>,        // buffer for hidden dimension in the ffn. (hidden_dim, )
    hb2: Vec<f32>,       // buffer for hidden dimension in the ffn. (hidden_dim, )
    xq: QuantizedTensor, // quantized x (dim, )
    hq: QuantizedTensor, // quantized hb (hidden_dim, )
    q: Vec<f32>,         // query (dim, )
    k: Vec<f32>,         // key (dim, )
    v: Vec<f32>,         // value (dim, )
    att: Vec<f32>,       // attention values. (n_heads, seq_len)
    logits: Vec<f32>,    // output logits. (vocab_size, )

    // kv cache
    key_cache: Vec<f32>,   // (layer, seq_len, kv_dim)
    value_cache: Vec<f32>, // (layer, seq_len, kv_dim)
}

impl RunState {
    /// Creates a new `RunState` instance based on the provided configuration.
    pub fn new(config: &Config) -> RunState {
        let dim = config.dim as usize;
        let kv_dim = ((config.dim / config.n_heads) * config.n_kv_heads) as usize;
        let hidden_dim = config.hidden_dim as usize;

        RunState {
            x: vec![0.0; dim],
            xb: vec![0.0; dim],
            xb2: vec![0.0; dim],
            hb: vec![0.0; hidden_dim],
            hb2: vec![0.0; hidden_dim],
            xq: QuantizedTensor::new(&vec![0.0; dim], config.group_size),
            hq: QuantizedTensor::new(&vec![0.0; hidden_dim], config.group_size),
            q: vec![0.0; dim],
            k: vec![0.0; dim],
            v: vec![0.0; dim],
            att: vec![0.0; config.n_heads as usize * config.seq_len as usize],
            logits: vec![0.0; config.vocab_size as usize],

            key_cache: vec![0.0; config.n_layers as usize * config.seq_len as usize * kv_dim],
            value_cache: vec![0.0; config.n_layers as usize * config.seq_len as usize * kv_dim],
        }
    }
}

const EPS: f32 = 1e-5;

impl Transformer {
    /// Creates a new `Transformer` instance by reading the configuration and weights from the model's
    /// checkpoint file.
    pub fn from_file(checkpoint_path: &str) -> io::Result<Transformer> {
        let file = File::open(checkpoint_path)?;
        let mut reader = BufReader::new(file);
        let config = Config::from_reader(&mut reader)?;
        let weights = TransformerWeights::from_reader(&mut reader, &config)?;
        let state = RunState::new(&config);
        Ok(Transformer {
            config,
            weights,
            state,
        })
    }

    /// Runs the Transformer model forward pass with the given token and position.
    pub fn forward(&mut self, token: u32, pos: usize) -> &[f32] {
        // a few convenience variables
        let config = &self.config;
        let state = &mut self.state;
        let dim = config.dim as usize;
        let head_dim = dim / config.n_heads as usize;
        let weights = &self.weights;
        let kv_dim = ((config.dim / config.n_heads) * config.n_kv_heads) as usize;
        let hidden_dim = config.hidden_dim as usize;

        let token = token as usize;
        let x = &mut state.x;
        x.copy_from_slice(weights.token_embedding_table[token * dim..(token + 1) * dim].as_ref());

        for layer in 0..config.n_layers as usize {
            // attention rmsnorm
            nn::rmsnorm(
                &mut state.xb,
                x,
                &weights.rms_att_weight[layer * dim..(layer + 1) * dim],
                EPS,
            );

            // qkv matmuls for this position
            state.xq.quantize(x);

            let qw_size = dim * dim;
            let kvw_size = dim * kv_dim;
            let wq = &weights.wq[layer];
            let wk = &weights.wk[layer];
            let wv = &weights.wv[layer];
            super::matmul(&mut state.q, &state.xq, wq, dim, dim);
            super::matmul(&mut state.k, &state.xq, wk, kv_dim, dim);
            super::matmul(&mut state.v, &state.xq, wv, kv_dim, dim);

            // RoPE relative positional encoding: complex-valued rotation q and k in each head
            for i in (0..dim).step_by(2) {
                // RoPE is applied within each head.
                let index_within_head = i % head_dim;
                let freq = 1.0 / 10000.0f32.powf(index_within_head as f32 / head_dim as f32);
                let val = pos as f32 * freq;
                let fcr = val.cos();
                let fci = val.sin();
                // when using grouped query attention, kv_dim is less than dim.
                let rotn = if i < kv_dim { 2 } else { 1 };
                for v in 0..rotn {
                    let vec = if v == 0 {
                        &mut state.q[i..i + 2]
                    } else {
                        &mut state.k[i..i + 2]
                    };
                    let (v0, v1) = (vec[0], vec[1]);
                    vec[0] = v0 * fcr - v1 * fci;
                    vec[1] = v0 * fci + v1 * fcr;
                }
            }

            // save key,value at this time step (pos) to our kv cache
            let kv_size = config.seq_len as usize * kv_dim;
            let offset = layer * kv_size + pos * kv_dim;
            let key_cache_row = &mut state.key_cache[offset..(offset + kv_dim)];
            let value_cache_row = &mut state.value_cache[offset..(offset + kv_dim)];
            key_cache_row.copy_from_slice(&state.k);
            value_cache_row.copy_from_slice(&state.v);

            // multihead attention. Iterate over all heads.
            thread::scope(|s| {
                let att_heads = state.att.chunks_mut(config.seq_len as usize);
                let xb_heads = state.xb.chunks_mut(head_dim);
                for (h, (att, xb)) in att_heads.zip(xb_heads).enumerate() {
                    // query for this head
                    let q = &state.q[h * head_dim..(h + 1) * head_dim];
                    // key and value cache for this layer
                    let key_cache = &state.key_cache[layer * kv_size..(layer + 1) * kv_size];
                    let value_cache = &state.value_cache[layer * kv_size..(layer + 1) * kv_size];
                    // head index in the kv cache
                    let kv_h = h / (config.n_heads / config.n_kv_heads) as usize;
                    s.spawn(move || {
                        // iterate over all timesteps, including the current one.
                        for t in 0..=pos {
                            let k_start = t * kv_dim + kv_h * head_dim;
                            let k = &key_cache[k_start..k_start + head_dim];
                            let score = q
                                .iter()
                                .zip(k.iter())
                                .map(|(&q_val, &k_val)| q_val * k_val)
                                .sum::<f32>();
                            att[t] = score / (head_dim as f32).sqrt();
                        }

                        // softmax the attention scores, from 0..pos inclusively
                        nn::softmax(&mut att[..=pos]);

                        // weighted sum of the values, store back into xb
                        xb.fill(0.0);

                        for t in 0..=pos {
                            // get value vector for this head and this timestep
                            let v_start = t * kv_dim + kv_h * head_dim;
                            let v = &value_cache[v_start..v_start + head_dim];
                            // attention weight for this timestep
                            let a = att[t];

                            // accumulate the weighted value into xb
                            for i in 0..head_dim {
                                xb[i] += a * v[i];
                            }
                        }
                    });
                }
            });

            // final matmul to get the output of the attention layer
            state.xq.quantize(&state.xb);
            super::matmul(&mut state.xb2, &state.xq, &weights.wo[layer], dim, dim);

            // residual connection back into x
            for i in 0..dim {
                x[i] += state.xb2[i];
            }

            // ffn rmsnorm
            nn::rmsnorm(
                &mut state.xb,
                x,
                &weights.rms_ffn_weight[layer * dim..(layer + 1) * dim],
                EPS,
            );

            // SwiGLU non-linearity
            // SwiGLU(x) = silu(w1 @ x) * (w3 @ x)
            state.xq.quantize(&state.xb);
            super::matmul(
                &mut state.hb,
                &state.xq,
                &weights.w1[layer],
                hidden_dim,
                dim,
            );
            super::matmul(
                &mut state.hb2,
                &state.xq,
                &weights.w3[layer],
                hidden_dim,
                dim,
            );
            for (hb_val, hb2_val) in state.hb.iter_mut().zip(state.hb2.iter()) {
                let silu = *hb_val * 1.0 / (1.0 + (-*hb_val).exp());
                *hb_val = silu * hb2_val;
            }

            // final matmul to get the output of the ffn layer
            state.hq.quantize(&state.hb);
            super::matmul(
                &mut state.xb,
                &state.hq,
                &weights.w2[layer],
                dim,
                hidden_dim,
            );

            // residual connection back into x
            for i in 0..dim {
                x[i] += state.xb[i];
            }
        }
        // final rmsnorm
        // in llama2.c, it writes into x, but Rust doesn't allow it. It writes into xb instead.
        nn::rmsnorm(&mut state.xb, &x, &weights.rms_final_weight, EPS);

        // classifier into logits
        state.xq.quantize(&state.xb);
        super::matmul(
            &mut state.logits,
            &state.xq,
            &weights.wcls,
            config.vocab_size as usize,
            dim,
        );
        &state.logits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_state_new() {
        let config = Config {
            dim: 64,
            hidden_dim: 172,
            n_layers: 5,
            n_heads: 8,
            n_kv_heads: 4,
            vocab_size: 512,
            seq_len: 512,
            shared_weights: true,
            group_size: 4,
        };
        let state = RunState::new(&config);
        assert_eq!(64, state.x.len());
        assert_eq!(64, state.xb.len());
        assert_eq!(64, state.xb2.len());
        assert_eq!(172, state.hb.len());
        assert_eq!(172, state.hb2.len());
        assert_eq!(64, state.xq.q.len());
        assert_eq!(16, state.xq.s.len());
        assert_eq!(4, state.xq.gs);
        assert_eq!(172, state.hq.q.len());
        assert_eq!(43, state.hq.s.len());
        assert_eq!(4, state.hq.gs);
        assert_eq!(64, state.q.len());
        assert_eq!(64, state.k.len());
        assert_eq!(64, state.v.len());
        assert_eq!(8 * 512, state.att.len());
        assert_eq!(512, state.logits.len());

        // 5 layers, 512 seq_len, 32 kv_dim
        assert_eq!(5 * 512 * 32, state.key_cache.len());
        assert_eq!(5 * 512 * 32, state.value_cache.len());
    }
}
