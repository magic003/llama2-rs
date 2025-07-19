use std::io::{self, BufRead};
use std::mem;
use std::rc::Rc;

use super::QuantizedTensor;
use super::config::Config;

/// Weights for the Transformer model.
pub struct TransformerWeights {
    // token embedding table
    pub q_tokens: Rc<QuantizedTensor>,   // (vocab_size, dim)
    pub token_embedding_table: Vec<f32>, // same, but dequantized
    // weights for rmsnorms
    pub rms_att_weight: Vec<f32>, // (layer, dim,)
    pub rms_ffn_weight: Vec<f32>, // (layer, dim,)
    // weights for matmuls. note dim == n_heads * head_size
    pub wq: Vec<QuantizedTensor>, // (layer, dim, n_heads * head_size)
    pub wk: Vec<QuantizedTensor>, // (layer, dim, n_kv_heads * head_size)
    pub wv: Vec<QuantizedTensor>, // (layer, dim, n_kv_heads * head_size)
    pub wo: Vec<QuantizedTensor>, // (layer, n_heads * head_size, dim)
    // weights for ffn
    pub w1: Vec<QuantizedTensor>, // (layer, hidden_dim, dim)
    pub w2: Vec<QuantizedTensor>, // (layer, dim, hidden_dim)
    pub w3: Vec<QuantizedTensor>, // (layer, hidden_dim, dim)
    // final rmsnorm
    pub rms_final_weight: Vec<f32>, // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    pub wcls: Rc<QuantizedTensor>,
}

impl TransformerWeights {
    /// Reads the transformer weights from a reader.
    pub fn from_reader(
        reader: &mut dyn BufRead,
        config: &Config,
    ) -> io::Result<TransformerWeights> {
        let head_size = config.dim / config.n_heads;
        let group_size = config.group_size;

        // first are the parameters that are kept in fp32 (the rmsnorm (1D) weights)
        let rms_att_weight = {
            let len = (config.n_layers * config.dim) as usize;
            Self::read_f32_vec(reader, len)?
        };
        let rms_ffn_weight = {
            let len = (config.n_layers * config.dim) as usize;
            Self::read_f32_vec(reader, len)?
        };
        let rms_final_weight = {
            let len = config.dim as usize;
            Self::read_f32_vec(reader, len)?
        };

        // now read all the quantized weights
        let q_tokens = {
            Rc::new(
                Self::read_quantized_tensors(
                    reader,
                    1,
                    config.vocab_size * config.dim,
                    group_size,
                )?
                .remove(0),
            )
        };
        let token_embedding_table = q_tokens.dequantize();

        let wq = Self::read_quantized_tensors(
            reader,
            config.n_layers,
            config.dim * config.n_heads * head_size,
            group_size,
        )?;
        let wk = Self::read_quantized_tensors(
            reader,
            config.n_layers,
            config.dim * config.n_kv_heads * head_size,
            group_size,
        )?;
        let wv = Self::read_quantized_tensors(
            reader,
            config.n_layers,
            config.dim * config.n_kv_heads * head_size,
            group_size,
        )?;
        let wo = Self::read_quantized_tensors(
            reader,
            config.n_layers,
            config.n_heads * head_size * config.dim,
            group_size,
        )?;

        let w1 = Self::read_quantized_tensors(
            reader,
            config.n_layers,
            config.hidden_dim * config.dim,
            group_size,
        )?;
        let w2 = Self::read_quantized_tensors(
            reader,
            config.n_layers,
            config.dim * config.hidden_dim,
            group_size,
        )?;
        let w3 = Self::read_quantized_tensors(
            reader,
            config.n_layers,
            config.hidden_dim * config.dim,
            group_size,
        )?;

        let wcls = if config.shared_weights {
            Rc::clone(&q_tokens)
        } else {
            Rc::new(
                Self::read_quantized_tensors(
                    reader,
                    1,
                    config.vocab_size * config.dim,
                    group_size,
                )?
                .remove(0),
            )
        };

        Ok(TransformerWeights {
            q_tokens,
            token_embedding_table,
            rms_att_weight,
            rms_ffn_weight,
            wq,
            wk,
            wv,
            wo,
            w1,
            w2,
            w3,
            rms_final_weight,
            wcls,
        })
    }

    fn read_f32_vec(reader: &mut dyn BufRead, len: usize) -> io::Result<Vec<f32>> {
        let f32_size = mem::size_of::<f32>();
        let size = len * f32_size;
        let mut buf = vec![0; size];
        reader.read_exact(buf.as_mut_slice())?;

        let vec: Vec<f32> = buf
            .chunks_exact(f32_size)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
            .collect();

        Ok(vec)
    }

    fn read_quantized_tensors(
        reader: &mut dyn BufRead,
        n: u32,
        size_each: u32,
        group_size: u32,
    ) -> io::Result<Vec<QuantizedTensor>> {
        let i8_size = mem::size_of::<i8>();
        let f32_size = mem::size_of::<f32>();
        let q_byte_size_each = (size_each as usize) * i8_size;
        let s_byte_size_each = (size_each / group_size) as usize * f32_size;
        let mut q_buf = vec![0; q_byte_size_each];
        let mut s_buf = vec![0; s_byte_size_each];

        (0..n)
            .map(|_| {
                reader.read_exact(q_buf.as_mut_slice())?;
                let q: Vec<i8> = q_buf
                    .chunks_exact(i8_size)
                    .map(|chunk| i8::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();

                reader.read_exact(s_buf.as_mut_slice())?;
                let s: Vec<f32> = s_buf
                    .chunks_exact(f32_size)
                    .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
                    .collect();

                Ok(QuantizedTensor {
                    q,
                    s,
                    gs: group_size,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::io::Seek;
    use std::{fs::File, io::BufReader};

    use super::super::config::Config;
    use super::*;

    #[test]
    fn test_transformer_weights_from_file() -> io::Result<()> {
        let file = File::open("stories260K/stories260K_q80.bin")?;
        let mut reader = BufReader::new(file);
        reader.seek_relative(2 * mem::size_of::<u32>() as i64)?; // Skip the first 8 bytes (magic number and version)
        let config = Config::from_reader(&mut reader)?;

        // skip header bytes
        reader.rewind()?;
        reader.seek_relative(256)?;

        let group_size = config.group_size;
        let weights = TransformerWeights::from_reader(&mut reader, &config)?;

        assert_eq!(
            (config.vocab_size * config.dim) as usize,
            weights.q_tokens.q.len()
        );
        assert_eq!(
            (config.vocab_size * config.dim / group_size) as usize,
            weights.q_tokens.s.len(),
        );
        assert_eq!(group_size, weights.q_tokens.gs,);
        assert_eq!(vec![-42, 81, 25], weights.q_tokens.q[..3]);
        assert_eq!(
            vec![-37, -26, -67],
            weights.q_tokens.q[weights.q_tokens.q.len() - 3..]
        );
        assert_eq!(
            vec![0.0069355448, 0.0061636097, 0.0069353422],
            weights.q_tokens.s[..3]
        );
        assert_eq!(
            vec![0.0063424981, 0.0065780268, 0.0062970924],
            weights.q_tokens.s[weights.q_tokens.s.len() - 3..]
        );

        assert_eq!(
            (config.vocab_size * config.dim) as usize,
            weights.token_embedding_table.len()
        );
        assert_eq!(
            vec![-0.29129288f32, 0.56177914, 0.17338862],
            weights.token_embedding_table[..3]
        );
        assert_eq!(
            vec![-0.23299243f32, -0.16372441, -0.4219052],
            weights.token_embedding_table[weights.token_embedding_table.len() - 3..]
        );

        assert_eq!(
            (config.n_layers * config.dim) as usize,
            weights.rms_att_weight.len()
        );
        assert_eq!(
            vec![0.9722429514, 0.6754933596, 0.8203195333],
            weights.rms_att_weight[..3]
        );
        assert_eq!(
            vec![1.1054599285, 1.1300727129, 1.0649559498],
            weights.rms_att_weight[weights.rms_att_weight.len() - 3..]
        );

        assert_eq!(config.n_layers as usize, weights.wq.len());
        let wq0 = &weights.wq[0];
        assert_eq!(
            (config.dim * config.n_heads * (config.dim / config.n_heads)) as usize,
            wq0.q.len()
        );
        assert_eq!(
            (config.dim * config.n_heads * (config.dim / config.n_heads) / group_size) as usize,
            wq0.s.len(),
        );
        assert_eq!(group_size, wq0.gs,);
        assert_eq!(vec![18, 33, -4], wq0.q[..3]);
        assert_eq!(vec![-19, 26, 28], wq0.q[wq0.q.len() - 3..]);
        assert_eq!(vec![0.0024166764, 0.0043652947, 0.0035389762], wq0.s[..3]);
        assert_eq!(
            vec![0.0040387376, 0.0073745283, 0.0012340999],
            wq0.s[wq0.s.len() - 3..]
        );

        assert_eq!(config.n_layers as usize, weights.wk.len());
        let wk0 = &weights.wk[0];
        assert_eq!(
            (config.dim * config.n_kv_heads * (config.dim / config.n_heads)) as usize,
            wk0.q.len()
        );
        assert_eq!(
            (config.dim * config.n_kv_heads * (config.dim / config.n_heads) / group_size) as usize,
            wk0.s.len(),
        );
        assert_eq!(group_size, wk0.gs,);
        assert_eq!(vec![126, 108, -127], wk0.q[..3]);
        assert_eq!(vec![59, -88, -22], wk0.q[wk0.q.len() - 3..]);
        assert_eq!(vec![0.0017085772, 0.0034466865, 0.0026363560], wk0.s[..3]);
        assert_eq!(
            vec![0.0032873671, 0.0018399471, 0.0085180933],
            wk0.s[wk0.s.len() - 3..]
        );

        assert_eq!(config.n_layers as usize, weights.wv.len());
        let wv0 = &weights.wv[0];
        assert_eq!(
            (config.dim * config.n_kv_heads * (config.dim / config.n_heads)) as usize,
            wv0.q.len()
        );
        assert_eq!(
            (config.dim * config.n_kv_heads * (config.dim / config.n_heads) / group_size) as usize,
            wv0.s.len(),
        );
        assert_eq!(group_size, wv0.gs,);
        assert_eq!(vec![75, 30, -68], wv0.q[..3]);
        assert_eq!(vec![-33, 8, 42], wv0.q[wv0.q.len() - 3..]);
        assert_eq!(vec![0.0012168435, 0.0011718184, 0.0006762989], wv0.s[..3]);
        assert_eq!(
            vec![0.0009439196, 0.00095155236, 0.0013974739],
            wv0.s[wv0.s.len() - 3..]
        );

        assert_eq!(config.n_layers as usize, weights.wo.len());
        let wo0 = &weights.wo[0];
        assert_eq!(
            (config.dim * config.n_heads * (config.dim / config.n_heads)) as usize,
            wo0.q.len()
        );
        assert_eq!(
            (config.dim * config.n_heads * (config.dim / config.n_heads) / group_size) as usize,
            wo0.s.len(),
        );
        assert_eq!(group_size, wo0.gs,);
        assert_eq!(vec![24, 7, 90], wo0.q[..3]);
        assert_eq!(vec![-111, 3, 31], wo0.q[wo0.q.len() - 3..]);
        assert_eq!(vec![0.0007201580, 0.0014941860, 0.0010181476], wo0.s[..3]);
        assert_eq!(
            vec![0.0014645642, 0.0007479792, 0.0014292379],
            wo0.s[wo0.s.len() - 3..]
        );

        assert_eq!(
            (config.n_layers * config.dim) as usize,
            weights.rms_ffn_weight.len()
        );
        assert_eq!(
            vec![0.7367770672, 0.5533834100, 0.6776317954],
            weights.rms_ffn_weight[..3]
        );
        assert_eq!(
            vec![0.8462600708, 0.9927387834, 1.2342611551],
            weights.rms_ffn_weight[weights.rms_ffn_weight.len() - 3..]
        );

        assert_eq!(config.n_layers as usize, weights.w1.len());
        let w10 = &weights.w1[0];
        assert_eq!((config.hidden_dim * config.dim) as usize, w10.q.len());
        assert_eq!(
            (config.hidden_dim * config.dim / group_size) as usize,
            w10.s.len(),
        );
        assert_eq!(group_size, w10.gs,);
        assert_eq!(vec![93, -20, -86], w10.q[..3]);
        assert_eq!(vec![4, 34, -30], w10.q[w10.q.len() - 3..]);
        assert_eq!(vec![0.0022891448, 0.0023302527, 0.0017877360], w10.s[..3]);
        assert_eq!(
            vec![0.0025160150, 0.0015472349, 0.0043522734],
            w10.s[w10.s.len() - 3..]
        );

        assert_eq!(config.n_layers as usize, weights.w2.len());
        let w20 = &weights.w2[0];
        assert_eq!((config.hidden_dim * config.dim) as usize, w20.q.len());
        assert_eq!(
            (config.hidden_dim * config.dim / group_size) as usize,
            w20.s.len(),
        );
        assert_eq!(group_size, w20.gs,);
        assert_eq!(vec![86, 43, 72], w20.q[..3]);
        assert_eq!(vec![70, 18, -50], w20.q[w20.q.len() - 3..]);
        assert_eq!(vec![0.0015668459, 0.0022174465, 0.0021896209], w20.s[..3]);
        assert_eq!(
            vec![0.0023135042, 0.0024053995, 0.0022704226],
            w20.s[w20.s.len() - 3..]
        );

        assert_eq!(config.n_layers as usize, weights.w3.len());
        let w30 = &weights.w3[0];
        assert_eq!((config.hidden_dim * config.dim) as usize, w30.q.len());
        assert_eq!(
            (config.hidden_dim * config.dim / group_size) as usize,
            w30.s.len(),
        );
        assert_eq!(group_size, w30.gs,);
        assert_eq!(vec![39, 27, 58], w30.q[..3]);
        assert_eq!(vec![35, -44, 36], w30.q[w30.q.len() - 3..]);
        assert_eq!(vec![0.0017923390, 0.0018956634, 0.0020396893], w30.s[..3]);
        assert_eq!(
            vec![0.0022133770, 0.0025261454, 0.0044109798],
            w30.s[w30.s.len() - 3..]
        );

        assert_eq!(config.dim as usize, weights.rms_final_weight.len());
        assert_eq!(
            vec![2.4583089352, 1.3673226833, 1.7504382133],
            weights.rms_final_weight[..3]
        );
        assert_eq!(
            vec![0.9720380902, 1.4526383877, 1.7077769041],
            weights.rms_final_weight[weights.rms_final_weight.len() - 3..]
        );

        assert_eq!(
            (config.vocab_size * config.dim) as usize,
            weights.wcls.q.len(),
        );
        assert!(Rc::ptr_eq(&weights.wcls, &weights.q_tokens));

        Ok(())
    }
}
