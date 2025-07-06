use std::io::{self, BufRead, Read};
use std::mem;
use std::rc::Rc;

use crate::model::config::Config;
pub struct TransformerWeights {
    token_embedding_table: Rc<Vec<f32>>, // (vocab_size, dim)
    // weights for rmsnorms
    rms_att_weight: Vec<f32>, // (layer, dim,)
    rms_ffn_weight: Vec<f32>, // (layer, dim,)
    // weights for matmuls. note dim == n_heads * head_size
    wq: Vec<f32>, // (layer, dim, n_heads * head_size)
    wk: Vec<f32>, // (layer, dim, n_kv_heads * head_size)
    wv: Vec<f32>, // (layer, dim, n_kv_heads * head_size)
    wo: Vec<f32>, // (layer, n_heads * head_size, dim)
    // weights for ffn
    w1: Vec<f32>, // (layer, hidden_dim, dim)
    w2: Vec<f32>, // (layer, dim, hidden_dim)
    w3: Vec<f32>, // (layer, hidden_dim, dim)
    // final rmsnorm
    rms_final_weight: Vec<f32>, // (dim,)
    // (optional) classifier weights for the logits, on the last layer
    wcls: Rc<Vec<f32>>,
}

impl TransformerWeights {
    pub fn from_reader(
        reader: &mut dyn BufRead,
        config: &Config,
    ) -> io::Result<TransformerWeights> {
        let head_size = config.dim / config.n_heads;

        let token_embedding_table = {
            let len = (config.vocab_size * config.dim) as usize;
            Rc::new(Self::read_f32_vec(reader, len)?)
        };

        let rms_att_weight = {
            let len = (config.n_layers * config.dim) as usize;
            Self::read_f32_vec(reader, len)?
        };

        let wq = {
            let len = (config.n_layers * config.dim * config.n_heads * head_size) as usize;
            Self::read_f32_vec(reader, len)?
        };

        let wk = {
            let len = (config.n_layers * config.dim * config.n_kv_heads * head_size) as usize;
            Self::read_f32_vec(reader, len)?
        };

        let wv = {
            let len = (config.n_layers * config.dim * config.n_kv_heads * head_size) as usize;
            Self::read_f32_vec(reader, len)?
        };

        let wo = {
            let len = (config.n_layers * config.n_heads * head_size * config.dim) as usize;
            Self::read_f32_vec(reader, len)?
        };

        let rms_ffn_weight = {
            let len = (config.n_layers * config.dim) as usize;
            Self::read_f32_vec(reader, len)?
        };

        let w1 = {
            let len = (config.n_layers * config.hidden_dim * config.dim) as usize;
            Self::read_f32_vec(reader, len)?
        };

        let w2 = {
            let len = (config.n_layers * config.dim * config.hidden_dim) as usize;
            Self::read_f32_vec(reader, len)?
        };

        let w3 = {
            let len = (config.n_layers * config.hidden_dim * config.dim) as usize;
            Self::read_f32_vec(reader, len)?
        };

        let rms_final_weight = {
            let len = config.dim as usize;
            Self::read_f32_vec(reader, len)?
        };

        // from llama2.c:
        // skip what used to be freq_cis_real and freq_cis_imag (for RoPE)
        let skip_bytes = config.seq_len * head_size * (mem::size_of::<f32>() as u32);
        reader.take(skip_bytes as u64);

        let wcls = if config.shared_weights {
            Rc::clone(&token_embedding_table)
        } else {
            let len = (config.vocab_size * config.dim) as usize;
            Rc::new(Self::read_f32_vec(reader, len)?)
        };

        Ok(TransformerWeights {
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
}

#[cfg(test)]
mod tests {
    use std::{fs::File, io::BufReader};

    use super::*;
    use crate::model::config::Config;

    #[test]
    fn test_transformer_weights_from_file() -> io::Result<()> {
        let file = File::open("stories260K/stories260K.bin")?;
        let mut reader = BufReader::new(file);
        let config = Config::from_reader(&mut reader)?;
        let weights = TransformerWeights::from_reader(&mut reader, &config)?;

        assert_eq!(
            (config.vocab_size * config.dim) as usize,
            weights.token_embedding_table.len()
        );
        assert_eq!(
            vec![-0.2947487831f32, 0.5618956089, 0.1754225343],
            weights.token_embedding_table[..3]
        );
        assert_eq!(
            vec![-0.2340072989f32, -0.1648577899, -0.4240325987],
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

        assert_eq!(
            (config.n_layers * config.dim * config.n_heads * (config.dim / config.n_heads))
                as usize,
            weights.wq.len()
        );
        assert_eq!(
            vec![0.0443927869, 0.0797477067, -0.0089253467],
            weights.wq[..3]
        );
        assert_eq!(
            vec![-0.0594694205, 0.0033121579, -0.1890138239],
            weights.wq[weights.wq.len() - 3..]
        );

        assert_eq!(
            (config.n_layers * config.dim * config.n_kv_heads * (config.dim / config.n_heads))
                as usize,
            weights.wk.len()
        );
        assert_eq!(
            vec![0.2161301076, 0.1840710789, -0.2169892937],
            weights.wk[..3]
        );
        assert_eq!(
            vec![0.4967898428, -0.3751862049, -0.2651301622],
            weights.wk[weights.wk.len() - 3..]
        );

        assert_eq!(
            (config.n_layers * config.dim * config.n_kv_heads * (config.dim / config.n_heads))
                as usize,
            weights.wv.len()
        );
        assert_eq!(
            vec![0.0911254361, 0.0370284133, -0.0827622712],
            weights.wv[..3]
        );
        assert_eq!(
            vec![0.0126384906, -0.0676224604, 0.0883691236],
            weights.wv[weights.wv.len() - 3..]
        );

        assert_eq!(
            (config.n_layers * config.n_heads * (config.dim / config.n_heads) * config.dim)
                as usize,
            weights.wo.len()
        );
        assert_eq!(
            vec![0.0173441470, 0.0046848683, 0.0651651025],
            weights.wo[..3]
        );
        assert_eq!(
            vec![-0.0930427834, -0.0208466128, 0.0062035914],
            weights.wo[weights.wo.len() - 3..]
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

        assert_eq!(
            (config.n_layers * config.hidden_dim * config.dim) as usize,
            weights.w1.len()
        );
        assert_eq!(
            vec![0.2136866599, -0.0463941656, -0.1967670619],
            weights.w1[..3]
        );
        assert_eq!(
            vec![-0.0355275273, -0.1373713762, -0.1435981095],
            weights.w1[weights.w1.len() - 3..]
        );

        assert_eq!(
            (config.n_layers * config.dim * config.hidden_dim) as usize,
            weights.w2.len()
        );
        assert_eq!(
            vec![0.1349841952, 0.0671184734, 0.1124933064],
            weights.w2[..3]
        );
        assert_eq!(
            vec![-0.0748373643, -0.0097656352, -0.0035068716],
            weights.w2[weights.w2.len() - 3..]
        );

        assert_eq!(
            (config.n_layers * config.hidden_dim * config.dim) as usize,
            weights.w3.len()
        );
        assert_eq!(
            vec![0.0703711882, 0.0479559004, 0.1033418030],
            weights.w3[..3]
        );
        assert_eq!(
            vec![-0.0983150154, 0.2827627361, 0.1060319394],
            weights.w3[weights.w3.len() - 3..]
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
            weights.wcls.len(),
        );
        assert!(Rc::ptr_eq(&weights.wcls, &weights.token_embedding_table));

        Ok(())
    }
}
