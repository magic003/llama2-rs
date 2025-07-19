use std::io::{self, BufRead};
use std::mem;

/// Configuration for the Transformer model.
pub struct Config {
    pub dim: u32,             // transformer dimension
    pub hidden_dim: u32,      // for ffn layers
    pub n_layers: u32,        // number of layers
    pub n_heads: u32,         // number of query heads
    pub n_kv_heads: u32, // number of key/value heads (can be < query heads because of multiquery)
    pub vocab_size: u32, // vocabulary size
    pub seq_len: u32,    // max sequence length
    pub shared_weights: bool, // whether to share the weights between embedding and output classifier layer
    pub group_size: u32,      // group size for quantization
}

impl Config {
    /// Creates a new `Config` instance by reading settings from a reader.
    pub fn from_reader(reader: &mut dyn BufRead) -> io::Result<Config> {
        let mut buf = [0; mem::size_of::<u32>()];

        reader.read_exact(buf.as_mut_slice())?;
        let dim = u32::from_le_bytes(buf);

        reader.read_exact(buf.as_mut_slice())?;
        let hidden_dim = u32::from_le_bytes(buf);

        reader.read_exact(buf.as_mut_slice())?;
        let n_layers = u32::from_le_bytes(buf);

        reader.read_exact(buf.as_mut_slice())?;
        let n_heads = u32::from_le_bytes(buf);

        reader.read_exact(buf.as_mut_slice())?;
        let n_kv_heads = u32::from_le_bytes(buf);

        reader.read_exact(buf.as_mut_slice())?;
        let vocab_size = u32::from_le_bytes(buf);

        reader.read_exact(buf.as_mut_slice())?;
        let seq_len = u32::from_le_bytes(buf);

        let mut byte_buffer = [0u8; 1];
        reader.read_exact(&mut byte_buffer)?;
        let shared_weights = byte_buffer[0] != 0;

        reader.read_exact(buf.as_mut_slice())?;
        let group_size = u32::from_le_bytes(buf);

        Ok(Config {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            seq_len,
            shared_weights,
            group_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{fs::File, io::BufReader};

    #[test]
    fn test_config_from_file() -> io::Result<()> {
        let file = File::open("stories260K/stories260K_q80.bin")?;
        let mut reader = BufReader::new(file);
        reader.seek_relative(2 * mem::size_of::<u32>() as i64)?; // Skip the first 8 bytes (magic number and version)
        let config = Config::from_reader(&mut reader)?;

        assert_eq!(64, config.dim);
        assert_eq!(172, config.hidden_dim);
        assert_eq!(5, config.n_layers);
        assert_eq!(8, config.n_heads);
        assert_eq!(4, config.n_kv_heads);
        assert_eq!(512, config.vocab_size);
        assert_eq!(512, config.seq_len);
        assert_eq!(true, config.shared_weights);
        assert_eq!(64, config.group_size);

        Ok(())
    }
}
