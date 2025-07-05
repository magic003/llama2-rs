use std::fs::File;
use std::io::{self, BufReader, Read};
use std::mem;

pub struct Config {
    dim: u32,        // transformer dimension
    hidden_dim: u32, // for ffn layers
    n_layers: u32,   // number of layers
    n_heads: u32,    // number of query heads
    n_kv_heads: u32, // number of key/value heads (can be < query heads because of multiquery)
    vocab_size: u32, // vocabulary size
    seq_len: u32,    // max sequence length
}

impl Config {
    pub fn from_file(checkpoint: &str) -> io::Result<Config> {
        let file = File::open(checkpoint)?;
        let mut reader = BufReader::new(file);

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

        Ok(Config {
            dim,
            hidden_dim,
            n_layers,
            n_heads,
            n_kv_heads,
            vocab_size,
            seq_len,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_from_file() -> io::Result<()> {
        let config = Config::from_file("stories260K/stories260K.bin")?;

        assert_eq!(64, config.dim);
        assert_eq!(172, config.hidden_dim);
        assert_eq!(5, config.n_layers);
        assert_eq!(8, config.n_heads);
        assert_eq!(4, config.n_kv_heads);
        assert_eq!(512, config.vocab_size);
        assert_eq!(512, config.seq_len);

        Ok(())
    }
}
