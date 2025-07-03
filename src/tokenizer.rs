use std::fs::File;
use std::io::{self, BufReader, Read};
use std::mem;

pub struct Tokenizer {
    vocab_size: u32,
    max_token_length: u32,
    vocab_scores: Vec<f32>,
    vocab_tokens: Vec<String>,
}

impl Tokenizer {
    pub fn new(tokenizer_path: &str, vocab_size: u32) -> io::Result<Self> {
        let file = File::open(tokenizer_path)?;
        let mut reader = BufReader::new(file);

        let mut buf = [0; mem::size_of::<u32>()];
        reader.read_exact(buf.as_mut_slice())?;
        let max_token_length = u32::from_le_bytes(buf);

        let mut vocab_scores = Vec::with_capacity(vocab_size as usize);
        let mut vocab_tokens = Vec::with_capacity(vocab_size as usize);
        for _ in 0..vocab_size {
            reader.read_exact(buf.as_mut_slice())?;
            vocab_scores.push(f32::from_le_bytes(buf));

            reader.read_exact(buf.as_mut_slice())?;
            let len = u32::from_le_bytes(buf);
            let mut token_buf = vec![0; len as usize];
            reader.read_exact(token_buf.as_mut_slice())?;
            let token = String::from_utf8(token_buf)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            vocab_tokens.push(token);
        }

        Ok(Tokenizer {
            vocab_size,
            max_token_length,
            vocab_scores,
            vocab_tokens,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::usize;

    use super::*;

    #[test]
    fn test_tokenizer_new() -> io::Result<()> {
        const VOCAB_SIZE: usize = 512;
        let tokenizer = Tokenizer::new("stories260K/tok512.bin", VOCAB_SIZE as u32)?;
        assert_eq!(VOCAB_SIZE, tokenizer.vocab_size as usize);
        assert_eq!(7, tokenizer.max_token_length);
        assert_eq!(VOCAB_SIZE, tokenizer.vocab_scores.len());
        assert_eq!(VOCAB_SIZE, tokenizer.vocab_tokens.len());

        // spot check some tokens
        assert_eq!(0.0, *tokenizer.vocab_scores.get(0).unwrap());
        assert_eq!("<unk>", tokenizer.vocab_tokens.get(0).unwrap());

        assert_eq!(-41.0, *tokenizer.vocab_scores.get(300).unwrap());
        assert_eq!(" ha", tokenizer.vocab_tokens.get(300).unwrap());

        assert_eq!(-252.0, *tokenizer.vocab_scores.get(VOCAB_SIZE - 1).unwrap());
        assert_eq!(
            "\u{200a}",
            tokenizer.vocab_tokens.get(VOCAB_SIZE - 1).unwrap()
        );

        Ok(())
    }
}
