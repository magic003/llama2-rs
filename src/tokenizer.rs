use std::fs::File;
use std::io::{self, BufReader, Read};
use std::mem;

pub struct Tokenizer {
    vocab_size: u32,
    max_token_length: u32,
    vocab_scores: Vec<f32>,
    vocab_tokens: Vec<String>,
}

const BOS_TOKEN: u32 = 1;

// The ASCII characters saved as an array of strings.
static BYTE_PIECES: [&'static str; 128] = [
    "\x00", "\x01", "\x02", "\x03", "\x04", "\x05", "\x06", "\x07", "\x08", "\x09", "\x0A", "\x0B",
    "\x0C", "\x0D", "\x0E", "\x0F", "\x10", "\x11", "\x12", "\x13", "\x14", "\x15", "\x16", "\x17",
    "\x18", "\x19", "\x1A", "\x1B", "\x1C", "\x1D", "\x1E", "\x1F", "\x20", "\x21", "\x22", "\x23",
    "\x24", "\x25", "\x26", "\x27", "\x28", "\x29", "\x2A", "\x2B", "\x2C", "\x2D", "\x2E", "\x2F",
    "\x30", "\x31", "\x32", "\x33", "\x34", "\x35", "\x36", "\x37", "\x38", "\x39", "\x3A", "\x3B",
    "\x3C", "\x3D", "\x3E", "\x3F", "\x40", "\x41", "\x42", "\x43", "\x44", "\x45", "\x46", "\x47",
    "\x48", "\x49", "\x4A", "\x4B", "\x4C", "\x4D", "\x4E", "\x4F", "\x50", "\x51", "\x52", "\x53",
    "\x54", "\x55", "\x56", "\x57", "\x58", "\x59", "\x5A", "\x5B", "\x5C", "\x5D", "\x5E", "\x5F",
    "\x60", "\x61", "\x62", "\x63", "\x64", "\x65", "\x66", "\x67", "\x68", "\x69", "\x6A", "\x6B",
    "\x6C", "\x6D", "\x6E", "\x6F", "\x70", "\x71", "\x72", "\x73", "\x74", "\x75", "\x76", "\x77",
    "\x78", "\x79", "\x7A", "\x7B", "\x7C", "\x7D", "\x7E", "\x7F",
];

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

    pub fn decode(&self, prev_token: u32, token: u32) -> &str {
        let mut piece: &str = self.vocab_tokens.get(token as usize).unwrap();
        // from llama2.c:
        // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
        if prev_token == BOS_TOKEN && piece.chars().next() == Some(' ') {
            piece = &piece[1..];
        }
        // some tokens designate raw bytes, e.g. "<0x01>"
        // extract the hex value and get the corresponding byte
        println!("token: {}, piece: {}", token, piece);
        if piece.len() == 6 && piece.starts_with("<0x") && piece.ends_with('>') {
            let hex_str = &piece[3..piece.len() - 1];
            if let Ok(byte) = u8::from_str_radix(hex_str, 16) {
                // add 3 because the first 3 tokens are <unk>, <s>, and </s>
                piece = BYTE_PIECES[byte as usize + 3];
            }
        }
        piece
    }
}

#[cfg(test)]
mod tests {
    use std::usize;

    use super::*;

    const STORIES_260K_TOKENIZER_PATH: &str = "stories260K/tok512.bin";
    const VOCAB_SIZE: usize = 512;

    #[test]
    fn test_tokenizer_new() -> io::Result<()> {
        let tokenizer = Tokenizer::new(STORIES_260K_TOKENIZER_PATH, VOCAB_SIZE as u32)?;
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

    #[test]
    fn test_decode() {
        let tokenizer = Tokenizer::new(STORIES_260K_TOKENIZER_PATH, VOCAB_SIZE as u32).unwrap();

        // test decoding of a token after BOS, with leading whitespace
        let decoded = tokenizer.decode(1, 300);
        assert_eq!("ha", decoded);

        // test decoding of a token not after BOS, with leading whitespace
        let decoded = tokenizer.decode(2, 300);
        assert_eq!(" ha", decoded);

        // test decoding of a token after BOS, without leading whitespace
        let decoded = tokenizer.decode(1, 302);
        assert_eq!("en", decoded);

        // test decoding of a token that designates a raw byte
        let decoded = tokenizer.decode(1, 65);
        assert_eq!("A", decoded);
    }
}
