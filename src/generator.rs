use std::time::Instant;

use crate::Sampler;
use crate::Tokenizer;
use crate::Transformer;
use crate::tokenizer;

/// A generator for Llama-2 Transformer model inference.
pub struct Generator {
    tokenizer: Tokenizer,
    transformer: Transformer,
    sampler: Sampler,
    report_stats: bool,
}

impl Generator {
    /// Creates a new `Generator` instance.
    pub fn new(
        tokenizer: Tokenizer,
        transformer: Transformer,
        sampler: Sampler,
        report_stats: bool,
    ) -> Self {
        Generator {
            tokenizer,
            transformer,
            sampler,
            report_stats,
        }
    }

    /// Generates text based on the provided prompt and number of steps.
    pub fn generate(&mut self, prompt: &str, steps: u32) -> String {
        let mut output = String::new();
        let prompt_tokens = self.tokenizer.encode(prompt, true, false);

        let mut token_count = 0u32;
        let mut token = prompt_tokens[0];
        let mut start_time = Option::None;
        for pos in 0..steps as usize {
            let logits = self.transformer.forward(token, pos);

            let next = if pos < prompt_tokens.len() - 1 {
                prompt_tokens[pos + 1]
            } else {
                self.sampler.sample(logits)
            };

            token_count += 1;

            // from llama2.c
            // data-dependent terminating condition: the BOS (=1) token delimits sequences
            if next == tokenizer::BOS_TOKEN {
                break;
            }

            let piece = self.tokenizer.decode(token, next);
            if !Self::is_bad_piece(piece) {
                output.extend(piece.chars());
            }

            token = next;

            // init the timer here because the first iteration can be slower
            if pos == 0 {
                start_time = Some(Instant::now());
            }
        }

        if self.report_stats {
            if let Some(start_time) = start_time {
                eprintln!(
                    "acheived tok/s {}\n",
                    token_count as f64 / start_time.elapsed().as_millis() as f64 * 1000.0
                );
            }
        }

        output
    }

    /// Checks if a piece of text is considered "bad" if it's not printable.
    fn is_bad_piece(piece: &str) -> bool {
        if piece.len() == 1 {
            let ch = piece.chars().nth(0).unwrap();
            return !(ch.is_ascii_graphic() || ch.is_ascii_whitespace());
        }
        false
    }
}
