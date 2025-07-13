use std::time::Instant;

use crate::Sampler;
use crate::Tokenizer;
use crate::Transformer;
use crate::tokenizer;

pub struct Generator {
    tokenizer: Tokenizer,
    transformer: Transformer,
    sampler: Sampler,
    report_stats: bool,
}

impl Generator {
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

    pub fn generate(&mut self, prompt: &str, steps: u32) -> String {
        let mut output = String::new();
        let prompt_tokens = self.tokenizer.encode(prompt, true, false);

        let mut token_count = 0u32;
        let mut token = prompt_tokens[0];
        let start_time = Instant::now();
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
            output.extend(piece.chars());

            token = next;
        }

        if self.report_stats {
            eprintln!(
                "acheived tok/s {}\n",
                token_count as f64 / start_time.elapsed().as_millis() as f64 * 1000.0
            );
        }

        output
    }
}
