use crate::nn;
use std::cmp::Ordering;

pub struct Sampler {
    temperature: f32,
    top_p: Option<f32>,
    rng_state: u64,
    logits: Vec<f32>,
}

impl Sampler {
    pub fn new(vocab_size: u32, temperature: f32, top_p: Option<f32>, rng_seed: u64) -> Self {
        Sampler {
            temperature,
            top_p,
            rng_state: rng_seed,
            logits: vec![0.0; vocab_size as usize],
        }
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        if self.temperature == 0.0 {
            // take the token with the highest probability
            return Self::argmax(logits) as u32;
        }

        // apply temperature to logits
        self.logits.copy_from_slice(logits);
        self.logits.iter_mut().for_each(|logit| {
            *logit /= self.temperature;
        });
        // apply softmax to get the probabilities
        nn::softmax(self.logits.as_mut_slice());
        // flip a (float) coin (this is our source of entropy for sampling)
        let coin = self.random_f32();
        return if let Some(top_p) = self.top_p {
            Self::sample_top_p(&self.logits, top_p, coin) as u32
        } else {
            Self::sample_mult(&self.logits, coin) as u32
        };
    }

    fn random_f32(&mut self) -> f32 {
        // copied from llama2.c
        // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        let state = &mut self.rng_state;
        *state ^= *state >> 12;
        *state ^= *state << 25;
        *state ^= *state >> 27;
        let val = state.wrapping_mul(0x2545F4914F6CDD1D) >> 32;

        // random f32 in [0,1)
        (val >> 8) as f32 / 16777216.0
    }

    /// Returns the index that has the highest probability.
    fn argmax(probabilities: &[f32]) -> usize {
        if probabilities.is_empty() {
            return 0;
        }

        let mut max_index = 0;
        let mut max_p = probabilities[0];
        for i in 1..probabilities.len() {
            if probabilities[i] > max_p {
                max_p = probabilities[i];
                max_index = i;
            }
        }

        max_index
    }

    fn sample_mult(probabilities: &[f32], coin: f32) -> usize {
        // sample from the probabilities using the coin flip
        let mut cumulative = 0.0;
        for (i, &p) in probabilities.iter().enumerate() {
            cumulative += p;
            if cumulative >= coin {
                return i;
            }
        }
        probabilities.len() - 1 // fallback to last index if no match found
    }

    fn sample_top_p(probabilities: &[f32], top_p: f32, coin: f32) -> usize {
        println!("Probabilities: {:?}", probabilities);
        // pre-filter the extremely low probability tokens
        let cutoff = (1.0 - top_p) / (probabilities.len() - 1) as f32;
        let mut probs: Vec<(usize, &f32)> = probabilities
            .iter()
            .enumerate()
            .filter(|&(_, &p)| p >= cutoff)
            .collect();
        probs.sort_by(|a, b| b.1.partial_cmp(a.1).unwrap_or(Ordering::Equal));

        // truncate the list where the cumulative probability exceeds top_p
        let mut cumulative = 0.0;
        let mut last_index = probs.len() - 1;
        for (i, (_, p)) in probs.iter().enumerate() {
            cumulative += *p;
            if cumulative > top_p {
                last_index = i;
                break;
            }
        }
        let probs = &probs[..=last_index];

        // sample from the truncated list
        let r = coin * cumulative;
        print!(
            "Sampling with top_p: {}, coin: {}, cumulative: {}\n",
            top_p, r, cumulative
        );
        cumulative = 0.0;
        for (index, p) in probs {
            cumulative += *p;
            if cumulative >= r {
                return *index;
            }
        }

        return probs.last().unwrap().0; // fallback to last index if no match found
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampler_new() {
        let sampler = Sampler::new(32, 1.0, Some(0.9), 42);
        assert_eq!(1.0, sampler.temperature);
        assert_eq!(Some(0.9), sampler.top_p);
        assert_eq!(42, sampler.rng_state);
        assert_eq!(32, sampler.logits.capacity());
    }

    #[test]
    fn test_argmax() {
        let probabilities = vec![0.1, 0.5, 0.1, 0.3];
        let index = Sampler::argmax(&probabilities);
        assert_eq!(1, index);

        let empty: Vec<f32> = vec![];
        let index_empty = Sampler::argmax(&empty);
        assert_eq!(0, index_empty);
    }

    #[test]
    fn test_sample_mult() {
        let probabilities = vec![0.1, 0.2, 0.3, 0.4];

        // should fall into the first bucket
        let index = Sampler::sample_mult(&probabilities, 0.05);
        assert_eq!(0, index);

        // should fall into the second bucket
        let index = Sampler::sample_mult(&probabilities, 0.25);
        assert_eq!(1, index);

        // should fall into the third bucket
        let index = Sampler::sample_mult(&probabilities, 0.6);
        assert_eq!(2, index);

        // should fall into the fourth bucket
        let index = Sampler::sample_mult(&probabilities, 0.75);
        assert_eq!(3, index);
    }

    #[test]
    fn test_sample_top_p() {
        let probabilities = vec![
            0.07, 0.03, 0.17, 0.13, 0.14, 0.16, 0.155, 0.1, 0.01, 0.015, 0.02,
        ];
        let top_p = 0.8;
        // The final probs are: [0.17, 0.16, 0.155, 0.14, 0.13, 0.1]
        // final cumulative is: 0.855

        // should pick the token with prob 0.17
        let index = Sampler::sample_top_p(&probabilities, top_p, 0.15);
        assert_eq!(2, index);

        // should pick the token with prob 0.16
        let index = Sampler::sample_top_p(&probabilities, top_p, 0.25);
        assert_eq!(5, index);

        // should pick the token with prob 0.1
        let index = Sampler::sample_top_p(&probabilities, top_p, 0.9);
        assert_eq!(7, index);
    }

    #[test]
    fn test_sample_zero_temperature() {
        let mut sampler = Sampler::new(16, 0.0, None, 42);
        let logits = vec![0.1, 0.1, 0.4, 0.3, 0.1];
        let token = sampler.sample(&logits);
        assert_eq!(2, token);
    }

    #[test]
    fn test_sample_apply_temperature() {
        let logits = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let mut sampler = Sampler::new(5, 2.0, None, 42);
        sampler.sample(&logits);

        assert!((sampler.logits[0] - 0.01165623096).abs() < 1e-6);
        assert!((sampler.logits[1] - 0.0316849208).abs() < 1e-6);
        assert!((sampler.logits[2] - 0.08612854441).abs() < 1e-6);
        assert!((sampler.logits[3] - 0.2341216573).abs() < 1e-6);
        assert!((sampler.logits[4] - 0.6364086465).abs() < 1e-6);
    }

    #[test]
    fn test_sample_with_sample_mult() {
        let logits = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let mut sampler = Sampler::new(5, 2.0, None, 12345);

        // based on the values in test_sample_apply_temperature and test_random_f32, it falls into the last bucket
        let token = sampler.sample(&logits);
        assert_eq!(4, token);
    }

    #[test]
    fn test_sample_with_sample_top_p() {
        let logits = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        let mut sampler = Sampler::new(5, 2.0, Some(0.8), 12345);

        // based on the values in test_sample_apply_temperature and test_random_f32, it picks 10.0
        let token = sampler.sample(&logits);
        assert_eq!(4, token);
    }

    #[test]
    fn test_random_f32() {
        let mut sampler = Sampler::new(16, 1.0, None, 12345);
        assert!((sampler.random_f32() - 0.595092).abs() < 1e-6);
        assert!((sampler.random_f32() - 0.753154).abs() < 1e-6);
        assert!((sampler.random_f32() - 0.076566).abs() < 1e-6);
        assert!((sampler.random_f32() - 0.736076).abs() < 1e-6);
        assert!((sampler.random_f32() - 0.119520).abs() < 1e-6);
        assert!((sampler.random_f32() - 0.210562).abs() < 1e-6);
    }
}
