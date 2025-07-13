use crate::nn;

pub struct Sampler {
    temperature: f32,
    top_p: Option<f32>,
    rng_seed: u64,
}

impl Sampler {
    pub fn new(temperature: f32, top_p: Option<f32>, rng_seed: u64) -> Self {
        Sampler {
            temperature,
            top_p,
            rng_seed,
        }
    }

    pub fn sample(&self, logits: &mut [f32]) -> u32 {
        if self.temperature == 0.0 {
            // take the token with the highest probability
            return Self::argmax(logits) as u32;
        }

        // apply temperature to logits
        logits.iter_mut().for_each(|logit| {
            *logit /= self.temperature;
        });
        // apply softmax to get the probabilities
        nn::softmax(logits);

        return 0;
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
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampler_new() {
        let sampler = Sampler::new(1.0, Some(0.9), 42);
        assert_eq!(1.0, sampler.temperature);
        assert_eq!(Some(0.9), sampler.top_p);
        assert_eq!(42, sampler.rng_seed);
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
    fn test_sample_zero_temperature() {
        let sampler = Sampler::new(0.0, None, 42);
        let mut logits = vec![0.1, 0.1, 0.4, 0.3, 0.1];
        let token = sampler.sample(&mut logits);
        assert_eq!(2, token);
    }

    #[test]
    fn test_sample_apply_temperature() {
        let mut logits = vec![2.0, 4.0, 6.0, 8.0, 10.0];
        Sampler::new(2.0, None, 42).sample(&mut logits);

        assert!((logits[0] - 0.01165623096).abs() < 1e-6);
        assert!((logits[1] - 0.0316849208).abs() < 1e-6);
        assert!((logits[2] - 0.08612854441).abs() < 1e-6);
        assert!((logits[3] - 0.2341216573).abs() < 1e-6);
        assert!((logits[4] - 0.6364086465).abs() < 1e-6);
    }
}
