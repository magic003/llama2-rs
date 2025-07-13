pub struct Sampler {
    vocab_size: u32,
    temperature: f32,
    top_p: Option<f32>,
    rng_seed: u64,
}

impl Sampler {
    pub fn new(vocab_size: u32, temperature: f32, top_p: Option<f32>, rng_seed: u64) -> Self {
        Sampler {
            vocab_size,
            temperature,
            top_p,
            rng_seed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sampler_new() {
        let sampler = Sampler::new(512, 1.0, Some(0.9), 42);
        assert_eq!(512, sampler.vocab_size);
        assert_eq!(1.0, sampler.temperature);
        assert_eq!(Some(0.9), sampler.top_p);
        assert_eq!(42, sampler.rng_seed);
    }
}
