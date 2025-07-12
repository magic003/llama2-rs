use std::cmp;
use std::thread;

pub fn rmsnorm(dest: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let sum_of_squares: f32 = x.iter().map(|&v| v * v).sum();
    let rms = (sum_of_squares / x.len() as f32 + eps).sqrt();

    let normalized = x.iter().zip(weight.iter()).map(|(&v, &w)| v * w / rms);
    for (norm_elem, dest_elem) in normalized.zip(dest.iter_mut()) {
        *dest_elem = norm_elem;
    }
}

pub fn matmul(dest: &mut [f32], x: &[f32], w: &[f32], m: usize, k: usize) {
    // W (m, k) * x (k, ) = dest (m, )
    thread::scope(|s| {
        let rows_per_chunk = cmp::max(m / thread::available_parallelism().unwrap().get(), 1);
        let chunks = dest.chunks_mut(rows_per_chunk);
        for (i, chunk) in chunks.enumerate() {
            s.spawn(move || {
                for (j, val) in chunk.iter_mut().enumerate() {
                    let row_start = (i * rows_per_chunk + j) * k;
                    let w_row = &w[row_start..row_start + k];
                    *val = w_row
                        .iter()
                        .zip(x.iter())
                        .map(|(&w_val, &x_val)| w_val * x_val)
                        .sum::<f32>();
                }
            });
        }
    });
}

pub fn softmax(logits: &mut [f32]) {
    // find max value (for numerical stability)
    let max = logits
        .iter()
        .cloned()
        .reduce(f32::max)
        .expect("logits must not be empty");
    println!("max: {}", max);
    let sum: f32 = logits
        .iter_mut()
        .map(|logit| {
            *logit = (*logit - max).exp();
            *logit
        })
        .sum();
    println!("sum: {}", sum);
    println!("logits: {:?}", logits);

    for logit in logits.iter_mut() {
        *logit /= sum;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rmsnorm() {
        let x = vec![1.0, 2.0, 3.0];
        let weight = vec![0.2, 0.4, 0.5];
        let mut dest = vec![0.0; x.len()];
        let eps = 1e-6;

        rmsnorm(&mut dest, &x, &weight, eps);

        assert!((dest[0] - 0.09258200998).abs() < 1e-6);
        assert!((dest[1] - 0.3703280399).abs() < 1e-6);
        assert!((dest[2] - 0.6943650748).abs() < 1e-6);
    }

    #[test]
    fn test_matmul() {
        let x = vec![1.0, 2.0, 3.0];
        let w = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let m = 2;
        let k = 3;
        let mut dest = vec![0.0; m];

        matmul(&mut dest, &x, &w, m, k);

        assert_eq!(14.0, dest[0]);
        assert_eq!(32.0, dest[1]);
    }

    #[test]
    fn test_softmax() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        softmax(&mut logits);

        assert!((logits[0] - 0.01165623096).abs() < 1e-6);
        assert!((logits[1] - 0.0316849208).abs() < 1e-6);
        assert!((logits[2] - 0.08612854441).abs() < 1e-6);
        assert!((logits[3] - 0.2341216573).abs() < 1e-6);
        assert!((logits[4] - 0.6364086465).abs() < 1e-6);
    }
}
