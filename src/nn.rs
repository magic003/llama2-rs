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
}
