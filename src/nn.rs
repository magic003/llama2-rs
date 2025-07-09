use std::sync::{Arc, mpsc};
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
    let x = Arc::new(x.to_vec());

    let (tx, rx) = mpsc::channel();
    for i in 0..m {
        let tx = tx.clone();
        let w_row = w[i * k..(i + 1) * k].to_vec();
        let x = Arc::clone(&x);
        thread::spawn(move || {
            let mut val = 0.0f32;
            for j in 0..k {
                val += w_row[j] * x[j];
            }
            tx.send((i, val)).expect("Failed to send value");
        });
    }

    drop(tx);

    for (i, val) in rx {
        dest[i] = val;
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
}
