pub fn rmsnorm(dest: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
    let sum_of_squares: f32 = x.iter().map(|&v| v * v).sum();
    let rms = (sum_of_squares / x.len() as f32 + eps).sqrt();

    let normalized = x.iter().zip(weight.iter()).map(|(&v, &w)| v * w / rms);
    for (norm_elem, dest_elem) in normalized.zip(dest.iter_mut()) {
        *dest_elem = norm_elem;
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
}
