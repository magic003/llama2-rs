use std::{cmp, thread};

mod config;
mod transformer;
mod weights;

pub use transformer::Transformer;

/// The quantized tensor. It has a list of values and the scale factors for each group.
#[derive(Debug)]
struct QuantizedTensor {
    /// quantized values
    q: Vec<i8>,
    /// scale factors for each group
    s: Vec<f32>,
    /// group size
    gs: u32,
}

impl QuantizedTensor {
    /// Creates a QuantizedTensor from f32 values.
    pub fn new(values: &[f32], group_size: u32) -> QuantizedTensor {
        let mut tensor = QuantizedTensor {
            q: Vec::with_capacity(values.len()),
            s: Vec::with_capacity(values.len() / group_size as usize),
            gs: group_size,
        };
        tensor.quantize(values);
        tensor
    }

    /// Quantizes the tensor values into the current instance.
    pub fn quantize(&mut self, values: &[f32]) {
        let gs = self.gs as usize;
        self.q.clear();
        self.s.clear();
        values.chunks_exact(gs).for_each(|group| {
            // find the max absolute value in the current group
            let wmax = group.iter().map(|&v| v.abs()).fold(0.0, f32::max);
            let scale = wmax / Self::Q_MAX;

            self.s.push(scale);
            group.iter().for_each(|v| {
                let quant_v = v / scale;
                let quantized = quant_v.round() as i8;
                self.q.push(quantized);
            });
        });
    }

    /// Dequantizes the tensor values into f32 values.
    pub fn dequantize(&self) -> Vec<f32> {
        self.q
            .iter()
            .enumerate()
            .map(|(i, q)| *q as f32 * self.s[i / self.gs as usize])
            .collect()
    }

    /// Max of the q8 value.
    const Q_MAX: f32 = 127.0;
}

/// Performs matrix multiplication of `w` (m * k) and `x` (k) and stores the result in `dest` (m). It uses
/// multi-threading to parallelize the computation.
fn matmul(dest: &mut [f32], x: &QuantizedTensor, w: &QuantizedTensor, m: usize, k: usize) {
    // W (m, k) * x (k, ) = dest (m, )
    thread::scope(|s| {
        let rows_per_chunk = cmp::max(m / thread::available_parallelism().unwrap().get(), 1);
        let chunks = dest.chunks_mut(rows_per_chunk);
        let group_size = x.gs as usize;
        for (i, chunk) in chunks.enumerate() {
            s.spawn(move || {
                for (j, val) in chunk.iter_mut().enumerate() {
                    let row_start = (i * rows_per_chunk + j) * k;
                    let mut res = 0.0f32;
                    let mut group_sum = 0i32;
                    for xi in (0..=k - group_size).step_by(group_size) {
                        for index_in_group in 0..group_size {
                            let w_val = w.q[row_start + xi + index_in_group] as i32;
                            let x_val = x.q[xi + index_in_group] as i32;
                            group_sum += w_val * x_val;
                        }
                        res += group_sum as f32
                            * w.s[(row_start + xi) / group_size]
                            * x.s[xi / group_size];
                        group_sum = 0;
                    }
                    *val = res;
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let group_size = 2;

        let quantized = QuantizedTensor::new(&values, group_size);

        assert_eq!(values.len(), quantized.q.len());
        assert_eq!(values.len() / group_size as usize, quantized.s.len());
        assert_eq!(group_size, quantized.gs);

        assert_eq!(64, quantized.q[0]);
        assert_eq!(127, quantized.q[1]);
        assert_eq!(95, quantized.q[2]);
        assert_eq!(127, quantized.q[3]);
        assert_eq!(106, quantized.q[4]);
    }

    #[test]
    fn test_dequantize() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let group_size = 2;

        let quantized = QuantizedTensor::new(&values, group_size);
        let dequantized = quantized.dequantize();

        assert_eq!(dequantized.len(), values.len());
        for (d, v) in dequantized.iter().zip(values.iter()) {
            assert!((d - v).abs() < 1e-2);
        }
    }

    #[test]
    fn test_matmul() {
        let x = QuantizedTensor::new(&vec![1.0, 2.0, 3.0], 3);
        let w = QuantizedTensor::new(&vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3);
        let m = 2;
        let k = 3;
        let mut dest = vec![0.0; m];

        matmul(&mut dest, &x, &w, m, k);

        assert_eq!(14.015871, dest[0]);
        assert_eq!(32.039307, dest[1]);
    }
}
