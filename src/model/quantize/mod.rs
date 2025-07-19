mod config;
mod weights;

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
    pub fn quantize(values: &[f32], group_size: u32) -> QuantizedTensor {
        let gs = group_size as usize;
        let mut q = Vec::with_capacity(values.len());
        let mut s = Vec::with_capacity(values.len() / gs);
        values.chunks_exact(gs).for_each(|group| {
            // find the max absolute value in the current group
            let wmax = group.iter().map(|&v| v.abs()).fold(0.0, f32::max);
            let scale = wmax / Self::Q_MAX;

            s.push(scale);
            group.iter().for_each(|v| {
                let quant_v = v / scale;
                let quantized = quant_v.round() as i8;
                q.push(quantized);
            });
        });

        QuantizedTensor {
            q,
            s,
            gs: group_size,
        }
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let group_size = 2;

        let quantized = QuantizedTensor::quantize(&values, group_size);

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

        let quantized = QuantizedTensor::quantize(&values, group_size);
        let dequantized = quantized.dequantize();

        assert_eq!(dequantized.len(), values.len());
        for (d, v) in dequantized.iter().zip(values.iter()) {
            assert!((d - v).abs() < 1e-2);
        }
    }
}
