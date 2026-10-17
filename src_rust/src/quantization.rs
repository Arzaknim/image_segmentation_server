#![allow(dead_code)]

const INT4_MIN: i8 = -7;
const INT4_MAX: i8 = 7;

#[derive(Debug, Clone, PartialEq)]
pub struct Int4QuantizedTensor {
    pub packed_nibbles: Vec<u8>,
    pub scale: f32,
    pub element_count: usize,
}

/// Quantizes a slice of 32-bit floating-point weights into symmetric signed INT4 (`[-7, 7]`),
/// packing two 4-bit values into each `u8` byte (low nibble = even index, high nibble = odd index).
///
/// This function is intentionally left unused by the main FP32 inference pipeline.
pub fn quantize_to_int4(weights: &[f32]) -> Int4QuantizedTensor {
    if weights.is_empty() {
        return Int4QuantizedTensor {
            packed_nibbles: Vec::new(),
            scale: 1.0,
            element_count: 0,
        };
    }

    let max_abs = weights
        .iter()
        .fold(0.0f32, |acc, &v| acc.max(v.abs()));
    let scale = if max_abs > 0.0 {
        max_abs / (INT4_MAX as f32)
    } else {
        1.0
    };

    let packed_len = weights.len().div_ceil(2);
    let mut packed_nibbles = Vec::with_capacity(packed_len);

    for pair in weights.chunks(2) {
        let low = quantize_scalar_int4(pair[0], scale);
        let high = if pair.len() > 1 {
            quantize_scalar_int4(pair[1], scale)
        } else {
            0
        };
        packed_nibbles.push(pack_int4_pair(low, high));
    }

    Int4QuantizedTensor {
        packed_nibbles,
        scale,
        element_count: weights.len(),
    }
}

/// Dequantizes a packed `Int4QuantizedTensor` back into 32-bit floats.
pub fn dequantize_from_int4(quantized: &Int4QuantizedTensor) -> Vec<f32> {
    let mut output = Vec::with_capacity(quantized.element_count);

    for &byte in &quantized.packed_nibbles {
        let (low, high) = unpack_int4_pair(byte);
        if output.len() < quantized.element_count {
            output.push(low as f32 * quantized.scale);
        }
        if output.len() < quantized.element_count {
            output.push(high as f32 * quantized.scale);
        }
    }

    output
}

#[inline]
fn quantize_scalar_int4(value: f32, scale: f32) -> i8 {
    let scaled = (value / scale).round();
    (scaled as i32).clamp(INT4_MIN as i32, INT4_MAX as i32) as i8
}

#[inline]
fn pack_int4_pair(low: i8, high: i8) -> u8 {
    let low_nibble = (low as u8) & 0x0F;
    let high_nibble = ((high as u8) & 0x0F) << 4;
    low_nibble | high_nibble
}

#[inline]
fn unpack_int4_pair(packed: u8) -> (i8, i8) {
    let low = sign_extend_4bit(packed & 0x0F);
    let high = sign_extend_4bit((packed >> 4) & 0x0F);
    (low, high)
}

#[inline]
fn sign_extend_4bit(nibble: u8) -> i8 {
    let shifted = (nibble << 4) as i8;
    shifted >> 4
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_signed_int4_values() {
        let weights = vec![-1.4, -0.6, 0.0, 0.4, 1.4];
        let quantized = quantize_to_int4(&weights);
        assert_eq!(quantized.packed_nibbles.len(), 3);
        assert_eq!(quantized.element_count, 5);

        let restored = dequantize_from_int4(&quantized);
        assert_eq!(restored.len(), weights.len());
        for (orig, deq) in weights.iter().zip(restored.iter()) {
            assert!((orig - deq).abs() <= 0.05, "orig={orig}, deq={deq}");
        }
    }
}

