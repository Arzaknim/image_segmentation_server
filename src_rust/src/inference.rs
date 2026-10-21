use std::error::Error;

use crate::model::EfficientNetB0Unet;
use crate::tensor::Tensor3d;

pub const NUM_CLASSES: usize = 6;
pub const IGNORE_INDEX: u8 = 5;
pub const DEFAULT_TILE_SIZE: usize = 512;
pub const DEFAULT_TILE_STRIDE: usize = 256;

const IMAGENET_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const IMAGENET_STD: [f32; 3] = [0.229, 0.224, 0.225];

pub const CLASS_NAMES: [&str; NUM_CLASSES] = [
    "Water",
    "Land (unpaved area)",
    "Road",
    "Building",
    "Vegetation",
    "Unlabeled",
];

pub const CLASS_COLORS_RGB: [[u8; 3]; NUM_CLASSES] = [
    [0xE2, 0xA9, 0x29], // 0: Water (#E2A929)
    [0x84, 0x29, 0xF6], // 1: Land (#8429F6)
    [0x6E, 0xC1, 0xE4], // 2: Road (#6EC1E4)
    [0x3C, 0x10, 0x98], // 3: Building (#3C1098)
    [0xFE, 0xDD, 0x3A], // 4: Vegetation (#FEDD3A)
    [0x9B, 0x9B, 0x9B], // 5: Unlabeled (#9B9B9B)
];

pub trait SegmentationPredictor {
    fn predict_tile(&self, tile: &Tensor3d) -> Result<Tensor3d, Box<dyn Error>>;
}

impl SegmentationPredictor for EfficientNetB0Unet {
    fn predict_tile(&self, tile: &Tensor3d) -> Result<Tensor3d, Box<dyn Error>> {
        Ok(self.forward(tile))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct TilingConfig {
    pub tile_size: usize,
    pub stride: usize,
}

impl Default for TilingConfig {
    fn default() -> Self {
        Self {
            tile_size: DEFAULT_TILE_SIZE,
            stride: DEFAULT_TILE_STRIDE,
        }
    }
}

pub fn tile_starts(length: usize, tile_size: usize, stride: usize) -> Vec<usize> {
    assert!(tile_size > 0 && stride > 0, "tile_size and stride must be positive");
    if length <= tile_size {
        return vec![0];
    }

    let mut starts: Vec<usize> = (0..=length - tile_size).step_by(stride).collect();
    let final_start = length - tile_size;
    if starts.last().copied() != Some(final_start) {
        starts.push(final_start);
    }
    starts
}

pub fn normalize_rgb_tile(
    rgb_bytes: &[u8],
    image_width: usize,
    image_height: usize,
    x_start: usize,
    y_start: usize,
    tile_width: usize,
    tile_height: usize,
) -> Tensor3d {
    assert_eq!(rgb_bytes.len(), image_width * image_height * 3);
    let mut tensor = Tensor3d::zeros(3, tile_height, tile_width);

    for ty in 0..tile_height {
        let iy = y_start + ty;
        for tx in 0..tile_width {
            let ix = x_start + tx;
            let pixel_offset = (iy * image_width + ix) * 3;
            for c in 0..3 {
                let raw = rgb_bytes[pixel_offset + c] as f32 / 255.0;
                let normalized = (raw - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
                tensor.set(c, ty, tx, normalized);
            }
        }
    }

    tensor
}

use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, Default)]
pub struct InferenceTiming {
    pub total_inference_time: Duration,
    pub model_forward_time: Duration,
    pub tile_count: usize,
}

impl InferenceTiming {
    pub fn avg_tile_forward_ms(&self) -> f64 {
        if self.tile_count > 0 {
            self.model_forward_time.as_secs_f64() * 1000.0 / self.tile_count as f64
        } else {
            0.0
        }
    }
}

pub fn predict_full_resolution<P: SegmentationPredictor>(
    model: &P,
    rgb_bytes: &[u8],
    width: usize,
    height: usize,
    config: TilingConfig,
) -> Result<Vec<u8>, Box<dyn Error>> {
    let (mask, _) = predict_full_resolution_with_timing(model, rgb_bytes, width, height, config)?;
    Ok(mask)
}

pub fn predict_full_resolution_with_timing<P: SegmentationPredictor>(
    model: &P,
    rgb_bytes: &[u8],
    width: usize,
    height: usize,
    config: TilingConfig,
) -> Result<(Vec<u8>, InferenceTiming), Box<dyn Error>> {
    let total_start = Instant::now();
    assert_eq!(rgb_bytes.len(), width * height * 3);

    let mut probability_canvas = Tensor3d::zeros(NUM_CLASSES, height, width);
    let mut count_canvas = vec![0.0f32; height * width];

    let y_starts = tile_starts(height, config.tile_size, config.stride);
    let x_starts = tile_starts(width, config.tile_size, config.stride);

    let mut model_forward_time = Duration::ZERO;
    let mut tile_count = 0usize;

    for &y_start in &y_starts {
        let tile_h = config.tile_size.min(height - y_start);
        for &x_start in &x_starts {
            let tile_w = config.tile_size.min(width - x_start);
            let tile_input = normalize_rgb_tile(
                rgb_bytes, width, height, x_start, y_start, tile_w, tile_h,
            );

            let padded_tile = if tile_w < config.tile_size || tile_h < config.tile_size {
                pad_tile(&tile_input, config.tile_size, config.tile_size)
            } else {
                tile_input
            };

            let fwd_start = Instant::now();
            let logits = model.predict_tile(&padded_tile)?;
            model_forward_time += fwd_start.elapsed();
            tile_count += 1;

            let probabilities = logits.softmax_along_channels();

            let valid_probabilities = if tile_w < config.tile_size || tile_h < config.tile_size {
                crop_tile(&probabilities, tile_h, tile_w)
            } else {
                probabilities
            };

            accumulate_tile_probabilities(
                &mut probability_canvas,
                &mut count_canvas,
                &valid_probabilities,
                x_start,
                y_start,
            );
        }
    }

    normalize_probability_canvas(&mut probability_canvas, &count_canvas);
    let mask = probability_canvas.argmax_along_channels();
    let timing = InferenceTiming {
        total_inference_time: total_start.elapsed(),
        model_forward_time,
        tile_count,
    };

    Ok((mask, timing))
}

pub fn render_class_mask_to_rgb(class_mask: &[u8]) -> Vec<u8> {
    let mut rgb = Vec::with_capacity(class_mask.len() * 3);
    for &class_id in class_mask {
        let color = CLASS_COLORS_RGB
            .get(class_id as usize)
            .copied()
            .unwrap_or(CLASS_COLORS_RGB[NUM_CLASSES - 1]);
        rgb.extend_from_slice(&color);
    }
    rgb
}

#[derive(Debug, Clone)]
pub struct ClassIouResult {
    pub per_class_iou: [f64; NUM_CLASSES - 1], // NaN if absent (union == 0)
    pub mean_iou: f64,
    pub intersections: [u64; NUM_CLASSES - 1],
    pub unions: [u64; NUM_CLASSES - 1],
}

pub fn decode_ground_truth_mask(
    raw_bytes: &[u8],
    width: usize,
    height: usize,
    channels: usize,
) -> Result<Vec<u8>, String> {
    let pixel_count = width * height;
    if channels == 1 {
        if raw_bytes.len() < pixel_count {
            return Err(format!(
                "Mask buffer size {} is smaller than expected pixel count {}",
                raw_bytes.len(),
                pixel_count
            ));
        }
        let mut mask = vec![0u8; pixel_count];
        for i in 0..pixel_count {
            let val = raw_bytes[i];
            mask[i] = if (val as usize) < NUM_CLASSES { val } else { IGNORE_INDEX };
        }
        return Ok(mask);
    }

    if channels >= 3 {
        let expected_len = pixel_count * channels;
        if raw_bytes.len() < expected_len {
            return Err(format!(
                "Mask RGB buffer size {} is smaller than expected {}",
                raw_bytes.len(),
                expected_len
            ));
        }
        let mut mask = vec![IGNORE_INDEX; pixel_count];
        for i in 0..pixel_count {
            let r = raw_bytes[i * channels];
            let g = raw_bytes[i * channels + 1];
            let b = raw_bytes[i * channels + 2];

            if r == 0 && g == 0 && b == 0 {
                mask[i] = IGNORE_INDEX;
                continue;
            }

            let mut matched = false;
            for (class_id, color) in CLASS_COLORS_RGB.iter().enumerate() {
                if r == color[0] && g == color[1] && b == color[2] {
                    mask[i] = class_id as u8;
                    matched = true;
                    break;
                }
            }
            if !matched {
                mask[i] = IGNORE_INDEX;
            }
        }
        return Ok(mask);
    }

    Err(format!("Unsupported mask channel count: {}", channels))
}

pub fn calculate_mask_iou(prediction: &[u8], target: &[u8]) -> Result<ClassIouResult, String> {
    if prediction.len() != target.len() {
        return Err(format!(
            "Prediction length {} does not match target length {}",
            prediction.len(),
            target.len()
        ));
    }

    let active_classes = NUM_CLASSES - 1;
    let mut intersections = [0u64; NUM_CLASSES - 1];
    let mut unions = [0u64; NUM_CLASSES - 1];

    for i in 0..prediction.len() {
        let t = target[i];
        if t == IGNORE_INDEX {
            continue;
        }
        let p = prediction[i];
        for c in 0..active_classes {
            let class_u8 = c as u8;
            let in_p = p == class_u8;
            let in_t = t == class_u8;
            if in_p && in_t {
                intersections[c] += 1;
            }
            if in_p || in_t {
                unions[c] += 1;
            }
        }
    }

    let mut per_class_iou = [f64::NAN; NUM_CLASSES - 1];
    let mut sum_iou = 0.0;
    let mut count_iou = 0usize;

    for c in 0..active_classes {
        if unions[c] > 0 {
            let iou = intersections[c] as f64 / unions[c] as f64;
            per_class_iou[c] = iou;
            sum_iou += iou;
            count_iou += 1;
        }
    }

    let mean_iou = if count_iou > 0 {
        sum_iou / count_iou as f64
    } else {
        0.0
    };

    Ok(ClassIouResult {
        per_class_iou,
        mean_iou,
        intersections,
        unions,
    })
}

fn pad_tile(tile: &Tensor3d, target_h: usize, target_w: usize) -> Tensor3d {
    let mut padded = Tensor3d::zeros(tile.channels, target_h, target_w);
    for c in 0..tile.channels {
        for y in 0..tile.height {
            for x in 0..tile.width {
                padded.set(c, y, x, tile.get(c, y, x));
            }
        }
    }
    padded
}

fn crop_tile(tile: &Tensor3d, crop_h: usize, crop_w: usize) -> Tensor3d {
    let mut cropped = Tensor3d::zeros(tile.channels, crop_h, crop_w);
    for c in 0..tile.channels {
        for y in 0..crop_h {
            for x in 0..crop_w {
                cropped.set(c, y, x, tile.get(c, y, x));
            }
        }
    }
    cropped
}

fn accumulate_tile_probabilities(
    probability_canvas: &mut Tensor3d,
    count_canvas: &mut [f32],
    tile_probabilities: &Tensor3d,
    x_start: usize,
    y_start: usize,
) {
    let full_width = probability_canvas.width;
    let tile_h = tile_probabilities.height;
    let tile_w = tile_probabilities.width;

    for ty in 0..tile_h {
        let dst_y = y_start + ty;
        for tx in 0..tile_w {
            let dst_x = x_start + tx;
            count_canvas[dst_y * full_width + dst_x] += 1.0;
        }
    }

    for c in 0..probability_canvas.channels {
        let src_plane = tile_probabilities.channel_slice(c);
        let dst_plane = probability_canvas.channel_slice_mut(c);
        for ty in 0..tile_h {
            let dst_row_start = (y_start + ty) * full_width + x_start;
            let src_row_start = ty * tile_w;
            for tx in 0..tile_w {
                dst_plane[dst_row_start + tx] += src_plane[src_row_start + tx];
            }
        }
    }
}

fn normalize_probability_canvas(probability_canvas: &mut Tensor3d, count_canvas: &[f32]) {
    let plane_size = probability_canvas.height * probability_canvas.width;
    for c in 0..probability_canvas.channels {
        let plane = probability_canvas.channel_slice_mut(c);
        for i in 0..plane_size {
            let count = count_canvas[i].max(1.0);
            plane[i] /= count;
        }
    }
}
