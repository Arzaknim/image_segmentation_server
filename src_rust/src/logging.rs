use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use crate::inference::TilingConfig;

const DEFAULT_LOG_DIRS: &[&str] = &["../inference_logs", "inference_logs"];

pub fn resolve_log_dir(custom_log_dir: Option<&Path>) -> PathBuf {
    if let Some(p) = custom_log_dir {
        return p.to_path_buf();
    }
    for candidate in DEFAULT_LOG_DIRS {
        let p = Path::new(candidate);
        if p.exists() {
            return p.to_path_buf();
        }
    }
    PathBuf::from("inference_logs")
}

pub fn get_utc_timestamp() -> (String, String) {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    let sec = secs % 60;
    let total_mins = secs / 60;
    let min = total_mins % 60;
    let total_hours = total_mins / 60;
    let hour = total_hours % 24;
    let mut days = (total_hours / 24) as i64;

    let mut year = 1970i64;
    loop {
        let leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
        let days_in_year = if leap { 366 } else { 365 };
        if days < days_in_year {
            break;
        }
        days -= days_in_year;
        year += 1;
    }

    let leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
    let days_in_months = [
        31, if leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31,
    ];
    let mut month = 1usize;
    for &dim in &days_in_months {
        if days < dim {
            break;
        }
        days -= dim;
        month += 1;
    }
    let day = days + 1;

    let iso = format!("{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}Z");
    let slug = format!("{year:04}{month:02}{day:02}_{hour:02}{min:02}{sec:02}");
    (iso, slug)
}

fn escape_json(s: &str) -> String {
    let mut escaped = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            other => escaped.push(other),
        }
    }
    escaped
}

#[derive(Debug, Clone)]
pub struct IouMetricsLog {
    pub mean_iou: f64,
    pub per_class_iou: [Option<f64>; 5], // Water, Land, Road, Building, Vegetation
}

impl IouMetricsLog {
    pub fn to_json_snippet(&self, indent: &str) -> String {
        let class_names = [
            "Water",
            "Land (unpaved area)",
            "Road",
            "Building",
            "Vegetation",
        ];
        let mut s = format!(
            "{indent}\"iou\": {{\n{indent}  \"mean_iou\": {:.4},\n{indent}  \"per_class\": {{\n",
            self.mean_iou
        );
        for (i, name) in class_names.iter().enumerate() {
            let comma = if i + 1 < class_names.len() { "," } else { "" };
            match self.per_class_iou[i] {
                Some(val) if !val.is_nan() => {
                    s.push_str(&format!("{indent}    \"{}\": {:.4}{comma}\n", name, val));
                }
                _ => {
                    s.push_str(&format!("{indent}    \"{}\": null{comma}\n", name));
                }
            }
        }
        s.push_str(&format!("{indent}  }}\n{indent}}}"));
        s
    }
}

#[derive(Debug, Clone)]
pub struct ImageInferenceLogEntry {
    pub image_filename: String,
    pub image_path: String,
    pub output_mask_path: String,
    pub width: usize,
    pub height: usize,
    pub total_inference_time_secs: f64,
    pub pure_onnx_forward_secs: f64,
    pub tile_count: usize,
    pub avg_tile_forward_ms: f64,
    pub pre_post_processing_secs: f64,
    pub iou: Option<IouMetricsLog>,
}

#[derive(Debug, Clone)]
pub struct RunSummaryMetrics {
    pub images_processed: usize,
    pub total_tiles_computed: usize,
    pub total_inference_time_secs: f64,
    pub avg_inference_time_per_image_secs: f64,
    pub pure_onnx_forward_secs: f64,
    pub avg_tile_forward_ms: f64,
    pub total_elapsed_secs: f64,
    pub results_saved_to: String,
    pub iou: Option<IouMetricsLog>,
}

#[derive(Debug, Clone)]
pub struct SingleRunInferenceLog {
    pub timestamp: String,
    pub engine: String,
    pub model_path: String,
    pub input_path: String,
    pub output_path: String,
    pub tiling: TilingConfig,
    pub images: Vec<ImageInferenceLogEntry>,
    pub summary: RunSummaryMetrics,
}

impl SingleRunInferenceLog {
    pub fn to_json_string(&self) -> String {
        let mut images_json = String::new();
        for (i, img) in self.images.iter().enumerate() {
            if i > 0 {
                images_json.push_str(",\n");
            }
            let iou_str = match &img.iou {
                Some(iou) => format!(",\n{}", iou.to_json_snippet("      ")),
                None => String::new(),
            };
            images_json.push_str(&format!(
                "    {{\n      \"image_filename\": \"{}\",\n      \"image_path\": \"{}\",\n      \"output_mask_path\": \"{}\",\n      \"dimensions\": {{\n        \"width\": {},\n        \"height\": {}\n      }},\n      \"timing\": {{\n        \"total_inference_time_seconds\": {:.4},\n        \"total_inference_time_ms\": {:.1},\n        \"pure_onnx_forward_seconds\": {:.4},\n        \"tile_count\": {},\n        \"avg_tile_forward_ms\": {:.2},\n        \"pre_post_processing_seconds\": {:.4}\n      }}{}\n    }}",
                escape_json(&img.image_filename),
                escape_json(&img.image_path),
                escape_json(&img.output_mask_path),
                img.width,
                img.height,
                img.total_inference_time_secs,
                img.total_inference_time_secs * 1000.0,
                img.pure_onnx_forward_secs,
                img.tile_count,
                img.avg_tile_forward_ms,
                img.pre_post_processing_secs,
                iou_str
            ));
        }

        let summary_iou_str = match &self.summary.iou {
            Some(iou) => format!(",\n{}", iou.to_json_snippet("    ")),
            None => String::new(),
        };

        format!(
            "{{\n  \"timestamp\": \"{}\",\n  \"engine\": \"{}\",\n  \"model_path\": \"{}\",\n  \"input_path\": \"{}\",\n  \"output_path\": \"{}\",\n  \"tiling\": {{\n    \"tile_size\": {},\n    \"stride\": {}\n  }},\n  \"images\": [\n{}\n  ],\n  \"summary\": {{\n    \"images_processed\": {},\n    \"total_tiles_computed\": {},\n    \"total_inference_time_seconds\": {:.4},\n    \"avg_inference_time_per_image_seconds\": {:.4},\n    \"pure_onnx_forward_secs\": {:.4},\n    \"avg_tile_forward_ms\": {:.2},\n    \"total_elapsed_time_seconds\": {:.4},\n    \"results_saved_to\": \"{}\"{}\n  }}\n}}\n",
            self.timestamp,
            self.engine,
            escape_json(&self.model_path),
            escape_json(&self.input_path),
            escape_json(&self.output_path),
            self.tiling.tile_size,
            self.tiling.stride,
            images_json,
            self.summary.images_processed,
            self.summary.total_tiles_computed,
            self.summary.total_inference_time_secs,
            self.summary.avg_inference_time_per_image_secs,
            self.summary.pure_onnx_forward_secs,
            self.summary.avg_tile_forward_ms,
            self.summary.total_elapsed_secs,
            escape_json(&self.summary.results_saved_to),
            summary_iou_str
        )
    }
}

pub fn save_run_log(
    log_dir: &Path,
    log: &SingleRunInferenceLog,
    time_slug: &str,
) -> Result<PathBuf, std::io::Error> {
    fs::create_dir_all(log_dir)?;
    let filename = format!("{}_inference_{}.json", log.engine, time_slug);
    let log_path = log_dir.join(filename);
    let mut file = fs::File::create(&log_path)?;
    file.write_all(log.to_json_string().as_bytes())?;
    Ok(log_path)
}
