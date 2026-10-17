use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process;
use std::time::Instant;

use dubai_segmentation_inference::inference::{
    predict_full_resolution_with_timing, render_class_mask_to_rgb, InferenceTiming, TilingConfig,
    CLASS_NAMES, NUM_CLASSES,
};
use dubai_segmentation_inference::logging::{
    get_utc_timestamp, resolve_log_dir, save_run_log, ImageInferenceLogEntry, RunSummaryMetrics,
    SingleRunInferenceLog,
};
use dubai_segmentation_inference::onnx_model::OnnxModel;
use image::{ColorType, GenericImageView};

const DEFAULT_ONNX_CANDIDATES: &[&str] = &[
    "../w_onnx/efficientnet_b0_tiled.onnx",
    "w_onnx/efficientnet_b0_tiled.onnx",
    "../efficientnet_b0_tiled.onnx",
    "efficientnet_b0_tiled.onnx",
];

const DEFAULT_OUTPUT_DIRS: &[&str] = &["../preds_rust", "preds_rust", "../preds", "preds"];

const SUPPORTED_IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "bmp", "tif", "tiff"];

#[derive(Debug)]
struct CliConfig {
    model_path: PathBuf,
    input_path: PathBuf,
    output_path: Option<PathBuf>,
    log_dir: PathBuf,
    tiling: TilingConfig,
}

fn main() {
    if let Err(err) = run() {
        eprintln!("\nError: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let config = parse_cli_args()?;

    println!("============================================================");
    println!(" Dubai Satellite Aerial Image Segmentation (ONNX Runtime)   ");
    println!("============================================================");
    println!("Model:      {}", config.model_path.display());
    println!("Input:      {}", config.input_path.display());
    println!(
        "Tiling:     tile_size={}, stride={}",
        config.tiling.tile_size, config.tiling.stride
    );
    println!("Log Dir:    {}", config.log_dir.display());

    let start_load = Instant::now();
    let model = OnnxModel::new(&config.model_path)?;
    println!(
        "ONNX model loaded successfully in {:.2}s.\n",
        start_load.elapsed().as_secs_f64()
    );

    if config.input_path.is_dir() {
        process_directory(&model, &config)?;
    } else if config.input_path.is_file() {
        process_single_file(&model, &config)?;
    } else {
        return Err(format!(
            "Input path '{}' does not exist as a file or directory.",
            config.input_path.display()
        )
        .into());
    }

    Ok(())
}

fn process_single_file(
    model: &OnnxModel,
    config: &CliConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let out_file_path = resolve_single_output_path(&config.input_path, config.output_path.as_deref())?;

    if let Some(parent) = out_file_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let (width, height, predicted_mask, timing) = segment_image(model, &config.input_path, config.tiling)?;

    let color_mask_rgb = render_class_mask_to_rgb(&predicted_mask);
    image::save_buffer(
        &out_file_path,
        &color_mask_rgb,
        width,
        height,
        ColorType::Rgb8,
    )?;

    let (iso_time, time_slug) = get_utc_timestamp();
    let total_secs = timing.total_inference_time.as_secs_f64();
    let fwd_secs = timing.model_forward_time.as_secs_f64();
    let pre_post_secs = timing
        .total_inference_time
        .saturating_sub(timing.model_forward_time)
        .as_secs_f64();

    let image_entry = ImageInferenceLogEntry {
        image_filename: config
            .input_path
            .file_name()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_else(|| "image".to_string()),
        image_path: config.input_path.to_string_lossy().to_string(),
        output_mask_path: out_file_path.to_string_lossy().to_string(),
        width: width as usize,
        height: height as usize,
        total_inference_time_secs: total_secs,
        pure_onnx_forward_secs: fwd_secs,
        tile_count: timing.tile_count,
        avg_tile_forward_ms: timing.avg_tile_forward_ms(),
        pre_post_processing_secs: pre_post_secs,
    };

    let summary = RunSummaryMetrics {
        images_processed: 1,
        total_tiles_computed: timing.tile_count,
        total_inference_time_secs: total_secs,
        avg_inference_time_per_image_secs: total_secs,
        pure_onnx_forward_secs: fwd_secs,
        avg_tile_forward_ms: timing.avg_tile_forward_ms(),
        total_elapsed_secs: total_secs,
        results_saved_to: out_file_path.to_string_lossy().to_string(),
    };

    let run_log = SingleRunInferenceLog {
        timestamp: iso_time,
        engine: "rust".to_string(),
        model_path: config.model_path.to_string_lossy().to_string(),
        input_path: config.input_path.to_string_lossy().to_string(),
        output_path: out_file_path.to_string_lossy().to_string(),
        tiling: config.tiling,
        images: vec![image_entry],
        summary,
    };

    let log_path = save_run_log(&config.log_dir, &run_log, &time_slug)?;

    println!(
        "Successfully segmented '{}' ({}x{}):",
        config.input_path.display(),
        width,
        height
    );
    println!("------------------------------------------------------------");
    println!(" Timing Breakdown:");
    println!(
        "  - Total inference:      {:.3}s ({:.1} ms)",
        total_secs,
        total_secs * 1000.0
    );
    println!(
        "  - Pure ONNX forward:    {:.3}s across {} tiles (avg {:.1} ms/tile)",
        fwd_secs,
        timing.tile_count,
        timing.avg_tile_forward_ms()
    );
    println!("  - Pre/post processing:  {:.3}s", pre_post_secs);
    println!("------------------------------------------------------------");
    println!("Saved predicted mask to: {}", out_file_path.display());
    println!("Saved inference log to:  {}", log_path.display());
    println!("\nClass distribution:");
    print_class_distribution(&predicted_mask);

    Ok(())
}

fn process_directory(
    model: &OnnxModel,
    config: &CliConfig,
) -> Result<(), Box<dyn std::error::Error>> {
    let out_dir = resolve_directory_output_path(config.output_path.as_deref());
    fs::create_dir_all(&out_dir)?;

    let mut image_files = find_images_in_dir(&config.input_path)?;
    image_files.sort();

    if image_files.is_empty() {
        return Err(format!(
            "No supported images (.jpg, .png, etc.) found in '{}'.",
            config.input_path.display()
        )
        .into());
    }

    println!(
        "Found {} images in '{}'. Saving masks to '{}'...\n",
        image_files.len(),
        config.input_path.display(),
        out_dir.display()
    );

    let total_start = Instant::now();
    let total_images = image_files.len();
    let mut total_inference_time = std::time::Duration::ZERO;
    let mut total_forward_time = std::time::Duration::ZERO;
    let mut total_tiles = 0usize;
    let mut image_entries = Vec::with_capacity(total_images);

    for (idx, img_path) in image_files.iter().enumerate() {
        let stem = img_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");
        let mask_file_name = format!("{stem}_mask.png");
        let out_file_path = out_dir.join(mask_file_name);

        match segment_image(model, img_path, config.tiling) {
            Ok((width, height, mask, timing)) => {
                total_inference_time += timing.total_inference_time;
                total_forward_time += timing.model_forward_time;
                total_tiles += timing.tile_count;

                let color_rgb = render_class_mask_to_rgb(&mask);
                image::save_buffer(
                    &out_file_path,
                    &color_rgb,
                    width,
                    height,
                    ColorType::Rgb8,
                )?;

                let total_secs = timing.total_inference_time.as_secs_f64();
                let fwd_secs = timing.model_forward_time.as_secs_f64();
                let pre_post_secs = timing
                    .total_inference_time
                    .saturating_sub(timing.model_forward_time)
                    .as_secs_f64();

                let entry = ImageInferenceLogEntry {
                    image_filename: img_path.file_name().unwrap().to_string_lossy().to_string(),
                    image_path: img_path.to_string_lossy().to_string(),
                    output_mask_path: out_file_path.to_string_lossy().to_string(),
                    width: width as usize,
                    height: height as usize,
                    total_inference_time_secs: total_secs,
                    pure_onnx_forward_secs: fwd_secs,
                    tile_count: timing.tile_count,
                    avg_tile_forward_ms: timing.avg_tile_forward_ms(),
                    pre_post_processing_secs: pre_post_secs,
                };
                image_entries.push(entry);

                println!(
                    "[{}/{}] {} ({}x{}) -> {} | Inference: {:.3}s ({} tiles, {:.1}ms/tile)",
                    idx + 1,
                    total_images,
                    img_path.file_name().unwrap().to_string_lossy(),
                    width,
                    height,
                    out_file_path.file_name().unwrap().to_string_lossy(),
                    timing.total_inference_time.as_secs_f64(),
                    timing.tile_count,
                    timing.avg_tile_forward_ms()
                );
            }
            Err(err) => {
                eprintln!(
                    "[{}/{}] Error processing '{}': {err}",
                    idx + 1,
                    total_images,
                    img_path.display()
                );
            }
        }
    }

    let elapsed = total_start.elapsed();
    let (global_iso, global_slug) = get_utc_timestamp();
    let total_inf_secs = total_inference_time.as_secs_f64();
    let total_fwd_secs = total_forward_time.as_secs_f64();
    let avg_tile_ms = if total_tiles > 0 {
        total_fwd_secs * 1000.0 / total_tiles as f64
    } else {
        0.0
    };

    let summary = RunSummaryMetrics {
        images_processed: image_entries.len(),
        total_tiles_computed: total_tiles,
        total_inference_time_secs: total_inf_secs,
        avg_inference_time_per_image_secs: if !image_entries.is_empty() {
            total_inf_secs / image_entries.len() as f64
        } else {
            0.0
        },
        pure_onnx_forward_secs: total_fwd_secs,
        avg_tile_forward_ms: avg_tile_ms,
        total_elapsed_secs: elapsed.as_secs_f64(),
        results_saved_to: out_dir.to_string_lossy().to_string(),
    };

    let run_log = SingleRunInferenceLog {
        timestamp: global_iso,
        engine: "rust".to_string(),
        model_path: config.model_path.to_string_lossy().to_string(),
        input_path: config.input_path.to_string_lossy().to_string(),
        output_path: out_dir.to_string_lossy().to_string(),
        tiling: config.tiling,
        images: image_entries,
        summary,
    };

    let log_path = save_run_log(&config.log_dir, &run_log, &global_slug)?;

    println!("\n============================================================");
    println!(" Batch Summary:");
    println!("  - Images processed:     {}", total_images);
    println!("  - Total tiles computed: {}", total_tiles);
    println!(
        "  - Total inference time: {:.2}s (avg {:.2}s/image)",
        total_inf_secs,
        if total_images > 0 {
            total_inf_secs / total_images as f64
        } else {
            0.0
        }
    );
    println!(
        "  - Pure ONNX forward:    {:.2}s (avg {:.1}ms/tile)",
        total_fwd_secs, avg_tile_ms
    );
    println!("  - Total elapsed time:   {:.2}s", elapsed.as_secs_f64());
    println!("  - Results saved to:     {}", out_dir.display());
    println!("  - Inference log saved:  {}", log_path.display());
    println!("============================================================");

    Ok(())
}

fn segment_image(
    model: &OnnxModel,
    image_path: &Path,
    tiling: TilingConfig,
) -> Result<(u32, u32, Vec<u8>, InferenceTiming), Box<dyn std::error::Error>> {
    let img = image::open(image_path)?;
    let (width, height) = img.dimensions();
    let rgb_bytes = img.to_rgb8().into_raw();

    let (predicted_mask, timing) = predict_full_resolution_with_timing(
        model,
        &rgb_bytes,
        width as usize,
        height as usize,
        tiling,
    )?;

    Ok((width, height, predicted_mask, timing))
}

fn find_images_in_dir(dir: &Path) -> Result<Vec<PathBuf>, std::io::Error> {
    let mut files = Vec::new();
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
                if SUPPORTED_IMAGE_EXTENSIONS.contains(&ext.to_lowercase().as_str()) {
                    files.push(path);
                }
            }
        }
    }
    Ok(files)
}

fn resolve_single_output_path(
    input_file: &Path,
    specified_output: Option<&Path>,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    if let Some(out) = specified_output {
        if out.is_dir() || out.to_string_lossy().ends_with('/') {
            let stem = input_file
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("output");
            Ok(out.join(format!("{stem}_mask.png")))
        } else {
            Ok(out.to_path_buf())
        }
    } else {
        let default_dir = resolve_directory_output_path(None);
        let stem = input_file
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");
        Ok(default_dir.join(format!("{stem}_mask.png")))
    }
}

fn resolve_directory_output_path(specified_output: Option<&Path>) -> PathBuf {
    if let Some(out) = specified_output {
        out.to_path_buf()
    } else {
        for candidate in DEFAULT_OUTPUT_DIRS {
            let p = Path::new(candidate);
            if p.exists() {
                return p.to_path_buf();
            }
        }
        PathBuf::from("preds")
    }
}

fn resolve_default_model_path() -> PathBuf {
    for candidate in DEFAULT_ONNX_CANDIDATES {
        let p = Path::new(candidate);
        if p.exists() {
            return p.to_path_buf();
        }
    }
    PathBuf::from(DEFAULT_ONNX_CANDIDATES[0])
}

fn parse_cli_args() -> Result<CliConfig, String> {
    let mut model_path = resolve_default_model_path();
    let mut input_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;
    let mut log_dir: Option<PathBuf> = None;
    let mut tiling = TilingConfig::default();

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input" | "-i" | "--image" | "--input-dir" => {
                let val = args
                    .next()
                    .ok_or_else(|| "Missing value for --input (-i)".to_string())?;
                input_path = Some(PathBuf::from(val));
            }
            "--output" | "-o" => {
                let val = args
                    .next()
                    .ok_or_else(|| "Missing value for --output (-o)".to_string())?;
                output_path = Some(PathBuf::from(val));
            }
            "--model" | "-m" | "--weights" | "-w" => {
                let val = args
                    .next()
                    .ok_or_else(|| "Missing value for --model (-m)".to_string())?;
                model_path = PathBuf::from(val);
            }
            "--log-dir" | "-l" => {
                let val = args
                    .next()
                    .ok_or_else(|| "Missing value for --log-dir".to_string())?;
                log_dir = Some(PathBuf::from(val));
            }
            "--tile-size" => {
                let val = args
                    .next()
                    .ok_or_else(|| "Missing value for --tile-size".to_string())?;
                tiling.tile_size = val
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid --tile-size: '{val}'"))?;
            }
            "--stride" => {
                let val = args
                    .next()
                    .ok_or_else(|| "Missing value for --stride".to_string())?;
                tiling.stride = val
                    .parse::<usize>()
                    .map_err(|_| format!("Invalid --stride: '{val}'"))?;
            }
            "--help" | "-h" => {
                print_help();
                process::exit(0);
            }
            other => {
                if !other.starts_with('-') && input_path.is_none() {
                    input_path = Some(PathBuf::from(other));
                } else {
                    return Err(format!(
                        "Unrecognized argument '{other}'. Run with --help for usage instructions."
                    ));
                }
            }
        }
    }

    let input_path = input_path.ok_or_else(|| {
        print_help();
        "\nPlease provide an input image file or directory using `--input <PATH>`.".to_string()
    })?;

    if tiling.tile_size == 0 || tiling.stride == 0 || tiling.stride > tiling.tile_size {
        return Err("tile-size and stride must be > 0 with stride <= tile-size".to_string());
    }

    let log_dir = resolve_log_dir(log_dir.as_deref());

    Ok(CliConfig {
        model_path,
        input_path,
        output_path,
        log_dir,
        tiling,
    })
}

fn print_class_distribution(mask: &[u8]) {
    let mut counts = [0usize; NUM_CLASSES];
    for &cls in mask {
        if (cls as usize) < NUM_CLASSES {
            counts[cls as usize] += 1;
        }
    }
    let total = mask.len().max(1) as f64;
    for (class_id, &count) in counts.iter().enumerate() {
        let percentage = (count as f64 / total) * 100.0;
        println!(
            "  - {:<22}: {:>8} pixels ({:5.2}%)",
            CLASS_NAMES[class_id], count, percentage
        );
    }
}

fn print_help() {
    println!(
        "Usage: dubai_segmentation_inference [OPTIONS] --input <FILE_OR_DIR>\n\n\
         Arguments / Options:\n  \
           -i, --input <PATH>       Path to a single image file OR a directory containing images [required]\n  \
           -o, --output <PATH>      Output PNG mask path (single file) or directory (batch mode) [default: preds_rust/]\n  \
           -m, --model <PATH>       Path to ONNX weights [default: ../w_onnx/efficientnet_b0_tiled.onnx]\n  \
           -l, --log-dir <PATH>     Directory to save JSON inference logs [default: inference_logs/]\n  \
               --tile-size <INT>    Sliding-window tile size [default: 512]\n  \
               --stride <INT>       Sliding-window stride [default: 256]\n  \
           -h, --help               Print this help message\n\n\
         Examples:\n  \
           # Single file inference:\n  \
           cargo run --release -- -i ../Dataset_Flattened/images/Tile1_image_part_001.jpg -o ../preds_rust/tile1_mask.png\n\n  \
           # Batch directory inference:\n  \
           cargo run --release -- -i ../Dataset_Flattened/images/ -o ../preds_rust/"
    );
}
