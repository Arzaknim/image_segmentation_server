use std::env;
use std::fs;
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

use axum::extract::{DefaultBodyLimit, Multipart, State};
use axum::http::header::CONTENT_TYPE;
use axum::http::HeaderName;
use axum::http::StatusCode;
use axum::response::Json;
use axum::routing::{get, post};
use axum::Router;
use base64::Engine;
use image::codecs::png::PngEncoder;
use image::{ColorType, GenericImageView, ImageEncoder};
use serde::Serialize;
use tower_http::cors::{Any, CorsLayer};

use dubai_segmentation_inference::inference::{
    predict_full_resolution_with_timing, render_class_mask_to_rgb, TilingConfig, CLASS_COLORS_RGB,
    CLASS_NAMES, NUM_CLASSES,
};
use dubai_segmentation_inference::logging::{
    get_utc_timestamp, resolve_log_dir, save_run_log, ImageInferenceLogEntry, RunSummaryMetrics,
    SingleRunInferenceLog,
};
use dubai_segmentation_inference::onnx_model::OnnxModel;

const DEFAULT_ONNX_CANDIDATES: &[&str] = &[
    "../w_onnx/efficientnet_b0_tiled.onnx",
    "w_onnx/efficientnet_b0_tiled.onnx",
    "../efficientnet_b0_tiled.onnx",
    "efficientnet_b0_tiled.onnx",
];

const EMBEDDED_INDEX_HTML: &str = include_str!("../../web/index.html");
const EMBEDDED_STYLE_CSS: &str = include_str!("../../web/style.css");
const EMBEDDED_APP_JS: &str = include_str!("../../web/app.js");

#[derive(Clone)]
struct AppState {
    model: Arc<OnnxModel>,
    model_path: PathBuf,
    log_dir: PathBuf,
}

#[derive(Serialize)]
struct ClassMetric {
    id: usize,
    name: String,
    hex_color: String,
    pixel_count: usize,
    percentage: f64,
}

#[derive(Serialize)]
struct TimingMetrics {
    total_inference_time_ms: f64,
    pure_onnx_forward_ms: f64,
    tile_count: usize,
    avg_tile_forward_ms: f64,
    pre_post_processing_ms: f64,
}

#[derive(Serialize)]
struct SegmentResponse {
    success: bool,
    filename: String,
    width: u32,
    height: u32,
    mask_png_base64: String,
    timing: TimingMetrics,
    classes: Vec<ClassMetric>,
    log_file: Option<String>,
}

#[derive(Serialize)]
struct ErrorResponse {
    success: bool,
    error: String,
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

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut model_path = resolve_default_model_path();
    let mut port: u16 = 3000;
    let mut host = "127.0.0.1".to_string();
    let mut log_dir_opt: Option<PathBuf> = None;

    if let Ok(p_str) = env::var("PORT") {
        if let Ok(p) = p_str.parse::<u16>() {
            port = p;
        }
    }

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--port" | "-p" => {
                if let Some(val) = args.next() {
                    port = val.parse::<u16>().expect("Invalid port number");
                }
            }
            "--host" => {
                if let Some(val) = args.next() {
                    host = val;
                }
            }
            "--model" | "-m" => {
                if let Some(val) = args.next() {
                    model_path = PathBuf::from(val);
                }
            }
            "--log-dir" | "-l" => {
                if let Some(val) = args.next() {
                    log_dir_opt = Some(PathBuf::from(val));
                }
            }
            "--help" | "-h" => {
                println!(
                    "Usage: dubai_segmentation_server [OPTIONS]\n\n\
                     Options:\n  \
                       -p, --port <PORT>      Server port [default: 3000]\n  \
                           --host <HOST>      Bind host [default: 127.0.0.1]\n  \
                       -m, --model <PATH>     Path to ONNX weights [default: ../w_onnx/efficientnet_b0_tiled.onnx]\n  \
                       -l, --log-dir <PATH>   Path to inference_logs directory [default: inference_logs/]\n  \
                       -h, --help             Print help"
                );
                return Ok(());
            }
            _ => {}
        }
    }

    let log_dir = resolve_log_dir(log_dir_opt.as_deref());

    println!("============================================================");
    println!(" Dubai Satellite Aerial Segmentation Web Server (Axum & ORT) ");
    println!("============================================================");
    println!("Model:      {}", model_path.display());
    println!("Log Dir:    {}", log_dir.display());

    let start_load = Instant::now();
    let model = Arc::new(OnnxModel::new(&model_path)?);
    println!(
        "ONNX model loaded successfully in {:.2}s.",
        start_load.elapsed().as_secs_f64()
    );

    let state = AppState {
        model,
        model_path,
        log_dir,
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/", get(serve_index))
        .route("/index.html", get(serve_index))
        .route("/style.css", get(serve_css))
        .route("/app.js", get(serve_js))
        .route("/health", get(health_check))
        .route("/api/segment", post(handle_segmentation))
        // 50 MB max upload limit for high-resolution aerial imagery
        .layer(DefaultBodyLimit::max(50 * 1024 * 1024))
        .layer(cors)
        .with_state(state);

    let addr: SocketAddr = format!("{}:{}", host, port).parse()?;
    println!("Server running on http://{}\n", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

async fn serve_index() -> ([(HeaderName, &'static str); 1], String) {
    let content = fs::read_to_string("web/index.html")
        .or_else(|_| fs::read_to_string("../web/index.html"))
        .unwrap_or_else(|_| EMBEDDED_INDEX_HTML.to_string());
    ([(CONTENT_TYPE, "text/html; charset=utf-8")], content)
}

async fn serve_css() -> ([(HeaderName, &'static str); 1], String) {
    let content = fs::read_to_string("web/style.css")
        .or_else(|_| fs::read_to_string("../web/style.css"))
        .unwrap_or_else(|_| EMBEDDED_STYLE_CSS.to_string());
    ([(CONTENT_TYPE, "text/css; charset=utf-8")], content)
}

async fn serve_js() -> ([(HeaderName, &'static str); 1], String) {
    let content = fs::read_to_string("web/app.js")
        .or_else(|_| fs::read_to_string("../web/app.js"))
        .unwrap_or_else(|_| EMBEDDED_APP_JS.to_string());
    ([(CONTENT_TYPE, "application/javascript; charset=utf-8")], content)
}

async fn health_check(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "status": "ok",
        "engine": "rust",
        "runtime": "onnxruntime",
        "framework": "axum",
        "model_path": state.model_path.to_string_lossy(),
        "num_classes": NUM_CLASSES
    }))
}

async fn handle_segmentation(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<SegmentResponse>, (StatusCode, Json<ErrorResponse>)> {
    let mut file_bytes: Option<Vec<u8>> = None;
    let mut filename = "uploaded_image.jpg".to_string();
    let mut tile_size = 512usize;
    let mut stride = 256usize;

    while let Ok(Some(field)) = multipart.next_field().await {
        let name = field.name().unwrap_or_default().to_string();
        if name == "image" || name == "file" {
            if let Some(orig_name) = field.file_name() {
                filename = orig_name.to_string();
            }
            match field.bytes().await {
                Ok(bytes) => file_bytes = Some(bytes.to_vec()),
                Err(err) => {
                    return Err((
                        StatusCode::BAD_REQUEST,
                        Json(ErrorResponse {
                            success: false,
                            error: format!("Failed to read image stream: {err}"),
                        }),
                    ))
                }
            }
        } else if name == "tile_size" {
            if let Ok(text) = field.text().await {
                if let Ok(val) = text.parse::<usize>() {
                    if val > 0 {
                        tile_size = val;
                    }
                }
            }
        } else if name == "stride" {
            if let Ok(text) = field.text().await {
                if let Ok(val) = text.parse::<usize>() {
                    if val > 0 {
                        stride = val;
                    }
                }
            }
        }
    }

    let bytes = file_bytes.ok_or_else(|| {
        (
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                success: false,
                error: "No image file provided in multipart request under 'image' or 'file'".into(),
            }),
        )
    })?;

    if stride > tile_size {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(ErrorResponse {
                success: false,
                error: "Stride cannot be greater than tile size".into(),
            }),
        ));
    }

    let model = state.model.clone();
    let model_path_str = state.model_path.to_string_lossy().to_string();
    let log_dir = state.log_dir.clone();
    let fname = filename.clone();

    let inference_result = tokio::task::spawn_blocking(move || {
        let img = image::load_from_memory(&bytes)
            .map_err(|e| format!("Failed to decode image: {e}"))?;
        let (width, height) = img.dimensions();
        let rgb_bytes = img.to_rgb8().into_raw();

        let tiling = TilingConfig { tile_size, stride };
        let (mask, timing) = predict_full_resolution_with_timing(
            model.as_ref(),
            &rgb_bytes,
            width as usize,
            height as usize,
            tiling,
        )
        .map_err(|e| format!("Inference error: {e}"))?;

        let color_mask_rgb = render_class_mask_to_rgb(&mask);

        let mut png_bytes = Vec::new();
        let encoder = PngEncoder::new(&mut png_bytes);
        encoder
            .write_image(&color_mask_rgb, width, height, ColorType::Rgb8.into())
            .map_err(|e| format!("Failed to encode output PNG: {e}"))?;

        let mask_png_base64 = format!(
            "data:image/png;base64,{}",
            base64::engine::general_purpose::STANDARD.encode(&png_bytes)
        );

        let mut class_counts = [0usize; NUM_CLASSES];
        for &cls in &mask {
            if (cls as usize) < NUM_CLASSES {
                class_counts[cls as usize] += 1;
            }
        }
        let total_pixels = mask.len().max(1) as f64;

        let classes = (0..NUM_CLASSES)
            .map(|i| {
                let cnt = class_counts[i];
                let color_rgb = CLASS_COLORS_RGB[i];
                ClassMetric {
                    id: i,
                    name: CLASS_NAMES[i].to_string(),
                    hex_color: format!(
                        "#{:02X}{:02X}{:02X}",
                        color_rgb[0], color_rgb[1], color_rgb[2]
                    ),
                    pixel_count: cnt,
                    percentage: (cnt as f64 / total_pixels) * 100.0,
                }
            })
            .collect();

        let total_secs = timing.total_inference_time.as_secs_f64();
        let fwd_secs = timing.model_forward_time.as_secs_f64();
        let pre_post_secs = timing
            .total_inference_time
            .saturating_sub(timing.model_forward_time)
            .as_secs_f64();

        // Save JSON run log
        let (iso_time, time_slug) = get_utc_timestamp();
        let image_entry = ImageInferenceLogEntry {
            image_filename: fname.clone(),
            image_path: format!("web_upload/{}", fname),
            output_mask_path: format!("web_mask_{}.png", time_slug),
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
            results_saved_to: "web_client".into(),
        };

        let run_log = SingleRunInferenceLog {
            timestamp: iso_time,
            engine: "rust_axum".to_string(),
            model_path: model_path_str,
            input_path: format!("web_upload/{}", fname),
            output_path: "base64_stream".into(),
            tiling,
            images: vec![image_entry],
            summary,
        };

        let log_file_str = save_run_log(&log_dir, &run_log, &time_slug)
            .ok()
            .map(|p| p.to_string_lossy().to_string());

        let timing_metrics = TimingMetrics {
            total_inference_time_ms: total_secs * 1000.0,
            pure_onnx_forward_ms: fwd_secs * 1000.0,
            tile_count: timing.tile_count,
            avg_tile_forward_ms: timing.avg_tile_forward_ms(),
            pre_post_processing_ms: pre_post_secs * 1000.0,
        };

        Ok::<_, String>((width, height, mask_png_base64, timing_metrics, classes, log_file_str))
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                success: false,
                error: format!("Internal worker task panicked: {e}"),
            }),
        )
    })?
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(ErrorResponse {
                success: false,
                error: e,
            }),
        )
    })?;

    let (width, height, mask_png_base64, timing, classes, log_file) = inference_result;

    Ok(Json(SegmentResponse {
        success: true,
        filename,
        width,
        height,
        mask_png_base64,
        timing,
        classes,
        log_file,
    }))
}
