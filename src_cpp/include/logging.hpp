#pragma once

#include <string>
#include <vector>
#include <filesystem>
#include <utility>
#include "types.hpp"

namespace segmentation {

std::pair<std::string, std::string> get_utc_timestamp();

std::filesystem::path resolve_log_dir(const std::string& custom_log_dir);

struct ImageInferenceLogEntry {
    std::string image_filename;
    std::string image_path;
    std::string output_mask_path;
    size_t width = 0;
    size_t height = 0;
    double total_inference_time_secs = 0.0;
    double pure_onnx_forward_secs = 0.0;
    size_t tile_count = 0;
    double avg_tile_forward_ms = 0.0;
    double pre_post_processing_secs = 0.0;
};

struct RunSummaryMetrics {
    size_t images_processed = 0;
    size_t total_tiles_computed = 0;
    double total_inference_time_secs = 0.0;
    double avg_inference_time_per_image_secs = 0.0;
    double pure_onnx_forward_secs = 0.0;
    double avg_tile_forward_ms = 0.0;
    double total_elapsed_secs = 0.0;
    std::string results_saved_to;
};

struct SingleRunInferenceLog {
    std::string timestamp;
    std::string engine = "cpp";
    std::string model_path;
    std::string input_path;
    std::string output_path;
    TilingConfig tiling;
    std::vector<ImageInferenceLogEntry> images;
    RunSummaryMetrics summary;

    std::string to_json_string() const;
};

std::filesystem::path save_run_log(
    const std::filesystem::path& log_dir,
    const SingleRunInferenceLog& log,
    const std::string& time_slug
);

} // namespace segmentation
