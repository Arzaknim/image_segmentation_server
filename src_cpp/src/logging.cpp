#include "logging.hpp"
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <ctime>
#include <algorithm>

namespace segmentation {

static const std::vector<std::string> DEFAULT_LOG_DIRS = {
    "../inference_logs",
    "inference_logs"
};

std::string escape_json(const std::string& s) {
    std::string escaped;
    escaped.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '"':  escaped += "\\\""; break;
            case '\\': escaped += "\\\\"; break;
            case '\n': escaped += "\\n";  break;
            case '\r': escaped += "\\r";  break;
            case '\t': escaped += "\\t";  break;
            default:   escaped += c;      break;
        }
    }
    return escaped;
}

std::filesystem::path resolve_log_dir(const std::string& custom_log_dir) {
    if (!custom_log_dir.empty()) {
        return std::filesystem::path(custom_log_dir);
    }
    for (const auto& candidate : DEFAULT_LOG_DIRS) {
        if (std::filesystem::exists(candidate)) {
            return std::filesystem::path(candidate);
        }
    }
    return std::filesystem::path("inference_logs");
}

std::pair<std::string, std::string> get_utc_timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf{};
    gmtime_r(&now_c, &tm_buf);

    std::ostringstream iso_oss;
    iso_oss << std::setfill('0')
            << std::setw(4) << (tm_buf.tm_year + 1900) << "-"
            << std::setw(2) << (tm_buf.tm_mon + 1) << "-"
            << std::setw(2) << tm_buf.tm_mday << "T"
            << std::setw(2) << tm_buf.tm_hour << ":"
            << std::setw(2) << tm_buf.tm_min << ":"
            << std::setw(2) << tm_buf.tm_sec << "Z";

    std::ostringstream slug_oss;
    slug_oss << std::setfill('0')
             << std::setw(4) << (tm_buf.tm_year + 1900)
             << std::setw(2) << (tm_buf.tm_mon + 1)
             << std::setw(2) << tm_buf.tm_mday << "_"
             << std::setw(2) << tm_buf.tm_hour
             << std::setw(2) << tm_buf.tm_min
             << std::setw(2) << tm_buf.tm_sec;

    return {iso_oss.str(), slug_oss.str()};
}

std::string SingleRunInferenceLog::to_json_string() const {
    std::ostringstream oss;
    oss << "{\n"
        << "  \"timestamp\": \"" << timestamp << "\",\n"
        << "  \"engine\": \"" << engine << "\",\n"
        << "  \"model_path\": \"" << escape_json(model_path) << "\",\n"
        << "  \"input_path\": \"" << escape_json(input_path) << "\",\n"
        << "  \"output_path\": \"" << escape_json(output_path) << "\",\n"
        << "  \"tiling\": {\n"
        << "    \"tile_size\": " << tiling.tile_size << ",\n"
        << "    \"stride\": " << tiling.stride << "\n"
        << "  },\n"
        << "  \"images\": [\n";

    for (size_t i = 0; i < images.size(); ++i) {
        const auto& img = images[i];
        if (i > 0) oss << ",\n";
        oss << "    {\n"
            << "      \"image_filename\": \"" << escape_json(img.image_filename) << "\",\n"
            << "      \"image_path\": \"" << escape_json(img.image_path) << "\",\n"
            << "      \"output_mask_path\": \"" << escape_json(img.output_mask_path) << "\",\n"
            << "      \"dimensions\": {\n"
            << "        \"width\": " << img.width << ",\n"
            << "        \"height\": " << img.height << "\n"
            << "      },\n"
            << "      \"timing\": {\n"
            << "        \"total_inference_time_seconds\": " << std::fixed << std::setprecision(4) << img.total_inference_time_secs << ",\n"
            << "        \"total_inference_time_ms\": " << std::fixed << std::setprecision(1) << (img.total_inference_time_secs * 1000.0) << ",\n"
            << "        \"pure_onnx_forward_seconds\": " << std::fixed << std::setprecision(4) << img.pure_onnx_forward_secs << ",\n"
            << "        \"tile_count\": " << img.tile_count << ",\n"
            << "        \"avg_tile_forward_ms\": " << std::fixed << std::setprecision(2) << img.avg_tile_forward_ms << ",\n"
            << "        \"pre_post_processing_seconds\": " << std::fixed << std::setprecision(4) << img.pre_post_processing_secs << "\n"
            << "      }\n"
            << "    }";
    }

    oss << "\n  ],\n"
        << "  \"summary\": {\n"
        << "    \"images_processed\": " << summary.images_processed << ",\n"
        << "    \"total_tiles_computed\": " << summary.total_tiles_computed << ",\n"
        << "    \"total_inference_time_seconds\": " << std::fixed << std::setprecision(4) << summary.total_inference_time_secs << ",\n"
        << "    \"avg_inference_time_per_image_seconds\": " << std::fixed << std::setprecision(4) << summary.avg_inference_time_per_image_secs << ",\n"
        << "    \"pure_onnx_forward_seconds\": " << std::fixed << std::setprecision(4) << summary.pure_onnx_forward_secs << ",\n"
        << "    \"avg_tile_forward_ms\": " << std::fixed << std::setprecision(2) << summary.avg_tile_forward_ms << ",\n"
        << "    \"total_elapsed_time_seconds\": " << std::fixed << std::setprecision(4) << summary.total_elapsed_secs << ",\n"
        << "    \"results_saved_to\": \"" << escape_json(summary.results_saved_to) << "\"\n"
        << "  }\n"
        << "}\n";

    return oss.str();
}

std::filesystem::path save_run_log(
    const std::filesystem::path& log_dir,
    const SingleRunInferenceLog& log,
    const std::string& time_slug
) {
    std::filesystem::create_directories(log_dir);
    std::string filename = log.engine + "_inference_" + time_slug + ".json";
    std::filesystem::path log_path = log_dir / filename;

    std::ofstream ofs(log_path);
    if (!ofs.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + log_path.string());
    }
    ofs << log.to_json_string();
    return log_path;
}

} // namespace segmentation
