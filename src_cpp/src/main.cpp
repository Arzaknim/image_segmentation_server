#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <filesystem>
#include <chrono>
#include <algorithm>
#include <cctype>

#include "types.hpp"
#include "onnx_model.hpp"
#include "inference.hpp"
#include "logging.hpp"

namespace fs = std::filesystem;
using namespace segmentation;

const std::vector<std::string> DEFAULT_ONNX_CANDIDATES = {
    "../w_onnx/efficientnet_b0_tiled.onnx",
    "w_onnx/efficientnet_b0_tiled.onnx",
    "../efficientnet_b0_tiled.onnx",
    "efficientnet_b0_tiled.onnx"
};

const std::vector<std::string> DEFAULT_OUTPUT_DIRS = {
    "../preds_cpp",
    "preds_cpp",
    "../preds",
    "preds"
};

const std::vector<std::string> SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
};

struct CliConfig {
    fs::path model_path;
    fs::path input_path;
    fs::path masks_path;
    std::string output_path_str;
    std::string log_dir_str;
    TilingConfig tiling;
};

void print_help() {
    std::cout << "Usage: dubai_segmentation_cpp [OPTIONS] --input <FILE_OR_DIR>\n\n"
              << "Arguments / Options:\n"
              << "  -i, --input <PATH>       Path to a single image file OR a directory containing images [required]\n"
              << "  -o, --output <PATH>      Output PNG mask path (single file) or directory (batch mode) [default: preds_cpp/]\n"
              << "  -g, --masks <PATH>       Path to ground-truth mask file or directory for IoU evaluation [optional]\n"
              << "  -m, --model <PATH>       Path to ONNX weights [default: ../w_onnx/efficientnet_b0_tiled.onnx]\n"
              << "  -l, --log-dir <PATH>     Directory to save JSON inference logs [default: inference_logs/]\n"
              << "      --tile-size <INT>    Sliding-window tile size [default: 512]\n"
              << "      --stride <INT>       Sliding-window stride [default: 256]\n"
              << "  -h, --help               Print this help message\n\n"
              << "Examples:\n"
              << "  # Single file inference:\n"
              << "  ./dubai_segmentation_cpp -i ../Dataset_Flattened/images/Tile1_image_part_001.jpg -o ../preds_cpp/tile1_mask.png\n\n"
              << "  # Batch directory inference with IoU evaluation:\n"
              << "  ./dubai_segmentation_cpp -i ../Dataset_Flattened/images/ -g ../Dataset_Flattened/masks/ -o ../preds_cpp/\n";
}

fs::path resolve_default_model_path() {
    for (const auto& candidate : DEFAULT_ONNX_CANDIDATES) {
        if (fs::exists(candidate)) {
            return fs::path(candidate);
        }
    }
    return fs::path(DEFAULT_ONNX_CANDIDATES[0]);
}

fs::path resolve_directory_output_path(const std::string& specified_output) {
    if (!specified_output.empty()) {
        return fs::path(specified_output);
    }
    for (const auto& candidate : DEFAULT_OUTPUT_DIRS) {
        if (fs::exists(candidate)) {
            return fs::path(candidate);
        }
    }
    return fs::path("preds");
}

fs::path resolve_single_output_path(const fs::path& input_file, const std::string& specified_output) {
    std::string stem = input_file.stem().string();
    std::string default_filename = stem + "_mask.png";

    if (!specified_output.empty()) {
        fs::path out(specified_output);
        if (fs::is_directory(out) || specified_output.back() == '/' || specified_output.back() == '\\') {
            return out / default_filename;
        }
        return out;
    }

    fs::path default_dir = resolve_directory_output_path("");
    return default_dir / default_filename;
}

bool is_supported_image(const fs::path& p) {
    if (!fs::is_regular_file(p)) return false;
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return std::tolower(c);
    });
    return std::find(SUPPORTED_EXTENSIONS.begin(), SUPPORTED_EXTENSIONS.end(), ext) != SUPPORTED_EXTENSIONS.end();
}

std::vector<fs::path> find_images_in_dir(const fs::path& dir) {
    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (is_supported_image(entry.path())) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    return files;
}

void print_class_distribution(const std::vector<uint8_t>& mask) {
    std::vector<size_t> counts(NUM_CLASSES, 0);
    for (uint8_t cls : mask) {
        if (cls < NUM_CLASSES) {
            counts[cls]++;
        }
    }
    double total = static_cast<double>(std::max<size_t>(1, mask.size()));
    for (size_t i = 0; i < NUM_CLASSES; ++i) {
        double percentage = (static_cast<double>(counts[i]) / total) * 100.0;
        std::cout << "  - " << std::left << std::setw(22) << CLASS_NAMES[i]
                  << ": " << std::right << std::setw(8) << counts[i]
                  << " pixels (" << std::fixed << std::setprecision(2) << std::setw(5) << percentage << "%)\n";
    }
}

CliConfig parse_cli_args(int argc, char* argv[]) {
    CliConfig config;
    config.model_path = resolve_default_model_path();

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-i" || arg == "--input" || arg == "--image" || arg == "--input-dir") {
            if (++i < argc) config.input_path = fs::path(argv[i]);
            else throw std::runtime_error("Missing value for --input (-i)");
        } else if (arg == "-o" || arg == "--output") {
            if (++i < argc) config.output_path_str = argv[i];
            else throw std::runtime_error("Missing value for --output (-o)");
        } else if (arg == "-g" || arg == "--masks" || arg == "--ground-truth") {
            if (++i < argc) config.masks_path = fs::path(argv[i]);
            else throw std::runtime_error("Missing value for --masks (-g)");
        } else if (arg == "-m" || arg == "--model" || arg == "--weights" || arg == "-w") {
            if (++i < argc) config.model_path = fs::path(argv[i]);
            else throw std::runtime_error("Missing value for --model (-m)");
        } else if (arg == "-l" || arg == "--log-dir") {
            if (++i < argc) config.log_dir_str = argv[i];
            else throw std::runtime_error("Missing value for --log-dir (-l)");
        } else if (arg == "--tile-size") {
            if (++i < argc) config.tiling.tile_size = std::stoull(argv[i]);
            else throw std::runtime_error("Missing value for --tile-size");
        } else if (arg == "--stride") {
            if (++i < argc) config.tiling.stride = std::stoull(argv[i]);
            else throw std::runtime_error("Missing value for --stride");
        } else if (arg == "-h" || arg == "--help") {
            print_help();
            std::exit(0);
        } else {
            if (arg.rfind("-", 0) != 0 && config.input_path.empty()) {
                config.input_path = fs::path(arg);
            } else {
                throw std::runtime_error("Unrecognized argument '" + arg + "'. Run with --help for usage instructions.");
            }
        }
    }

    if (config.input_path.empty()) {
        print_help();
        throw std::runtime_error("Please provide an input image file or directory using `--input <PATH>`.");
    }

    if (config.tiling.tile_size == 0 || config.tiling.stride == 0 || config.tiling.stride > config.tiling.tile_size) {
        throw std::runtime_error("tile-size and stride must be > 0 with stride <= tile-size");
    }

    return config;
}

fs::path find_ground_truth_mask(const fs::path& masks_path, const fs::path& img_path) {
    if (fs::is_regular_file(masks_path)) {
        return masks_path;
    }
    if (!fs::is_directory(masks_path)) {
        throw std::runtime_error("Mask path does not exist: " + masks_path.string());
    }
    std::string stem = img_path.stem().string();
    std::vector<std::string> candidates = {
        stem + ".png",
        stem + ".jpg",
        stem + ".jpeg",
        stem + ".tif",
        stem + ".tiff",
        stem + "_mask.png",
        stem + "_mask.jpg"
    };

    for (const auto& cand : candidates) {
        fs::path p = masks_path / cand;
        if (fs::exists(p)) {
            return p;
        }
    }

    for (const auto& entry : fs::directory_iterator(masks_path)) {
        std::string entry_stem = entry.path().stem().string();
        std::string s1 = entry_stem;
        std::string s2 = stem;
        std::transform(s1.begin(), s1.end(), s1.begin(), ::tolower);
        std::transform(s2.begin(), s2.end(), s2.begin(), ::tolower);
        if (s1 == s2 || s1 == s2 + "_mask") {
            return entry.path();
        }
    }

    throw std::runtime_error("No matching ground truth mask found for '" + img_path.string() + "' in '" + masks_path.string() + "'");
}

std::vector<uint8_t> load_and_decode_mask(const fs::path& mask_file, size_t expected_width, size_t expected_height) {
    int w = 0, h = 0, channels = 0;
    std::vector<uint8_t> raw = load_image_raw(mask_file.string(), w, h, channels);
    if (static_cast<size_t>(w) != expected_width || static_cast<size_t>(h) != expected_height) {
        throw std::runtime_error("Ground-truth mask '" + mask_file.string() + "' dimensions (" +
                                 std::to_string(w) + "x" + std::to_string(h) + ") do not match predicted dimensions (" +
                                 std::to_string(expected_width) + "x" + std::to_string(expected_height) + ")");
    }
    return decode_ground_truth_mask(raw.data(), expected_width, expected_height, channels);
}

void process_single_file(OnnxModel& model, const CliConfig& config) {
    fs::path log_dir = resolve_log_dir(config.log_dir_str);
    fs::path out_file_path = resolve_single_output_path(config.input_path, config.output_path_str);
    if (out_file_path.has_parent_path()) {
        fs::create_directories(out_file_path.parent_path());
    }

    int width = 0, height = 0;
    std::vector<uint8_t> rgb_bytes = load_image_rgb(config.input_path.string(), width, height);

    auto [mask, timing] = predict_full_resolution_with_timing(
        model, rgb_bytes.data(), width, height, config.tiling
    );

    std::vector<uint8_t> color_rgb = render_class_mask_to_rgb(mask);
    save_png(out_file_path.string(), width, height, color_rgb.data());

    auto [iso_time, time_slug] = get_utc_timestamp();

    double total_sec = timing.total_inference_time.count();
    double fwd_sec = timing.model_forward_time.count();
    double pre_post_sec = std::max(0.0, total_sec - fwd_sec);

    std::optional<IouMetricsLog> iou_metrics;
    if (!config.masks_path.empty()) {
        fs::path mask_file = find_ground_truth_mask(config.masks_path, config.input_path);
        std::vector<uint8_t> gt_mask = load_and_decode_mask(mask_file, static_cast<size_t>(width), static_cast<size_t>(height));
        ClassIouResult iou_res = calculate_mask_iou(mask.data(), gt_mask.data(), mask.size());

        std::cout << "\nIoU Evaluation (vs " << mask_file.string() << "):\n";
        std::cout << "  IoU: mIoU=" << std::fixed << std::setprecision(4) << iou_res.mean_iou << "\n";
        IouMetricsLog log_entry;
        log_entry.mean_iou = iou_res.mean_iou;
        for (size_t c = 0; c < NUM_CLASSES - 1; ++c) {
            if (iou_res.unions[c] == 0) {
                std::cout << "    " << CLASS_NAMES[c] << ": N/A (absent)\n";
                log_entry.per_class_iou[c] = std::nullopt;
            } else {
                double val = iou_res.per_class_iou[c];
                std::cout << "    " << CLASS_NAMES[c] << ": " << std::fixed << std::setprecision(4) << val << "\n";
                log_entry.per_class_iou[c] = val;
            }
        }
        iou_metrics = log_entry;
    }

    ImageInferenceLogEntry image_entry;
    image_entry.image_filename = config.input_path.filename().string();
    image_entry.image_path = config.input_path.string();
    image_entry.output_mask_path = out_file_path.string();
    image_entry.width = static_cast<size_t>(width);
    image_entry.height = static_cast<size_t>(height);
    image_entry.total_inference_time_secs = total_sec;
    image_entry.pure_onnx_forward_secs = fwd_sec;
    image_entry.tile_count = timing.tile_count;
    image_entry.avg_tile_forward_ms = timing.avg_tile_forward_ms();
    image_entry.pre_post_processing_secs = pre_post_sec;
    image_entry.iou = iou_metrics;

    RunSummaryMetrics summary;
    summary.images_processed = 1;
    summary.total_tiles_computed = timing.tile_count;
    summary.total_inference_time_secs = total_sec;
    summary.avg_inference_time_per_image_secs = total_sec;
    summary.pure_onnx_forward_secs = fwd_sec;
    summary.avg_tile_forward_ms = timing.avg_tile_forward_ms();
    summary.total_elapsed_secs = total_sec;
    summary.results_saved_to = out_file_path.string();
    summary.iou = iou_metrics;

    SingleRunInferenceLog run_log;
    run_log.timestamp = iso_time;
    run_log.engine = "cpp";
    run_log.model_path = config.model_path.string();
    run_log.input_path = config.input_path.string();
    run_log.output_path = out_file_path.string();
    run_log.tiling = config.tiling;
    run_log.images = {image_entry};
    run_log.summary = summary;

    fs::path log_path = save_run_log(log_dir, run_log, time_slug);

    std::cout << "Successfully segmented '" << config.input_path.string()
              << "' (" << width << "x" << height << "):\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << " Timing Breakdown:\n";
    std::cout << "  - Total inference:      " << std::fixed << std::setprecision(3) << total_sec << "s ("
              << std::fixed << std::setprecision(1) << (total_sec * 1000.0) << " ms)\n";
    std::cout << "  - Pure ONNX forward:    " << std::fixed << std::setprecision(3) << fwd_sec << "s across "
              << timing.tile_count << " tiles (avg "
              << std::fixed << std::setprecision(1) << timing.avg_tile_forward_ms() << " ms/tile)\n";
    std::cout << "  - Pre/post processing:  " << std::fixed << std::setprecision(3) << pre_post_sec << "s\n";
    std::cout << "------------------------------------------------------------\n";
    std::cout << "Saved predicted mask to: " << out_file_path.string() << "\n";
    std::cout << "Saved inference log to:  " << log_path.string() << "\n\n";
    std::cout << "Class distribution:\n";
    print_class_distribution(mask);
}

void process_directory(OnnxModel& model, const CliConfig& config) {
    fs::path log_dir = resolve_log_dir(config.log_dir_str);
    fs::path out_dir = resolve_directory_output_path(config.output_path_str);
    fs::create_directories(out_dir);

    auto image_files = find_images_in_dir(config.input_path);
    if (image_files.empty()) {
        throw std::runtime_error("No supported images (.jpg, .png, etc.) found in '" + config.input_path.string() + "'.");
    }

    std::cout << "Found " << image_files.size() << " images in '" << config.input_path.string()
              << "'. Saving masks to '" << out_dir.string() << "'...\n\n";

    auto total_batch_start = std::chrono::high_resolution_clock::now();
    size_t total_images = image_files.size();
    double total_inference_time_sec = 0.0;
    double total_forward_time_sec = 0.0;
    size_t total_tiles = 0;
    std::vector<ImageInferenceLogEntry> image_entries;
    image_entries.reserve(total_images);

    std::array<uint64_t, NUM_CLASSES - 1> total_intersections{};
    std::array<uint64_t, NUM_CLASSES - 1> total_unions{};
    bool has_masks = !config.masks_path.empty();

    for (size_t idx = 0; idx < total_images; ++idx) {
        const auto& img_path = image_files[idx];
        std::string stem = img_path.stem().string();
        std::string mask_filename = stem + "_mask.png";
        fs::path out_file_path = out_dir / mask_filename;

        try {
            int width = 0, height = 0;
            std::vector<uint8_t> rgb_bytes = load_image_rgb(img_path.string(), width, height);

            auto [mask, timing] = predict_full_resolution_with_timing(
                model, rgb_bytes.data(), width, height, config.tiling
            );

            std::vector<uint8_t> color_rgb = render_class_mask_to_rgb(mask);
            save_png(out_file_path.string(), width, height, color_rgb.data());

            double total_sec = timing.total_inference_time.count();
            double fwd_sec = timing.model_forward_time.count();
            double pre_post_sec = std::max(0.0, total_sec - fwd_sec);

            total_inference_time_sec += total_sec;
            total_forward_time_sec += fwd_sec;
            total_tiles += timing.tile_count;

            std::optional<IouMetricsLog> img_iou_metrics;
            if (has_masks) {
                try {
                    fs::path mask_file = find_ground_truth_mask(config.masks_path, img_path);
                    std::vector<uint8_t> gt_mask = load_and_decode_mask(mask_file, static_cast<size_t>(width), static_cast<size_t>(height));
                    ClassIouResult iou_res = calculate_mask_iou(mask.data(), gt_mask.data(), mask.size());

                    for (size_t c = 0; c < NUM_CLASSES - 1; ++c) {
                        total_intersections[c] += iou_res.intersections[c];
                        total_unions[c] += iou_res.unions[c];
                    }

                    std::cout << "  IoU: mIoU=" << std::fixed << std::setprecision(4) << iou_res.mean_iou << "\n";
                    IouMetricsLog log_entry;
                    log_entry.mean_iou = iou_res.mean_iou;
                    for (size_t c = 0; c < NUM_CLASSES - 1; ++c) {
                        if (iou_res.unions[c] == 0) {
                            std::cout << "    " << CLASS_NAMES[c] << ": N/A (absent)\n";
                            log_entry.per_class_iou[c] = std::nullopt;
                        } else {
                            double val = iou_res.per_class_iou[c];
                            std::cout << "    " << CLASS_NAMES[c] << ": " << std::fixed << std::setprecision(4) << val << "\n";
                            log_entry.per_class_iou[c] = val;
                        }
                    }
                    img_iou_metrics = log_entry;
                } catch (const std::exception& e) {
                    std::cerr << "  IoU evaluation warning for '" << img_path.string() << "': " << e.what() << "\n";
                }
            }

            ImageInferenceLogEntry entry;
            entry.image_filename = img_path.filename().string();
            entry.image_path = img_path.string();
            entry.output_mask_path = out_file_path.string();
            entry.width = static_cast<size_t>(width);
            entry.height = static_cast<size_t>(height);
            entry.total_inference_time_secs = total_sec;
            entry.pure_onnx_forward_secs = fwd_sec;
            entry.tile_count = timing.tile_count;
            entry.avg_tile_forward_ms = timing.avg_tile_forward_ms();
            entry.pre_post_processing_secs = pre_post_sec;
            entry.iou = img_iou_metrics;
            image_entries.push_back(entry);

            std::cout << "[" << (idx + 1) << "/" << total_images << "] "
                      << img_path.filename().string() << " (" << width << "x" << height << ") -> "
                      << mask_filename << " | Inference: "
                      << std::fixed << std::setprecision(3) << total_sec << "s ("
                      << timing.tile_count << " tiles, "
                      << std::fixed << std::setprecision(1) << timing.avg_tile_forward_ms() << "ms/tile)\n";
        } catch (const std::exception& e) {
            std::cerr << "[" << (idx + 1) << "/" << total_images << "] Error processing '"
                      << img_path.string() << "': " << e.what() << "\n";
        }
    }

    double elapsed_sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - total_batch_start).count();
    double avg_tile_ms = (total_tiles > 0) ? (total_forward_time_sec * 1000.0 / total_tiles) : 0.0;
    auto [global_iso, global_slug] = get_utc_timestamp();

    std::optional<IouMetricsLog> summary_iou;
    if (has_masks) {
        IouMetricsLog overall_log;
        double sum_dataset_iou = 0.0;
        size_t count_dataset_classes = 0;
        for (size_t c = 0; c < NUM_CLASSES - 1; ++c) {
            if (total_unions[c] > 0) {
                double val = static_cast<double>(total_intersections[c]) / static_cast<double>(total_unions[c]);
                overall_log.per_class_iou[c] = val;
                sum_dataset_iou += val;
                count_dataset_classes++;
            } else {
                overall_log.per_class_iou[c] = std::nullopt;
            }
        }
        overall_log.mean_iou = (count_dataset_classes > 0) ? (sum_dataset_iou / static_cast<double>(count_dataset_classes)) : 0.0;

        std::cout << "\nOverall dataset IoU: mIoU=" << std::fixed << std::setprecision(4) << overall_log.mean_iou << "\n";
        for (size_t c = 0; c < NUM_CLASSES - 1; ++c) {
            if (overall_log.per_class_iou[c].has_value()) {
                std::cout << "  " << CLASS_NAMES[c] << ": " << std::fixed << std::setprecision(4) << *overall_log.per_class_iou[c] << "\n";
            } else {
                std::cout << "  " << CLASS_NAMES[c] << ": N/A (absent from dataset)\n";
            }
        }
        summary_iou = overall_log;
    }

    RunSummaryMetrics summary;
    summary.images_processed = image_entries.size();
    summary.total_tiles_computed = total_tiles;
    summary.total_inference_time_secs = total_inference_time_sec;
    summary.avg_inference_time_per_image_secs = !image_entries.empty() ? (total_inference_time_sec / image_entries.size()) : 0.0;
    summary.pure_onnx_forward_secs = total_forward_time_sec;
    summary.avg_tile_forward_ms = avg_tile_ms;
    summary.total_elapsed_secs = elapsed_sec;
    summary.results_saved_to = out_dir.string();
    summary.iou = summary_iou;

    SingleRunInferenceLog run_log;
    run_log.timestamp = global_iso;
    run_log.engine = "cpp";
    run_log.model_path = config.model_path.string();
    run_log.input_path = config.input_path.string();
    run_log.output_path = out_dir.string();
    run_log.tiling = config.tiling;
    run_log.images = image_entries;
    run_log.summary = summary;

    fs::path log_path = save_run_log(log_dir, run_log, global_slug);

    std::cout << "\n============================================================\n";
    std::cout << " Batch Summary:\n";
    std::cout << "  - Images processed:     " << total_images << "\n";
    std::cout << "  - Total tiles computed: " << total_tiles << "\n";
    std::cout << "  - Total inference time: " << std::fixed << std::setprecision(2) << total_inference_time_sec
              << "s (avg " << (total_inference_time_sec / total_images) << "s/image)\n";
    std::cout << "  - Pure ONNX forward:    " << std::fixed << std::setprecision(2) << total_forward_time_sec
              << "s (avg " << std::fixed << std::setprecision(1) << avg_tile_ms << "ms/tile)\n";
    std::cout << "  - Total elapsed time:   " << std::fixed << std::setprecision(2) << elapsed_sec << "s\n";
    std::cout << "  - Results saved to:     " << out_dir.string() << "\n";
    std::cout << "  - Inference log saved:  " << log_path.string() << "\n";
    std::cout << "============================================================\n";
}

int main(int argc, char* argv[]) {
    try {
        CliConfig config = parse_cli_args(argc, argv);

        std::cout << "============================================================\n";
        std::cout << " Dubai Satellite Aerial Image Segmentation (ONNX Runtime)   \n";
        std::cout << "============================================================\n";
        std::cout << "Model:      " << config.model_path.string() << "\n";
        std::cout << "Input:      " << config.input_path.string() << "\n";
        if (!config.masks_path.empty()) {
            std::cout << "Masks:      " << config.masks_path.string() << "\n";
        }
        std::cout << "Tiling:     tile_size=" << config.tiling.tile_size
                  << ", stride=" << config.tiling.stride << "\n";
        std::cout << "Log Dir:    " << resolve_log_dir(config.log_dir_str).string() << "\n";

        auto load_start = std::chrono::high_resolution_clock::now();
        OnnxModel model(config.model_path.string());
        double load_sec = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - load_start).count();

        std::cout << "ONNX model loaded successfully in " << std::fixed << std::setprecision(2) << load_sec << "s.\n\n";

        if (fs::is_directory(config.input_path)) {
            process_directory(model, config);
        } else if (fs::is_regular_file(config.input_path)) {
            process_single_file(model, config);
        } else {
            throw std::runtime_error("Input path '" + config.input_path.string() + "' does not exist as a file or directory.");
        }

    } catch (const std::exception& e) {
        std::cerr << "\nError: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
