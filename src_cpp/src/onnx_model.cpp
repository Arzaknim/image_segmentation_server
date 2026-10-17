#include "onnx_model.hpp"
#include <filesystem>
#include <stdexcept>
#include <iostream>

namespace segmentation {

OnnxModel::OnnxModel(const std::string& model_path)
    : env_(ORT_LOGGING_LEVEL_WARNING, "dubai_segmentation_ort"),
      memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault)) {
    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("ONNX model file not found at '" + model_path + "'");
    }

    session_options_.SetIntraOpNumThreads(4);
    session_options_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

    Ort::AllocatorWithDefaultOptions allocator;
    size_t num_inputs = session_->GetInputCount();
    input_node_names_.reserve(num_inputs);
    input_names_ptrs_.reserve(num_inputs);
    for (size_t i = 0; i < num_inputs; ++i) {
        auto name_ptr = session_->GetInputNameAllocated(i, allocator);
        input_node_names_.emplace_back(name_ptr.get());
    }
    for (const auto& name : input_node_names_) {
        input_names_ptrs_.push_back(name.c_str());
    }

    size_t num_outputs = session_->GetOutputCount();
    output_node_names_.reserve(num_outputs);
    output_names_ptrs_.reserve(num_outputs);
    for (size_t i = 0; i < num_outputs; ++i) {
        auto name_ptr = session_->GetOutputNameAllocated(i, allocator);
        output_node_names_.emplace_back(name_ptr.get());
    }
    for (const auto& name : output_node_names_) {
        output_names_ptrs_.push_back(name.c_str());
    }

    if (num_outputs > 0) {
        auto type_info = session_->GetOutputTypeInfo(0);
        auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
        auto shape = tensor_info.GetShape();
        if (shape.size() >= 2 && shape[1] > 0) {
            num_classes_ = static_cast<size_t>(shape[1]);
        }
    }
}

std::vector<float> OnnxModel::predict_tile(const float* tile_data, size_t channels, size_t height, size_t width) {
    std::vector<int64_t> input_shape = {
        1,
        static_cast<int64_t>(channels),
        static_cast<int64_t>(height),
        static_cast<int64_t>(width)
    };
    size_t total_elements = channels * height * width;

    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
        memory_info_,
        const_cast<float*>(tile_data),
        total_elements,
        input_shape.data(),
        input_shape.size()
    );

    std::vector<Ort::Value> output_tensors;
    {
        std::lock_guard<std::mutex> lock(session_mutex_);
        output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_ptrs_.data(),
            &input_tensor,
            1,
            output_names_ptrs_.data(),
            output_names_ptrs_.size()
        );
    }

    if (output_tensors.empty()) {
        throw std::runtime_error("ONNX Runtime returned no output tensors.");
    }

    const auto& out = output_tensors.front();
    auto shape_info = out.GetTensorTypeAndShapeInfo();
    auto shape = shape_info.GetShape();
    if (shape.size() != 4) {
        throw std::runtime_error("Expected 4D model output [B, C, H, W], got shape dimensions: " +
                                 std::to_string(shape.size()));
    }

    size_t out_c = static_cast<size_t>(shape[1]);
    size_t out_h = static_cast<size_t>(shape[2]);
    size_t out_w = static_cast<size_t>(shape[3]);
    size_t out_count = out_c * out_h * out_w;

    const float* out_ptr = out.GetTensorData<float>();
    return std::vector<float>(out_ptr, out_ptr + out_count);
}

} // namespace segmentation

