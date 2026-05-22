// ********************************************************************************************************************
// Copyright [2026] Renesas Electronics Corporation and/or its licensors. All Rights Reserved.
//
// The contents of this file (the "contents") are proprietary and confidential to Renesas Electronics Corporation
// and/or its licensors ("Renesas") and subject to statutory and contractual protections.
//
// Unless otherwise expressly agreed in writing between Renesas and you: 1) you may not use, copy, modify, distribute,
// display, or perform the contents; 2) you may not use any name or mark of Renesas for advertising or publicity
// purposes or in connection with your use of the contents; 3) RENESAS MAKES NO WARRANTY OR REPRESENTATIONS ABOUT THE
// SUITABILITY OF THE CONTENTS FOR ANY PURPOSE; THE CONTENTS ARE PROVIDED "AS IS" WITHOUT ANY EXPRESS OR IMPLIED
// WARRANTY, INCLUDING THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, AND
// NON-INFRINGEMENT; AND 4) RENESAS SHALL NOT BE LIABLE FOR ANY DIRECT, INDIRECT, SPECIAL, OR CONSEQUENTIAL DAMAGES,
// INCLUDING DAMAGES RESULTING FROM LOSS OF USE, DATA, OR PROJECTS, WHETHER IN AN ACTION OF CONTRACT OR TORT, ARISING
// OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THE CONTENTS. Third-party contents included in this file may
// be subject to different terms.
// ********************************************************************************************************************
#include "rzv_yolox/yolox_model.hpp"

#include <algorithm>
#include <cmath>

#include "rzv_model/utils.hpp"

namespace rzv_model
{

class YoloxModel::Impl
{
public:
  int num_classes = 20;  // Default to Pascal VOC classes
  float conf_threshold = 0.25f;
  float iou_threshold = 0.45f;
  std::vector<std::string> class_names;
};

YoloxModel::YoloxModel() : BaseModel(), pimpl_(std::make_unique<Impl>())
{
  MODEL_INFO("YOLOX model instance created with default settings");

  // Default class names (can be overridden with set_class_names)
  pimpl_->class_names = {"empty"};

  // Set default padding color (YOLOX uses 114 for all channels)
  this->set_padding_color(cv::Scalar(114, 114, 114));
}

YoloxModel::~YoloxModel() = default;

void YoloxModel::set_class_names(const std::vector<std::string> & class_names)
{
  pimpl_->class_names = class_names;
  pimpl_->num_classes = static_cast<int>(class_names.size());
  MODEL_INFO("Set {} class names for YOLOX model", pimpl_->num_classes);
}

void YoloxModel::set_confidence_threshold(float threshold)
{
  if (threshold >= 0.0f && threshold <= 1.0f) {
    pimpl_->conf_threshold = threshold;
    MODEL_INFO("Set confidence threshold to {}", threshold);
  } else {
    MODEL_WARN("Invalid confidence threshold: {} (must be between 0.0 and 1.0)", threshold);
  }
}

void YoloxModel::set_iou_threshold(float threshold)
{
  if (threshold >= 0.0f && threshold <= 1.0f) {
    pimpl_->iou_threshold = threshold;
    MODEL_INFO("Set IoU threshold to {}", threshold);
  } else {
    MODEL_WARN("Invalid IoU threshold: {} (must be between 0.0 and 1.0)", threshold);
  }
}

float YoloxModel::get_confidence_threshold() const { return pimpl_->conf_threshold; }

float YoloxModel::get_iou_threshold() const { return pimpl_->iou_threshold; }

const std::vector<std::string> & YoloxModel::get_class_names() const { return pimpl_->class_names; }

cv::Mat YoloxModel::preprocess(const ModelInput & input) { return BaseModel::preprocess(input); }

cv::Mat YoloxModel::fallback_preprocess(const ModelInput & input)
{
  return BaseModel::software_preprocess(input, false);
}

std::unique_ptr<ModelResult> YoloxModel::postprocess(const std::vector<cv::Mat> & output_tensors)
{
  // YOLOX output explanation:
  // - Format: Multi-scale Feature Maps (FPN) with shape [1, 25, H, W]
  // - Each tensor represents detections at a different scale (typically 80×80, 40×40, and 20×20)
  //   allowing detection of objects at various sizes
  // - The 25 channels contain:
  //   Channels 0-3: Four bounding box parameters (x, y, width, height)
  //   Channel 4: Object confidence score
  //   Channels 5-24: Class probabilities for 20 object classes

  auto result = std::make_unique<YOLOXDetectionResult>();

  if (output_tensors.empty()) {
    MODEL_ERROR("Empty output from YOLOX model");
    return result;
  }

  MODEL_DEBUG("Processing {} output tensors", output_tensors.size());

  // Prepare vectors for detection data
  std::vector<cv::Rect2f> bboxes;
  std::vector<float> confidences;
  std::vector<int> class_ids;
  std::vector<std::string> class_names;

  int input_w = get_shape_info().input_width();
  int num_classes = pimpl_->num_classes;

  // Process each of the output tensors (from different FPN levels)
  for (size_t tensor_idx = 0; tensor_idx < output_tensors.size(); ++tensor_idx) {
    const cv::Mat & tensor = output_tensors[tensor_idx];

    if (tensor.empty()) {
      MODEL_WARN("Empty tensor at index {}", tensor_idx);
      continue;
    }

    // For the YOLOX FPN outputs with shape [1, 25, H, W]
    if (tensor.dims == 4 && tensor.size[0] == 1) {
      int channels = tensor.size[1];  // Should be num_classes + 5 (4 for bbox, 1 for obj conf)
      int height = tensor.size[2];
      int width = tensor.size[3];

      MODEL_DEBUG(
        "Processing tensor {} with shape [{}, {}, {}, {}]", tensor_idx, tensor.size[0], channels,
        height, width);

      // Each grid cell predicts one object
      float stride = (float)input_w / width;  // Assuming square input

      // Create a properly structured Mat view for each channel
      std::vector<cv::Mat> channel_mats(channels);
      for (int c = 0; c < channels; ++c) {
        size_t channel_offset =
          static_cast<size_t>(c) * static_cast<size_t>(height) * static_cast<size_t>(width);
        channel_mats[c] =
          cv::Mat(height, width, CV_32F, const_cast<float *>(tensor.ptr<float>()) + channel_offset);
      }

      // For each grid cell
      for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
          // Check objectness confidence
          float obj_conf = channel_mats[4].at<float>(h, w);
          if (obj_conf < pimpl_->conf_threshold) {
            continue;
          }

          // Find the class with maximum confidence
          float max_class_conf = 0.0f;
          int max_class_id = -1;

          for (int c = 0; c < num_classes; ++c) {
            float class_conf = channel_mats[5 + c].at<float>(h, w);
            if (class_conf > max_class_conf) {
              max_class_conf = class_conf;
              max_class_id = c;
            }
          }

          // Skip if no class is confident enough or combined confidence is too low
          float confidence = obj_conf * max_class_conf;
          if (max_class_id == -1 || confidence < pimpl_->conf_threshold) {
            continue;
          }

          // Extract bounding box coordinates
          float cx = (channel_mats[0].at<float>(h, w) + w) * stride;
          float cy = (channel_mats[1].at<float>(h, w) + h) * stride;
          float width_val = std::exp(channel_mats[2].at<float>(h, w)) * stride;
          float height_val = std::exp(channel_mats[3].at<float>(h, w)) * stride;

          // Convert from center format to corner format for NMS
          float x = cx - width_val / 2.0f;
          float y = cy - height_val / 2.0f;

          // Filter out small boxes (moved from post-NMS processing)
          if (width_val < 5.0f || height_val < 5.0f) {
            continue;
          }

          bboxes.push_back(cv::Rect2f(x, y, width_val, height_val));
          confidences.push_back(confidence);
          class_ids.push_back(max_class_id);

          // Get class name
          std::string class_name;
          if (max_class_id >= 0 && max_class_id < static_cast<int>(pimpl_->class_names.size())) {
            class_name = pimpl_->class_names[max_class_id];
          } else {
            class_name = "class_" + std::to_string(max_class_id);
          }
          class_names.push_back(class_name);
        }
      }
    } else {
      MODEL_WARN("Unexpected tensor format at index {}. Dims: {}", tensor_idx, tensor.dims);
    }
  }

  // Apply Non-Maximum Suppression
  std::vector<int> indices = Utils::non_maximum_suppression_batched(
    bboxes, confidences, class_ids, pimpl_->conf_threshold, pimpl_->iou_threshold);
  MODEL_DEBUG("NMS returned {} indices", indices.size());

  // Process the kept detections
  float best_score = 0.0f;
  for (int idx : indices) {
    // Convert detection to original image space
    float x1 = bboxes[idx].x;
    float y1 = bboxes[idx].y;
    float x2 = x1 + bboxes[idx].width;
    float y2 = y1 + bboxes[idx].height;

    // Map coordinates back to original image space
    cv::Point2f top_left = map_coordinates_to_original(cv::Point2f(x1, y1));
    cv::Point2f bottom_right = map_coordinates_to_original(cv::Point2f(x2, y2));

    // Calculate box width and height
    float box_w = bottom_right.x - top_left.x;
    float box_h = bottom_right.y - top_left.y;

    // Create final detection
    YOLOXDetection detection;
    detection.bbox = cv::Rect(
      static_cast<int>(std::round(top_left.x)), static_cast<int>(std::round(top_left.y)),
      static_cast<int>(std::round(box_w)), static_cast<int>(std::round(box_h)));
    detection.class_id = class_ids[idx];
    detection.confidence = confidences[idx];
    detection.class_name = class_names[idx];
    detection.is_valid = true;

    result->detections.push_back(detection);
    best_score = std::max(best_score, detection.confidence);

    MODEL_DEBUG(
      "Final detection: {} at: {}, {}, {}, {} with score {}", detection.class_name.c_str(),
      detection.bbox.x, detection.bbox.y, detection.bbox.width, detection.bbox.height,
      detection.confidence);
  }

  // Set highest confidence as overall score
  result->score = indices.empty() ? 0.0f : best_score;

  if (indices.empty()) {
    MODEL_DEBUG("No detections passed NMS");
  }

  return result;
}

}  // namespace rzv_model
