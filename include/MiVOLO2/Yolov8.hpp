#pragma once

#include <opencv2/opencv.hpp>

namespace MiVOLO2 {

struct Yolov8Result
{
  cv::Rect box;
  float boxConfidence{};
  float classConfidence{};
  std::string className;
};

class Yolov8
{
public:
  Yolov8(std::string const& torchScriptModel, float conf_thres = 0.4, float iou_thres = 0.7, int max_det = 600);
  ~Yolov8();
  auto predict(cv::Mat const& input) const -> std::vector<Yolov8Result>;

private:
  class Impl;
  std::unique_ptr<Impl> _impl;
};

}