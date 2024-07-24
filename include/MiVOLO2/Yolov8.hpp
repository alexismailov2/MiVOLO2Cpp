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
  Yolov8(std::string const& torchScriptModel);
  ~Yolov8();
  auto predict(cv::Mat const& input) const -> std::vector<Yolov8Result>;

private:
  class Impl;
  std::unique_ptr<Impl> _impl;
};

}