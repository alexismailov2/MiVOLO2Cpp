#pragma once

#include <opencv2/opencv.hpp>

#include "Yolov8.hpp"

#include <memory>
#include <string>

namespace MiVOLO2 {

using AgeAndGenderInput = Yolov8Result;
//struct AgeAndGenderInput {
//  cv::Rect box;
//  float boxConfidence{};
//  float classConfidence{};
//  std::string className;
//};

struct AgeAndGenderResult {
  size_t index{};
  std::string gender;
  float age;
  int genderConfidence;
};

class AgeAndGender {
public:
  AgeAndGender(std::string torchScriptModel);
  ~AgeAndGender();

  auto predict(cv::Mat const& frame,
               std::vector<AgeAndGenderInput> const& detected_bboxes) const -> std::vector<AgeAndGenderResult>;

private:
  class Impl;
  std::unique_ptr<Impl> _impl;
};

}