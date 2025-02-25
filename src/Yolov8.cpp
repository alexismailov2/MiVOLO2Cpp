#include <MiVOLO2/Yolov8.hpp>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <torch/torch.h>
#include <torch/script.h>

#include <iostream>

namespace MiVOLO2 {

using torch::indexing::Slice;
using torch::indexing::None;

namespace {

float letterbox(cv::Mat const& input_image, cv::Mat &output_image, const std::vector<int> &target_size)
{
  if (input_image.cols == target_size[1] &&
      input_image.rows == target_size[0])
  {
    if (input_image.data == output_image.data) {
      return 1.;
    } else {
      output_image = input_image.clone();
      return 1.;
    }
  }

  auto resize_scale = std::min(static_cast<float>(target_size[0]) / static_cast<float>(input_image.rows),
                                     static_cast<float>(target_size[1]) / static_cast<float>(input_image.cols));
  int new_shape_w = std::round((float)input_image.cols * resize_scale);
  int new_shape_h = std::round((float)input_image.rows * resize_scale);
  float padw = ((float)target_size[1] - (float)new_shape_w) / 2.f;
  float padh = ((float)target_size[0] - (float)new_shape_h) / 2.f;

  int top = std::round(padh - 0.1);
  int bottom = std::round(padh + 0.1);
  int left = std::round(padw - 0.1);
  int right = std::round(padw + 0.1);

  cv::resize(input_image, output_image,
             cv::Size(new_shape_w, new_shape_h),
             0, 0, cv::INTER_AREA);

  cv::copyMakeBorder(output_image, output_image, top, bottom, left, right,
                     cv::BORDER_CONSTANT, cv::Scalar(114.));
  return resize_scale;
}

torch::Tensor xywh2xyxy(const torch::Tensor &x)
{
  auto y = torch::empty_like(x);
  auto dw = x.index({"...", 2}).div(2);
  auto dh = x.index({"...", 3}).div(2);
  y.index_put_({"...", 0}, x.index({"...", 0}) - dw);
  y.index_put_({"...", 1}, x.index({"...", 1}) - dh);
  y.index_put_({"...", 2}, x.index({"...", 0}) + dw);
  y.index_put_({"...", 3}, x.index({"...", 1}) + dh);
  return y;
}

// Reference: https://github.com/pytorch/vision/blob/main/torchvision/csrc/ops/cpu/nms_kernel.cpp
torch::Tensor nms(const torch::Tensor &bboxes, const torch::Tensor &scores, float iou_threshold)
{
  if (bboxes.numel() == 0)
    return torch::empty({0}, bboxes.options().dtype(torch::kLong));

  auto x1_t = bboxes.select(1, 0).contiguous();
  auto y1_t = bboxes.select(1, 1).contiguous();
  auto x2_t = bboxes.select(1, 2).contiguous();
  auto y2_t = bboxes.select(1, 3).contiguous();

  torch::Tensor areas_t = (x2_t - x1_t) * (y2_t - y1_t);

  auto order_t = std::get<1>(
    scores.sort(/*stable=*/true, /*dim=*/0, /* descending=*/true));

  auto ndets = bboxes.size(0);
  torch::Tensor suppressed_t = torch::zeros({ndets}, bboxes.options().dtype(torch::kByte));
  torch::Tensor keep_t = torch::zeros({ndets}, bboxes.options().dtype(torch::kLong));

  auto suppressed = suppressed_t.data_ptr<uint8_t>();
  auto keep = keep_t.data_ptr<int64_t>();
  auto order = order_t.data_ptr<int64_t>();
  auto x1 = x1_t.data_ptr<float>();
  auto y1 = y1_t.data_ptr<float>();
  auto x2 = x2_t.data_ptr<float>();
  auto y2 = y2_t.data_ptr<float>();
  auto areas = areas_t.data_ptr<float>();

  int64_t num_to_keep = 0;

  for (int64_t _i = 0; _i < ndets; _i++) {
    auto i = order[_i];
    if (suppressed[i] == 1)
      continue;
    keep[num_to_keep++] = i;
    auto ix1 = x1[i];
    auto iy1 = y1[i];
    auto ix2 = x2[i];
    auto iy2 = y2[i];
    auto iarea = areas[i];

    for (int64_t _j = _i + 1; _j < ndets; _j++) {
      auto j = order[_j];
      if (suppressed[j] == 1)
        continue;
      auto xx1 = std::max(ix1, x1[j]);
      auto yy1 = std::max(iy1, y1[j]);
      auto xx2 = std::min(ix2, x2[j]);
      auto yy2 = std::min(iy2, y2[j]);

      auto w = std::max(static_cast<float>(0), xx2 - xx1);
      auto h = std::max(static_cast<float>(0), yy2 - yy1);
      auto inter = w * h;
      auto ovr = inter / (iarea + areas[j] - inter);
      if (ovr > iou_threshold)
        suppressed[j] = 1;
    }
  }
  return keep_t.narrow(0, 0, num_to_keep);
}

torch::Tensor
non_max_supperession(torch::Tensor &prediction, float conf_thres = 0.4, float iou_thres = 0.7, int max_det = 600)
{
  auto bs = prediction.size(0);
  auto nc = prediction.size(1) - 4;
  auto nm = prediction.size(1) - nc - 4;
  auto mi = 4 + nc;
  auto xc = prediction.index({Slice(), Slice(4, mi)}).amax(1) > conf_thres;

  prediction = prediction.transpose(-1, -2);
  prediction.index_put_({"...", Slice({None, 4})}, xywh2xyxy(prediction.index({"...", Slice(None, 4)})));

  std::vector<torch::Tensor> output;
  for (int i = 0; i < bs; i++) {
    output.push_back(torch::zeros({0, 6 + nm}, prediction.device()));
  }

  for (int xi = 0; xi < prediction.size(0); xi++) {
    auto x = prediction[xi];
    x = x.index({xc[xi]});
    auto x_split = x.split({4, nc, nm}, 1);
    auto box = x_split[0], cls = x_split[1], mask = x_split[2];
    auto [conf, j] = cls.max(1, true);
    x = torch::cat({box, conf, j.toType(torch::kFloat), mask}, 1);
    x = x.index({conf.view(-1) > conf_thres});
    int n = x.size(0);
    if (!n) { continue; }

    // NMS
    auto c = x.index({Slice(), Slice{5, 6}}) * 7680;
    auto boxes = x.index({Slice(), Slice(None, 4)}) + c;
    auto scores = x.index({Slice(), 4});
    auto i = nms(boxes, scores, iou_thres);
    i = i.index({Slice(None, max_det)});
    output[xi] = x.index({i});
  }

  return torch::stack(output);
}

torch::Tensor clip_boxes(torch::Tensor &boxes, const std::vector<int> &shape)
{
  boxes.index_put_({"...", 0}, boxes.index({"...", 0}).clamp(0, shape[1]));
  boxes.index_put_({"...", 1}, boxes.index({"...", 1}).clamp(0, shape[0]));
  boxes.index_put_({"...", 2}, boxes.index({"...", 2}).clamp(0, shape[1]));
  boxes.index_put_({"...", 3}, boxes.index({"...", 3}).clamp(0, shape[0]));
  return boxes;
}

torch::Tensor scale_boxes(const std::vector<int> &img1_shape, torch::Tensor &boxes, const std::vector<int> &img0_shape)
{
  auto gain = (std::min)((float) img1_shape[0] / img0_shape[0], (float) img1_shape[1] / img0_shape[1]);
  auto pad0 = std::round((float) (img1_shape[1] - img0_shape[1] * gain) / 2. - 0.1);
  auto pad1 = std::round((float) (img1_shape[0] - img0_shape[0] * gain) / 2. - 0.1);

  boxes.index_put_({"...", 0}, boxes.index({"...", 0}) - pad0);
  boxes.index_put_({"...", 2}, boxes.index({"...", 2}) - pad0);
  boxes.index_put_({"...", 1}, boxes.index({"...", 1}) - pad1);
  boxes.index_put_({"...", 3}, boxes.index({"...", 3}) - pad1);
  boxes.index_put_({"...", Slice(None, 4)}, boxes.index({"...", Slice(None, 4)}).div(gain));
  return boxes;
}

} // anonymous

#if 0
int main() {
  torch::Device device(/*torch::cuda::is_available() ? torch::kCUDA :*/torch::kCPU);

  try {
    std::string model_path = "/home/work/WORK/upwork/age_and_gender/mivolo/models/yolov8x_person_face.torchscript";
    auto yolo_model = torch::jit::load(model_path);
    yolo_model.eval();
    yolo_model.to(device, torch::kFloat32);

    // Load image and preprocess
    cv::Mat image = cv::imread("/home/work/WORK/upwork/age_and_gender/mivolo/jennifer_lawrence.jpg");

  return 0;
}
#endif

class Yolov8::Impl
{
public:
  Impl(std::string const& torchScriptModel, float conf_thres, float iou_thres, int max_det)
  : _module{torch::jit::load(torchScriptModel)}
  , _device{/*torch::cuda::is_available() ? torch::kCUDA :*/torch::kCPU} // TODO: temporary supported only CPU
  , _classes{"person", "face"}
  , _conf_thres{conf_thres}
  , _iou_thres{iou_thres}
  , _max_det{max_det}
  {
    _module.eval();
    _module.to(_device, torch::kFloat32);
  }

  auto predict(cv::Mat const& input) -> std::vector<Yolov8Result>
  {
    cv::Mat input_image;
    letterbox(input, input_image, {640, 640});

    torch::Tensor image_tensor = torch::from_blob(input_image.data,
                                                  {input_image.rows, input_image.cols, 3},
                                                  torch::kByte).to(_device);
    image_tensor = image_tensor.toType(torch::kFloat32).div(255);
    image_tensor = image_tensor.permute({2, 0, 1});
    image_tensor = image_tensor.unsqueeze(0);
    std::vector<torch::jit::IValue> inputs {image_tensor};

    torch::Tensor output = _module.forward(inputs).toTensor().cpu();

    auto keep = non_max_supperession(output, _conf_thres, _iou_thres, _max_det)[0];
    auto boxes = keep.index({Slice(), Slice(None, 4)});
    keep.index_put_({Slice(), Slice(None, 4)},
                    scale_boxes({input_image.rows, input_image.cols}, boxes, {input.rows, input.cols}));

    std::vector<Yolov8Result> results;
    results.reserve(keep.size(0));
    for (int i = 0; i < keep.size(0); i++) {
      int x1_ = (int)keep[i][0].item().toFloat();
      int y1_ = (int)keep[i][1].item().toFloat();
      int x2_ = (int)keep[i][2].item().toFloat();
      int y2_ = (int)keep[i][3].item().toFloat();
      int x1 = std::clamp(x1_, 0, input.cols-1);
      int y1 = std::clamp(y1_, 0, input.rows-1);
      int x2 = std::clamp(x2_, 0, input.cols-1);
      int y2 = std::clamp(y2_, 0, input.rows-1);
      if (((x2 - x1) <= 1) || ((y2 - y1) <= 1)) {
        continue;
      }
      float conf = keep[i][4].item().toFloat();
      int cls = keep[i][5].item().toInt();
      if (cls == 0) {
        continue;
      }
      std::cout << cls << "[" << x1 << "," << y1 << "," << x2 << "," << y2 << "]" << std::endl;
      results.push_back({cv::Rect{x1, y1, x2-x1, y2-y1}, 0, conf, _classes[cls]});
//      results[i].className = _classes[cls];
//      results[i].box = ;
//      results[i].classConfidence = conf;
//      results[i].boxConfidence = 0;
    }
//    for (int i = 0; i < keep.size(0); i++) {
//      int x1 = keep[i][0].item().toFloat();
//      int y1 = keep[i][1].item().toFloat();
//      int x2 = keep[i][2].item().toFloat();
//      int y2 = keep[i][3].item().toFloat();
//      float conf = keep[i][4].item().toFloat();
//      int cls = keep[i][5].item().toInt();
//
//      std::cout << "Rect: [" << x1 << "," << y1 << "," << x2 << "," << y2 << "]  Conf: " << conf << "  Class: " << _classes[cls] << std::endl;
//    }
    return results;
  }

private:
  mutable torch::jit::script::Module _module;
  torch::Device _device{torch::kCPU};
  std::vector<std::string> _classes;
  float _conf_thres{0.4};
  float _iou_thres{0.7};
  int _max_det{600};
};

Yolov8::Yolov8(std::string const& torchScriptModel, float conf_thres, float iou_thres, int max_det)
: _impl{std::make_unique<Impl>(torchScriptModel, conf_thres, iou_thres, max_det)}
{
}

Yolov8::~Yolov8() = default;

auto Yolov8::predict(cv::Mat const& input) const -> std::vector<Yolov8Result>
{
  return _impl->predict(input);
}

}

