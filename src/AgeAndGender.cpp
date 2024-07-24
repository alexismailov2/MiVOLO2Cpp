#include "MiVOLO2/AgeAndGender.hpp"

#include <torch/script.h>
#include <torch/library.h>
#include <opencv2/core_detect.hpp>

#include <filesystem>

namespace MiVOLO2 {

namespace {
const auto IMAGENET_DEFAULT_MEAN = cv::Scalar(0.485, 0.456, 0.406);
const auto IMAGENET_DEFAULT_STD = cv::Scalar(0.229, 0.224, 0.225);

void class_letterbox(cv::Mat& im, cv::Size new_shape = {640, 640}, cv::Scalar const& color = {}, bool scaleup = true)
{
  if ((im.rows == new_shape.height) &&
      (im.cols == new_shape.width))
  {
    return;
  }
  auto r = std::min((float)new_shape.height / (float)im.rows,
                          (float)new_shape.width / (float)im.cols);
  r = !scaleup ? std::min(r, 1.0f) : r;

  auto new_unpad = cv::Size{(int)round((float)im.cols * r), (int)round((float)im.rows * r)};
  auto dw = (float)(new_shape.width - new_unpad.width)/2.0f;
  auto dh = (float)(new_shape.height - new_unpad.height)/2.0f;

  if ((im.rows != new_unpad.width) or
      (im.cols != new_unpad.height))
  {
    cv::resize(im, im, new_unpad, 0, 0, cv::INTER_LINEAR);
  }
  auto top = (int)round(dh - 0.1);
  auto bottom = (int)round(dh + 0.1);
  auto left = (int)round(dw - 0.1);
  auto right = (int)round(dw + 0.1);
  cv::copyMakeBorder(im, im, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

#if 0
auto inference(cv::Mat const& img) -> torch::Tensor {
  //cv::resize(img, img, cv::Size(224, 224), 0, 0, 1);
  auto tensor_image = torch::from_blob(img.data, { img.rows, img.cols, img.channels() }, at::kByte);
  tensor_image = tensor_image.permute({ 2,0,1 });
  tensor_image.unsqueeze_(0);
  tensor_image = tensor_image.toType(c10::kFloat).sub(127.5).mul(0.0078125);
  tensor_image.to(c10::DeviceType::CPU);

  std::vector<torch::jit::IValue> inputs{tensor_image};

  at::Tensor output = _module.forward(inputs).toTensor();
  std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << '\n';
}
#endif
auto prepare_classification_images(std::vector<cv::Mat> img_list,
                                   int target_size = 224,
                                   cv::Scalar mean = IMAGENET_DEFAULT_MEAN,
                                   cv::Scalar std = IMAGENET_DEFAULT_STD,
                                   std::string device = "cpu") -> torch::Tensor
{
  std::vector<torch::Tensor> prepared_images;
  for (auto& img : img_list) {
    if (img.empty()) {
      throw std::runtime_error("not supported");
#if 0
      img = cv::dnn::blobFromImage(modelInput,//Aligned,
                             1.0 / 255.0,
                             _inputSize,
                             mean,
                             isNeededToBeSwappedRAndB,
                             false)

      auto img_new = cv::Mat::zeros(target_size,target_size, CV_32FC3);//torch::zeros((3, target_size, target_size), torch::kF32);
      img_new = F.normalize(img_new, mean, std);
      img = img.unsqueeze(0);
      prepared_images.append(img);
      append(img);
      continue;
#endif
    }
    class_letterbox(img, cv::Size{target_size, target_size});
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32FC3);
    //std::cout << "imgrgb=" << img << std::endl;
    //cv::imwrite("/home/work/WORK/upwork/age_and_gender/mivolo/output/0_crop_rgb.jpg", img);
    img /= 255.0;
    img = (img - mean) / std;
#if 0
    img = img.astype(dtype=np.float32);
    img = img.transpose({2, 0, 1});
    img = np.ascontiguousarray(img);
    img = torch::from_numpy(img);
    img = img.unsqueeze(0);
#else
    //std::cout << "img.channels()=" << img.channels() << std::endl;
    prepared_images.push_back(torch::from_blob(img.ptr<float>(), { img.rows, img.cols, img.channels() }, at::kFloat));//at::kByte));
    auto& tensor_image = prepared_images.back();
    //std::cout << "tensor_image.sizes()=" << tensor_image.sizes() << std::endl;
    //std::cout << "tensor_image=" << tensor_image << std::endl;
    tensor_image = tensor_image.permute({ 2,0,1 });
    tensor_image.unsqueeze_(0);
    //tensor_image = tensor_image.toType(c10::kFloat);//.sub(127.5).mul(0.0078125);
    tensor_image.to((device == "gpu") ? c10::DeviceType::CUDA : c10::DeviceType::CPU);
#endif
    //return prepared_images;
  }

  if (prepared_images.empty()) {
    return {};
  }
  return torch::concat(prepared_images);
}
}

class PersonAndFaceCrops
{
public:
  PersonAndFaceCrops() = default;
  //std::map<int, cv::Mat> crops_persons;
  std::map<int, cv::Mat> crops_faces;
  std::map<int, cv::Mat> crops_faces_wo_body;
  //std::map<int, cv::Mat> crops_persons_wo_face;

  void _add_to_output(std::map<int, cv::Mat> const& crops,
                      std::vector<cv::Mat>& out_crops,
                      std::vector<int>& out_crop_inds)
  {
    //out_crops.clear();
    //out_crop_inds.clear();
    for (auto const& item : crops) {
      out_crops.push_back(item.second);
      out_crop_inds.push_back(item.first);
    }
  }

  auto _get_all_faces(bool use_persons, bool use_faces) -> std::tuple<std::vector<int>, std::vector<cv::Mat>>
  {
    std::vector<int> faces_inds;
    std::vector<cv::Mat> faces_crops;

//    if (!use_faces)
//    {
//      faces_inds.resize(crops_persons.size() + crops_persons_wo_face.size());
//      faces_crops.resize(faces_inds.size());
//      return std::make_tuple(faces_inds, faces_crops);
//    }

    _add_to_output(crops_faces, faces_crops, faces_inds);
    _add_to_output(crops_faces_wo_body, faces_crops, faces_inds);

//    if (use_persons) {
//      faces_inds.resize(crops_persons_wo_face.size());
//      faces_crops.resize(faces_inds.size());
//    }
    return std::make_tuple(faces_inds, faces_crops);
  }
#if 0
  auto _get_all_bodies(bool use_persons, bool use_faces) -> std::tuple<std::vector<int>, std::vector<cv::Mat>>
  {
    std::vector<int> bodies_inds;
    std::vector<cv::Mat> bodies_crops;

    if (!use_persons)
    {
      bodies_inds.resize(crops_faces.size() + crops_faces_wo_body.size());
      bodies_crops.resize(bodies_inds.size());
      return std::make_tuple(bodies_inds, bodies_crops);
    }
    _add_to_output(crops_persons, bodies_crops, bodies_inds);
    if (use_faces) {
      bodies_inds.resize(crops_faces_wo_body.size());
      bodies_crops.resize(bodies_inds.size());
    }
    _add_to_output(crops_persons_wo_face, bodies_crops, bodies_inds);

    return std::make_tuple(bodies_inds, bodies_crops);
  }
#endif
  void save(std::string out_dir = "output") {
    std::filesystem::create_directories(out_dir);
    auto ind = 0;
    for (auto const& crops : {/*crops_persons,*/ crops_faces, crops_faces_wo_body/*, crops_persons_wo_face*/}) {
      for (auto const& crop : crops) {
        if (crop.second.empty()) {
          continue;
        }
        cv::imwrite(out_dir + "/" + std::to_string(ind) + "_crop.jpg", crop.second);
        ++ind;
      }
    }
  }
};

class PersonAndFaceResult
{
public:
  std::vector<AgeAndGenderInput> input;
  std::map<int, int> face_to_person_map;
  //std::vector<int> unassigned_persons_inds;
  std::vector<std::optional<float>> ages{};
  std::vector<std::optional<std::string>> genders{};
  std::vector<std::optional<float>> gender_scores{};

  PersonAndFaceResult(cv::Mat const& frame, std::vector<AgeAndGenderInput> input_)
    : input{std::move(input_)}
    , face_to_person_map{}
    //, unassigned_persons_inds{get_bboxes_inds("person")}
    , ages(input.size())
    , genders(input.size())
    , gender_scores(input.size())
  {
    auto bboxes = get_bboxes_inds("face");
    for (auto const& bbox : bboxes) {
      face_to_person_map.insert({bbox, -1});
    }
    //auto names = set(input.names.values());
    //assert(names.find("person") && names.find("face"));
  }

  auto get_bboxes_inds(std::string const& category) const -> std::vector<int> {
    std::vector<int> bboxes;
    bboxes.reserve(input.size());
    for (auto i = 0; i < input.size(); ++i) {
      if (input[i].className == category) {
        bboxes.push_back(i);
      }
    }
    return bboxes;
  }
#if 0
//  auto get_distance_to_center(int bbox_ind) -> float {
//    im_h, im_w = yolo_results[bbox_ind].orig_shape;
//    x1, y1, x2, y2 = get_bbox_by_ind(bbox_ind).cpu().numpy();
//    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2;
//    dist = math.dist([center_x, center_y], [im_w / 2, im_h / 2]);
//    return dist;
//  }

//  auto _get_id_by_ind(int ind = -1) -> int
//  {
//    if (ind == -1) {
//      return -1;
//    }
//    auto obj_id = input[ind].box.id;
//    if (obj_id == -1) {
//      return -1;
//    }
//    return obj_id.item();
//  }

//  auto get_bbox_by_ind(int ind, int im_h = -1, int im_w = -1) const -> torch::Tensor {
//    auto bb = input[ind].box.xyxy.squeeze().type(torch::kF32);
//    if ((im_h != -1) && (im_w != -1))
//    {
//      bb[0] = torch::clamp(bb[0], 0, im_w - 1);
//      bb[1] = torch::clamp(bb[1], 0, im_h - 1);
//      bb[2] = torch::clamp(bb[2], 0, im_w - 1);
//      bb[3] = torch::clamp(bb[3], 0, im_h - 1);
//    }
//    return bb;
//  }

//  void set_age(int ind, float age) {
//    if (ind != -1) {
//      ages[ind] = age;
//    }
//  }

//  void set_gender(int ind, std::string gender, float gender_score) {
//    if (ind != -1) {
//      genders[ind] = gender;
//      gender_scores[ind] = gender_score;
//    }
//  }

//    auto assign_faces(std::vector<torch::Tensor> persons_bboxes,
//                      std::vector<torch::Tensor> faces_bboxes,
//                      float iou_thresh = 0.0001) -> std::tuple<std::vector<int>, std::vector<int>>
//    {
//        std::vector<int> assigned_faces(faces_bboxes.size());
//        std::vector<int> unassigned_persons_inds(persons_bboxes.size());
//        for (int i = 0; i < persons_bboxes.size(); ++i) { unassigned_persons_inds[i] = i; }
//
//        if (persons_bboxes.empty() || faces_bboxes.empty()) {
//          return std::make_tuple(assigned_faces, unassigned_persons_inds);
//        }
//
//        auto cost_matrix = box_iou(torch::stack(persons_bboxes), torch::stack(faces_bboxes), over_second=True).cpu().numpy();
//        persons_indexes =[];
//        face_indexes =[];
//
//        if (!cost_matrix.empty()) {
//          [persons_indexes, face_indexes] = linear_sum_assignment(cost_matrix, maximize = True);
//        }
//
//        std::set<int> matched_persons;
//        for (auto i = 0; i < persons_indexes.size(); ++i)
//        //for (person_idx, face_idx in zip(persons_indexes, face_indexes))
//        {
//          auto person_idx = persons_indexes[i];
//          auto face_idx = face_indexes[i];
//          if (cost_matrix[person_idx][face_idx] > iou_thresh) {
//            if (matched_persons.count(person_idx)) {
//              // Person can not be assigned twice, in reality this should not happen
//              continue;
//            }
//            assigned_faces[face_idx] = person_idx;
//            matched_persons.insert(person_idx);
//          }
//        }
//
//        unassigned_persons_inds =[p_ind for p_ind in range(len(persons_bboxes)) if p_ind not in matched_persons];
//
//        return std::make_tuple(assigned_faces, unassigned_persons_inds);
//    }

//    void associate_faces_with_persons() {
//      auto face_bboxes_inds = get_bboxes_inds("face");
//      std::vector<torch::Tensor> face_bboxes(face_bboxes_inds.size());
//      for (auto const& ind : face_bboxes_inds) {
//        face_bboxes[ind] = get_bbox_by_ind(ind);
//      }
//
//      auto person_bboxes_inds = get_bboxes_inds("person");
//      std::vector<torch::Tensor> person_bboxes(person_bboxes_inds.size());
//      for (auto const& ind : person_bboxes_inds) {
//        person_bboxes[ind] = get_bbox_by_ind(ind);
//      }
//
//      face_to_person_map = {ind: None for ind in face_bboxes_inds};
//      auto [assigned_faces, unassigned_persons_inds] = assign_faces(person_bboxes, face_bboxes);
//
//      for (face_ind, person_ind in enumerate(assigned_faces)) {
//        face_ind = face_bboxes_inds[face_ind];
//        person_ind = person_bboxes_inds[person_ind] if person_ind is not None else None;
//        face_to_person_map[face_ind] = person_ind;
//      }
//
//      unassigned_persons_inds = [person_bboxes_inds[person_ind] for person_ind in unassigned_persons_inds];
//    }
#endif
    auto crop_object(cv::Mat const& full_image, int ind, std::vector<std::string> const& cut_other_classes = {}) const -> cv::Mat
    {
      const auto IOU_THRESH = 0.000001;
      const auto MIN_PERSON_CROP_AFTERCUT_RATIO = 0.4;
      const auto CROP_ROUND_RATE = 0.3;
      const auto MIN_PERSON_SIZE = 50;

      // get crop of face or person
      auto obj_image = full_image(input[ind].box).clone();

      if ((input[ind].className == "person") &&
          ((obj_image.rows < MIN_PERSON_SIZE) || (obj_image.cols < MIN_PERSON_SIZE))) {
        return {};
      }

      if (cut_other_classes.empty()) {
        return obj_image;
      }
      return obj_image;
#if 0
      // calc iou between obj_bbox and other bboxes
      std::vector<cv::Rect> other_bboxes;
      for (auto const& item : input) {
        other_bboxes.push_back(item.box);
      }
      cv::dnn_objdetect::InferBbox::intersection_over_union()
      auto iou_matrix = box_iou(torch.stack([obj_bbox]), torch.stack(other_bboxes)).cpu().numpy()[0];

      // cut out other objects in case of intersection
      for other_ind, (det, iou) in enumerate(zip(yolo_results.boxes, iou_matrix)):
      other_cat = input.names[int(det.cls)];
      if ((ind == other_ind) || (iou < IOU_THRESH) || (other_cat not in cut_other_classes)) {
        continue;
      }
      o_x1, o_y1, o_x2, o_y2 = det.xyxy.squeeze().type(torch.int32);

      // remap current_person_bbox to reference_person_bbox coordinates
      auto o_x1 = std::max(o_x1 - x1, 0);
      auto o_y1 = std::max(o_y1 - y1, 0);
      auto o_x2 = std::min(o_x2 - x1, obj_image.cols);
      auto o_y2 = std::min(o_y2 - y1, obj_image.rows);

      if (other_cat != "face") {
        if ((o_y1 / obj_image.rows) < CROP_ROUND_RATE) {
          o_y1 = 0;
        }
        if (((obj_image.rows - o_y2) / obj_image.rows) < CROP_ROUND_RATE) {
          o_y2 = obj_image.rows;
        }
        if ((o_x1 / obj_image.cols) < CROP_ROUND_RATE) {
          o_x1 = 0;
        }
        if (((obj_image.cols - o_x2) / obj_image.cols) < CROP_ROUND_RATE) {
          o_x2 = obj_image.cols;
        }
      }

      obj_image[o_y1:o_y2, o_x1:o_x2] = 0;

      auto remain_ratio = np.count_nonzero(obj_image) / (obj_image.shape[0] * obj_image.shape[1] * obj_image.shape[2]);
      if (remain_ratio < MIN_PERSON_CROP_AFTERCUT_RATIO) {
        return {};
      }

      return obj_image;
#endif
    }

    auto collect_crops(cv::Mat const &image) const -> PersonAndFaceCrops
    {
      auto crops_data = PersonAndFaceCrops();
      for (auto [face_ind, person_ind] : face_to_person_map) {
        auto face_image = crop_object(image, face_ind);
        if (person_ind != -1) {
          crops_data.crops_faces_wo_body[face_ind] = face_image;
          continue;
        }
        crops_data.crops_faces[face_ind] = face_image;
        //crops_data.crops_persons[person_ind.value()] = crop_object(image, person_ind.value(), {"face", "person"});;
      }

//      for (auto const &person_ind: unassigned_persons_inds) {
//        crops_data.crops_persons_wo_face[person_ind] = crop_object(image, person_ind, {"face", "person"});;
//      }
      crops_data.save();
      return crops_data;
    }

    auto to_AgeAndGenderResults() const -> std::vector<AgeAndGenderResult> {
      auto bboxes = get_bboxes_inds("face");
      auto ageAndGenderResults = std::vector<AgeAndGenderResult>(bboxes.size());
      for(auto& index : bboxes) {
        ageAndGenderResults[index].index = index;
        ageAndGenderResults[index].gender = genders[index].has_value() ? genders[index].value() : "Not detected";
        ageAndGenderResults[index].age = ages[index].has_value() ? ages[index].value() : -1.0f;
        ageAndGenderResults[index].genderConfidence = static_cast<int>(gender_scores[index].value() * 100);
      }
      return ageAndGenderResults;
    }
};

class AgeAndGender::Impl {
public:
  Impl(std::string torchScriptModel,
       bool half = false,//true,
       bool disable_faces = {},
       bool use_persons = false,//true,
       bool verbose = {},
       std::string torchcompile = {})
    : _module{torch::jit::load(torchScriptModel)}
    //, min_age{}
    //, max_age{}
    //, avg_age{}
    //, num_classes{}
    //, in_chans{3}
    //, with_persons_model{}
    , disable_faces{disable_faces}
    , use_persons{use_persons}
    , only_age{false}
    //, num_classes_gender{2}
    //, input_size{224}
  {
#if 0
    //state = torch.load(ckpt_path, map_location="cpu")

    min_age = state["min_age"];
    max_age = state["max_age"];
    avg_age = state["avg_age"];
    only_age = state["no_gender"];

    if ("with_persons_model" in state) {
      with_persons_model = state["with_persons_model"];
    } else {
      with_persons_model = "patch_embed.conv1.0.weight" in state["state_dict"];
    }

    num_classes = only_age ? 1 : 3;
    in_chans = !with_persons_model ? 3 : 6;
    use_persons = use_persons && with_persons_model;

    if (!with_persons_model && disable_faces) {
      throw std::runtime_error("You can not use disable-faces for faces-only model");
    }

    if (with_persons_model && disable_faces && !use_persons) {
      throw std::runtime_error("You can not disable faces and persons together. "
                               "Set --with-persons if you want to run with --disable-faces");
    }
    input_size = state["state_dict"]["pos_embed"].shape[1] * 16;
#endif
  }

  bool use_person_crops() const { return with_persons_model && use_persons; }
  bool use_face_crops() const { return !disable_faces && !with_persons_model; }

  auto inference(torch::Tensor const& model_input) const -> torch::Tensor {
//    //with torch.no_grad():
//    if (half) {
//      model_input = model_input.half();
//    }
    return _module({model_input}).toTensor();
  }

  auto predict(cv::Mat const& frame, std::vector<AgeAndGenderInput> const& input) const -> std::vector<AgeAndGenderResult>
  {
    if (input.empty()) {
      return {};
    }
    auto detected_bboxes = PersonAndFaceResult(frame, input);
    if ((use_persons && detected_bboxes.get_bboxes_inds("face").empty()) ||
        (disable_faces && detected_bboxes.get_bboxes_inds("persons").empty()))
    {
      return {};
    }
    auto [faces_input, /*person_input,*/ faces_inds/*, bodies_inds*/] = prepare_crops(frame, detected_bboxes);
    if (faces_inds.empty() /*&& bodies_inds.empty()*/) {
      return {};
    }
    auto output = inference(/*with_persons_model ? torch::cat((faces_input, person_input), 1) : */faces_input);

    // write gender and age results into detected_bboxes
    fill_in_results(output, detected_bboxes, faces_inds);//, bodies_inds)

    return detected_bboxes.to_AgeAndGenderResults();
  }

  void fill_in_results(torch::Tensor const& output,
                       PersonAndFaceResult& detected_bboxes,
                       std::vector<int> const& faces_inds/*,
                       std::vector<int> const& bodies_inds*/) const
  {
    using namespace torch::indexing;
    auto age_output = only_age ? output : output.index({ Slice(None, None), 2 });
    auto gender_output = only_age ? torch::Tensor{} : output.index({ Slice(None, None), Slice(None, 2)}).softmax(-1);
    auto [gender_probs, gender_indx] = only_age ? std::tuple<torch::Tensor, torch::Tensor>{} : gender_output.topk(1);

    //assert(output.shape[0] == len(faces_inds) == len(bodies_inds))

    for (auto index = 0; index < output.sizes()[0]; ++index) {
      auto face_ind = faces_inds[index];
      //auto body_ind = bodies_inds[index];

      auto age = age_output[index].item().toFloat();
      age = age * (max_age - min_age) + avg_age;
      age = std::round(age);//, 2);

      detected_bboxes.ages[face_ind] = age;
      //detected_bboxes.ages[body_ind] = age;

      std::cout << "\tage: " << age << std::endl;

      if (gender_probs.sizes()[0] != 0) {
        auto gender = (gender_indx[index].item().toInt() == 0) ? "male" : "female";
        auto gender_score = gender_probs[index].item().toFloat();

        std::cout << "\tgender: " << gender << " [" << static_cast<int>(gender_score * 100) << "%]" << std::endl;

        detected_bboxes.genders[face_ind] = gender;
        detected_bboxes.gender_scores[face_ind] = gender_score;
        //detected_bboxes.set_gender(body_ind, gender, gender_score);
      }
    }
  }

  auto prepare_crops(cv::Mat const& image,
                     PersonAndFaceResult const& detected_bboxes) const -> std::tuple<torch::Tensor/*, torch::Tensor*/, std::vector<int>/*, std::vector<int>*/>
  {
//    if (use_person_crops() && use_face_crops()) {
//      detected_bboxes.associate_faces_with_persons();
//    }

    auto crops = detected_bboxes.collect_crops(image);

//    auto [bodies_inds, bodies_crops] = crops._get_all_bodies(use_person_crops(), use_face_crops());
    auto [faces_inds, faces_crops] = crops._get_all_faces(use_person_crops(), use_face_crops());

//  if (!use_face_crops()) {
//    for (auto const& f : faces_crops) {
//      assert(f.empty());
//    }
//  }
//  if (!use_person_crops()) {
//    for (auto const& p : bodies_crops) {
//      assert(p.empty());
//    }
//  }

    auto faces_input = prepare_classification_images(faces_crops, input_size);
//    auto person_input = prepare_classification_images(bodies_crops, input_size);

    return std::make_tuple(faces_input/*, person_input*/, faces_inds/*, bodies_inds*/);
  }

private:
  mutable torch::jit::script::Module _module;
  int min_age{1};
  int max_age{95};
  int avg_age{48};
  int num_classes{3};
  int in_chans{3};
  bool with_persons_model{};
  bool disable_faces{};
  bool use_persons{true};
  bool only_age{};
  int num_classes_gender{2};
  int input_size{224};
};

AgeAndGender::AgeAndGender(std::string torchScriptModel)
    : _impl(std::make_unique<Impl>(std::move(torchScriptModel)))
{
}

AgeAndGender::~AgeAndGender() = default;

auto AgeAndGender::predict(cv::Mat const& frame,
                           std::vector<AgeAndGenderInput> const& detected_bboxes) const -> std::vector<AgeAndGenderResult>
{
  return _impl->predict(frame, detected_bboxes);
}

}
