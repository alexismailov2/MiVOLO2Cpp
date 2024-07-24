#include <MiVOLO2/AgeAndGender.hpp>
#include <MiVOLO2/Yolov8.hpp>

#include "../src/TimeMeasuring.hpp"

#include <filesystem>

void draw(std::vector<MiVOLO2::Yolov8Result> const& foundObjects,
          std::vector<MiVOLO2::AgeAndGenderResult> const& foundGenderInfo,
          cv::Mat &image) {
  auto textColor = cv::Scalar(255, 255, 255);
  auto boxColor = cv::Scalar(0,  0,255);

  for (auto const& info : foundGenderInfo) {
    auto const& item = foundObjects[info.index];
    cv::Rect const& box = item.box;
    cv::rectangle(image, box, boxColor, 1);
    std::string infoString = /*std::string("[c(") + std::to_string((int)std::round(item.classConfidence*100)) + "%) " + */info.gender[0] /*+ "(" + std::to_string(info.genderConfidence) + "%) "*/ + std::to_string((int)std::round(info.age)) /*+ "yo] "*/;
    cv::Size textSize = cv::getTextSize(infoString, cv::FONT_HERSHEY_DUPLEX, 1, 1, nullptr);
    cv::Rect textBox(box.x, box.y - 40, textSize.width + 10, textSize.height + 20);

    cv::rectangle(image, textBox, boxColor, cv::FILLED);
    cv::putText(image, infoString, cv::Point(box.x + 5, box.y - 10), cv::FONT_HERSHEY_DUPLEX, 1, textColor, 1, 0);
  }
}

int main(int argc, char** argv)
{
  if (argc < 2) {
    std::cout << "Usage: ageDetect -path <fileName or path to folder> -json <jsonFileName> -outPath <folderName>" << std::endl;
  }
  std::string path = "";
  std::string modelsDir = "./";
  std::string outPath = "./results/";
  for (int i = 1; i < argc; i += 2) {
    auto str = std::string(argv[i]);
    if (str == "-path") {
      path = argv[i+1];
    } else if (str == "-modelsDir") {
      modelsDir = argv[i+1];
    } else if (str == "-outPath") {
      outPath = argv[i+1];
    }
  }

  bool isFolder = std::filesystem::is_directory(path);
  std::cout << "std::filesystem::is_directory: " << isFolder << ", " << path << std::endl;
  std::cout << "modelsDir: " << modelsDir << std::endl;
  std::cout << "outPath:" << outPath << std::endl;
  std::filesystem::create_directories(outPath);

  auto yolo = MiVOLO2::Yolov8(modelsDir + "/yolov8x_person_face.torchscript");
  auto ageAndGender = MiVOLO2::AgeAndGender(modelsDir + "/scriptmodule_cpu.pt");

  cv::VideoCapture cap;
  if (path.empty()) {
    cap.open(0);
    while(cv::waitKey(1) < 0) {
      cv::Mat frame;
      cap.read(frame);
      if (frame.empty())
      {
        cv::waitKey();
        break;
      }
      auto foundObjects = yolo.predict(frame);
      auto foundGenderInfo = ageAndGender.predict(frame, foundObjects);
      draw(foundObjects, foundGenderInfo, frame);
      imshow("Frame", frame);
    }
  } else if (isFolder) {
    for (auto const& file : std::filesystem::directory_iterator(path)) {
      TAKEN_TIME();
      std::cout << "file: " << file.path().string() << std::endl;
      cv::Mat frame = cv::imread(file.path().string(), cv::IMREAD_COLOR);
      auto foundObjects = yolo.predict(frame);
      auto foundGenderInfo = ageAndGender.predict(frame, foundObjects);
      draw(foundObjects, foundGenderInfo, frame);
      cv::imwrite(outPath + "/" + file.path().filename().string(), frame);
    }
  } else {
    cv::Mat frame = cv::imread(path, cv::IMREAD_COLOR);
    auto foundObjects = yolo.predict(frame);
    auto foundGenderInfo = ageAndGender.predict(frame, foundObjects);
    draw(foundObjects, foundGenderInfo, frame);
    cv::imwrite(outPath + "/" + std::filesystem::path(path).filename().string(), frame);
  }
  return 0;
}