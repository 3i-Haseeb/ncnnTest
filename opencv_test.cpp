#include <filesystem>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;

namespace fs = std::filesystem;

int main() {
  fs::path currentPath = fs::current_path();
  std::string currentPathStr = currentPath.string();

  std::string imagePath = currentPathStr + "/test.png";
  std::cout << imagePath << std::endl;
  Mat img = imread(imagePath, IMREAD_COLOR);
  if (img.empty()) {
    std::cout << "Could not read the image: " << imagePath << std::endl;
    return 1;
  }
  imsho("Display window", img);
  while (true) {
    int key = waitKey(1);
    if (key == 27 || key == 'q') { // Esc key or 'q' key
      break;
    }
  }
  return 0;
}
