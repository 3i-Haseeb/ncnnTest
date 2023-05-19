#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdio.h>

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "net.h"

void printImage(const ncnn::Mat &mat) {
  int width = mat.w;
  int height = mat.h;
  int channels = mat.c;

  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < channels; ++c) {
        float value = mat.channel(c).row(h)[w];
        std::cout << value << " ";
      }
    }
    std::cout << std::endl;
  }
}

void printMinMaxValues(const ncnn::Mat &mat) {
  int size = mat.total();
  int width = mat.w;
  int height = mat.h;
  int channels = mat.c;

  float maxVal = 0;
  float minVal = 0;

  std::cout << minVal << std::endl;
  std::cout << maxVal << std::endl;

  for (int h = 0; h < height; ++h) {
    for (int w = 0; w < width; ++w) {
      for (int c = 0; c < channels; ++c) {
        float value = mat.channel(c).row(h)[w];
        maxVal = std::max(maxVal, value);
        minVal = std::min(minVal, value);
      }
    }
  }

  std::cout << "Maximum value: " << maxVal << std::endl;
  std::cout << "Minimum value: " << minVal << std::endl;
}

int main(int argc, char **argv) {
  // Load image
  std::string imagepath("./images/car.png");
  cv::Mat img = cv::imread(imagepath, cv::IMREAD_COLOR);
  if (img.empty()) {
    std::cerr << "Unable to read image file " << imagepath << std::endl;
    return -1;
  }

  // Specify the names of all classes for image classification
  std::vector<std::string> classes = {"plane", "car",  "bird", "cat",
                                      "deer",  "dog",  "frog", "horse",
                                      "ship",  "truck"};

  // Load NCNN model
  ncnn::Net net;
  int ret = net.load_param("./models/optimized_model.param");
  if (ret)
    std::cerr << "Failed to load model parameters!" << std::endl;
  ret = net.load_model("./models/optimized_model.bin");
  if (ret)
    std::cerr << "Failed to load model weights!" << std::endl;

  // Convert image data to ncnn format
  // Opencv image is bgr, model also expects bgr
  ncnn::Mat input = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR,
                                           img.cols, img.rows);

  // Preprocess the image
  const float mean_vals[3] = {0.5f * 255.f, 0.5f * 255.f, 0.5f * 255.f};
  const float norm_vals[3] = {1 / 0.5f / 255.f, 1 / 0.5f / 255.f,
                              1 / 0.5f / 255.f};
  input.substract_mean_normalize(mean_vals, norm_vals);
  // printMinMaxValues(input);

  // Inference
  ncnn::Extractor extractor = net.create_extractor();
  extractor.set_num_threads(4);
  extractor.input("input", input);
  ncnn::Mat output;
  extractor.extract("output", output);

  // Flatten
  ncnn::Mat out_flattened = output.reshape(output.w * output.h * output.c);
  std::vector<float> scores;
  scores.resize(out_flattened.w);
  for (int j = 0; j < out_flattened.w; j++) {
    scores[j] = out_flattened[j];
  }

  // Prediciton based on score
  std::string pred_class =
      classes[std::max_element(scores.begin(), scores.end()) - scores.begin()];
  std::cout << "Predicted class is " << pred_class << "." << std::endl;

  // Save and visualize image results
  cv::imwrite("./results/result.png", img);
  cv::namedWindow("Input image", cv::WINDOW_NORMAL);
  cv::imshow(pred_class, img);
  while (true) {
    char key = cv::waitKey(0);
    if (key == 27 || key == 'q') {
      std::cout << "Exiting." << std::endl;
      break; // exit the loop if Esc or q key is pressed
    }
  }

  cv::destroyAllWindows(); // Destroy all windows
  return 0;
}
