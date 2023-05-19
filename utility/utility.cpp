#include "net.h"
#include <algorithm>

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
