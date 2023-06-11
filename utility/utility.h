#ifndef UTILITY_H
#define UTILITY_H

#include <algorithm>
#include <iostream>
#include <net.h>

void printImage(const ncnn::Mat &mat);
void printMinMaxValues(const ncnn::Mat &mat, float *max_val, float *min_val);

#endif
