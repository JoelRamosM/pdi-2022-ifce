#pragma once
#include <opencv2/core.hpp>

void filtro_laplaciano(const cv::Mat* image, cv::Mat* result, int k_len = 3);
void filtro_prewitt(const cv::Mat* image, cv::Mat* result);
void filtro_sobel(const cv::Mat* image, cv::Mat* result);
