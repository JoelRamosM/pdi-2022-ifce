#pragma once
#include <opencv2/core.hpp>

void filtro_media(const cv::Mat* image, cv::Mat* result, int k_len = 3);
void  filtro_mediana(const cv::Mat* image, cv::Mat* result, int k_len = 3);
void filtro_gaussiano(const cv::Mat* image, cv::Mat* result, int k_len = 3, double sigma = 1);
