#pragma once
#include <opencv2/core.hpp>

cv::Mat histograma(cv::Mat imagem);
void limiarizacao(const cv::Mat* image, cv::Mat* result, const int threshold = 0);
void multilimiarizacao(const cv::Mat* image, cv::Mat* result, const std::multiset<std::tuple<int, int> >& multiset_of_tuples);
