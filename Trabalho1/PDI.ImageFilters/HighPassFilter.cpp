#pragma once
#include <iostream>
#include <opencv2/core.hpp>

using namespace cv;

void filtro_laplaciano(const Mat* image, Mat* result, int k_len = 3)
{
	auto* kernel = new float* [k_len];
	const int hood_factor = k_len / 2;
	for (int i = 0; i < k_len; ++i)
	{
		kernel[i] = new float[k_len];
		for (int j = 0; j < k_len; ++j) {
			if (j == hood_factor + 1 && i == hood_factor + 1)
				kernel[i][j] = -8;
			else
				kernel[i][j] = 1;
		}
	}
	Mat borders_wk;
	copyMakeBorder(*image, borders_wk, hood_factor, hood_factor, hood_factor, hood_factor, BORDER_REPLICATE);
	for (int row = 0; row < image->rows; ++row)
	{
		for (int col = 0; col < image->cols; ++col)
		{
			auto sum_window = 0;
			for (int i = 0; i < k_len; ++i)
				for (int j = 0; j < k_len; ++j) {
					sum_window += kernel[i][j] * borders_wk.at<uchar>(row + i, col + j);
				}
			result->at<uchar>(row, col) = sum_window / 9;
		}
	}

	std::cout << "Laplace";
}

void apply_2d_filter(const Mat* image, Mat* result, int k_len, float(* const kernel_ph)[3], float(* const kernel_pv)[3], const int hood_factor)
{
	Mat borders_wk;
	copyMakeBorder(*image, borders_wk, hood_factor, hood_factor, hood_factor, hood_factor, BORDER_CONSTANT, Scalar(0));
	for (int row = 0; row < image->rows; ++row)
	{
		for (int col = 0; col < image->cols; ++col)
		{
			auto sum_ph = 0;
			auto sum_pv = 0;
			for (int i = 0; i < k_len; ++i)
				for (int j = 0; j < k_len; ++j) {
					sum_ph += kernel_ph[i][j] * borders_wk.at<uchar>(row + i, col + j);
					sum_pv += kernel_pv[i][j] * borders_wk.at<uchar>(row + i, col + j);
				}
			result->at<uchar>(row, col) = sqrt((pow(sum_ph, 2)) + (pow(sum_pv, 2)));
		}
	}
}

void filtro_prewitt(const Mat* image, Mat* result)
{
	auto k_len = 3;

	const auto kernel_ph = new float[3][3]
	{
		{-1,0,1},
		{-1,0,1},
		{-1,0,1},
	};
	const auto kernel_pv = new float[3][3]
	{
		{-1,-1,-1},
		{0,0,0},
		{1,1,1},
	};

	const int hood_factor = 1;

	apply_2d_filter(image, result, k_len, kernel_ph, kernel_pv, hood_factor);

	std::cout << "Prewitt";
}

void filtro_sobel(const Mat* image, Mat* result)
{
	auto k_len = 3;

	const auto kernel_ph = new float[3][3]
	{
		{-1,0,1},
		{-2,0,2},
		{-1,0,1},
	};
	const auto kernel_pv = new float[3][3]
	{
		{-1,-2,-1},
		{0,0,0},
		{1,2,1},
	};

	const int hood_factor = 1;

	apply_2d_filter(image, result, k_len, kernel_ph, kernel_pv, hood_factor);

	std::cout << "Prewitt";
}