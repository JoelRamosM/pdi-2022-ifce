#include <iostream>
#include <opencv2/core.hpp>

using namespace cv;

void filtro_media(const Mat* image, Mat* result, int k_len = 3)
{
	auto* kernel = new float* [k_len];

	for (int i = 0; i < k_len; ++i)
	{
		kernel[i] = new float[k_len];
		for (int j = 0; j < k_len; ++j) {
			kernel[i][j] = 1;
		}
	}
	const int hood_factor = k_len / 2;
	Mat borders_wk;
	copyMakeBorder(*image, borders_wk, hood_factor, hood_factor, hood_factor, hood_factor, BORDER_REPLICATE);

	for (int row = 0; row < image->rows; ++row)
	{
		for (int col = 0; col < image->cols; ++col)
		{
			int sum_window = 0;
			for (int i = 0; i < k_len; ++i)
				for (int j = 0; j < k_len; ++j)
					sum_window += kernel[i][j] * borders_wk.at<unsigned char>(row + i, col + j);

			result->at<unsigned char>(row, col) = sum_window / (k_len * k_len);
		}
	}
}

void filter_median(const Mat* image, Mat* result, int k_len = 3)
{
	const int hood_factor = k_len / 2;
	Mat borders_wk;
	auto s = Scalar(0);
	copyMakeBorder(*image, borders_wk, hood_factor, hood_factor, hood_factor, hood_factor, BORDER_REPLICATE);
	for (int row = 0; row < image->rows; ++row) {
		for (int col = 0; col < image->cols; ++col)
		{
			auto sorted_flat = new int[k_len * k_len];
			int median_index = ((k_len * k_len) / 2) + 1;
			auto current = 0;

			for (int i = 0; i < k_len; ++i) {
				for (int j = 0; j < k_len; ++j) {
					sorted_flat[current] = borders_wk.at<unsigned char>(row + i, col + j);
					current++;
				}
			}

			std::sort(sorted_flat, 1 + (sorted_flat + sizeof sorted_flat));

			int median = sorted_flat[median_index];
			result->at<unsigned char>(row, col) = median;
		}
	}
}

void filtro_mediana(const Mat* image, Mat* result, int k_len = 3)
{
	filter_median(image, result);
	for (int i = 0; i < k_len - 1; ++i)
		filter_median(result, result);

	std::cout << "Median";
}

void gen_gaussian_kernel(int k_len, double** kernel, double sigma = 0)
{
	for (int i = 0; i < k_len; ++i)
	{
		kernel[i] = new double[k_len];
		for (int j = 0; j < k_len; ++j) {
			kernel[i][j] = 0;
		}
	}

	const double pi = 3.141592653589793238462643383279502884L /* pi */;
	const int hood_factor = k_len / 2;
	const double s = 2.0 * sigma * sigma;
	double sum = 0.0;
	for (int x = -hood_factor; x <= hood_factor; x++) {
		for (int y = -hood_factor; y <= hood_factor; y++) {
			double r = sqrt(x * x + y * y);
			int i = x + hood_factor;
			int j = y + hood_factor;
			kernel[i][j] = (exp(-(r * r) / s)) / (pi * s);
			sum += kernel[i][j];
		}
	}
	for (int i = 0; i < k_len; ++i)
		for (int j = 0; j < k_len; ++j)
			kernel[i][j] /= sum;
}

void filtro_gaussiano(const Mat* image, Mat* result, int k_len = 3, double sigma = 1)
{
	auto* kernel = new double* [k_len];

	gen_gaussian_kernel(k_len, kernel, sigma);

	const int hood_factor = k_len / 2;

	Mat borders_wk;
	copyMakeBorder(*image, borders_wk, hood_factor, hood_factor, hood_factor, hood_factor, BORDER_REPLICATE);

	for (int row = 0; row < image->rows; ++row)
	{
		for (int col = 0; col < image->cols; ++col)
		{
			int sum_window = 0;
			for (int i = 0; i < k_len; ++i)
				for (int j = 0; j < k_len; ++j)
					sum_window += kernel[i][j] * borders_wk.at<unsigned char>(row + i, col + j);

			result->at<unsigned char>(row, col) = sum_window;
		}
	}

	std::cout << "Gaussian";
}