#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "LowPassFilters.h"
#include "HighPassFilter.h"
#include "ImageOperations.h"

using  namespace  cv;
using  namespace  std;

void open_and_save()
{
	const Mat img = imread("./images/lena.png");
	const auto window_name = "Lena";
	namedWindow(window_name, WINDOW_AUTOSIZE);
	imshow(window_name, img);
	imwrite("./output/original_lena.png", img);
}

void open_rgb_gscale_and_save()
{
	Mat img_gray;
	const Mat img_rgb = imread("./images/lena.png");
	const auto window_original_name = "Lena";
	const auto window_gscale_name = "Lena Gray-Scale";
	namedWindow(window_original_name, WINDOW_AUTOSIZE);
	imshow(window_original_name, img_rgb);
	imwrite("./output/original_lena.png", img_rgb);
	cvtColor(img_rgb, img_gray, COLOR_RGB2GRAY);

	namedWindow(window_gscale_name, WINDOW_AUTOSIZE);
	imshow(window_gscale_name, img_gray);
	imwrite("./output/gray_lena.png", img_gray);
}

void run_media_filter()
{
	Mat img_blur;
	Mat img_gray = imread("./images/lena.png", IMREAD_GRAYSCALE);
	const auto window_original_name = "Lena";
	const auto window_avg_name = "Lena AVG (OpenCV Blur) Filter";

	namedWindow(window_original_name, WINDOW_AUTOSIZE);
	imshow(window_original_name, img_gray);
	imwrite("./output/original_lena.png", img_gray);

	blur(img_gray, img_blur, Size(5, 5), Point(-1, -1));

	namedWindow(window_avg_name, WINDOW_AUTOSIZE);
	imshow(window_avg_name, img_blur);
	imwrite("./output/blur_avg-filter_lena.png", img_blur);

	Mat img_custom_avg = Mat(img_gray.rows, img_gray.cols, IMREAD_GRAYSCALE, Scalar(1));
	filtro_media(&img_gray, &img_custom_avg, 5);
	imshow("Custom AVG", img_custom_avg);
	imwrite("./output/custom_avg-filter_lena.png", img_custom_avg);
}

void run_median_filter()
{
	Mat img_median;
	const Mat img_gray = imread("./images/lena.png", IMREAD_GRAYSCALE);
	const auto window_original_name = "Lena";
	const auto window_median_name = "Lena Median (OpenCV Blur) Filter";

	namedWindow(window_original_name, WINDOW_AUTOSIZE);
	imshow(window_original_name, img_gray);
	imwrite("./output/original_lena.png", img_gray);

	medianBlur(img_gray, img_median, 5);

	namedWindow(window_median_name, WINDOW_AUTOSIZE);
	imshow(window_median_name, img_median);
	imwrite("./output/median-filter_lena.png", img_median);

	auto img_custom_median = Mat(img_gray.rows, img_gray.cols, IMREAD_GRAYSCALE, Scalar(0));
	filtro_mediana(&img_gray, &img_custom_median, 3);
	imshow("Custom Median", img_custom_median);
	imwrite("./output/custom_median-filter_lena.png", img_custom_median);
}

void run_gaussian_filter()
{
	Mat img_gau;
	const Mat img_gray = imread("./images/lena.png", IMREAD_GRAYSCALE);
	const auto window_original_name = "Lena";
	const auto window_gaussian_name = "Lena Gaussian (OpenCV Blur) Filter";

	namedWindow(window_original_name, WINDOW_AUTOSIZE);
	imshow(window_original_name, img_gray);
	imwrite("./output/original_lena.png", img_gray);

	GaussianBlur(img_gray, img_gau, Size(5, 5), 3);

	namedWindow(window_gaussian_name, WINDOW_AUTOSIZE);
	imshow(window_gaussian_name, img_gau);
	imwrite("./output/gaussian-filter_lena.png", img_gau);

	auto img_custom_gaussian = Mat(img_gray.rows, img_gray.cols, IMREAD_GRAYSCALE, Scalar(0));
	filtro_gaussiano(&img_gray, &img_custom_gaussian, 5, 3);
	imshow("Custom Gaussian", img_custom_gaussian);
	imwrite("./output/custom_gaussian-filter_lena.png", img_custom_gaussian);
}

void run_laplacian_filter()
{
	Mat img_laplacian;
	const Mat img_gray = imread("./images/lena.png", IMREAD_GRAYSCALE);
	const auto window_original_name = "Lena";
	const auto window_laplace_name = "Lena Laplacian (OpenCV Edge) Filter";

	namedWindow(window_original_name, WINDOW_AUTOSIZE);
	imshow(window_original_name, img_gray);
	imwrite("./output/original_lena.png", img_gray);

	Laplacian(img_gray, img_laplacian, CV_64F);

	namedWindow(window_laplace_name, WINDOW_AUTOSIZE);
	imshow(window_laplace_name, img_laplacian);

	imwrite("./output/laplacian-filter_lena.png", img_laplacian);

	auto img_custom_laplacian = Mat(img_gray.rows, img_gray.cols, IMREAD_GRAYSCALE, Scalar(0));
	filtro_laplaciano(&img_gray, &img_custom_laplacian, 3);
	imshow("Custom Laplacian", img_custom_laplacian);
	imwrite("./output/custom_laplacian-filter_lena.png", img_custom_laplacian);
}

void run_prewitt_filter()
{
	Mat img_laplacian;
	const Mat img_gray = imread("./images/lena.png", IMREAD_GRAYSCALE);
	const auto window_original_name = "Lena";
	namedWindow(window_original_name, WINDOW_AUTOSIZE);
	imshow(window_original_name, img_gray);
	imwrite("./output/original_lena.png", img_gray);

	auto img_custom_prewitt = Mat(img_gray.rows, img_gray.cols, IMREAD_GRAYSCALE, Scalar(0));
	filtro_prewitt(&img_gray, &img_custom_prewitt);
	imshow("Custom Prewitt", img_custom_prewitt);
	imwrite("./output/custom_prewitt-filter_lena.png", img_custom_prewitt);
}

void run_sobel_filter()
{
	Mat img_laplacian;
	const Mat img_gray = imread("./images/lena.png", IMREAD_GRAYSCALE);
	const auto window_original_name = "Lena";
	namedWindow(window_original_name, WINDOW_AUTOSIZE);
	imshow(window_original_name, img_gray);
	imwrite("./output/original_lena.png", img_gray);

	auto img_custom_prewitt = Mat(img_gray.rows, img_gray.cols, IMREAD_GRAYSCALE, Scalar(0));
	filtro_sobel(&img_gray, &img_custom_prewitt);
	imshow("Custom SOBEL", img_custom_prewitt);
	imwrite("./output/custom_sobel-filter_lena.png", img_custom_prewitt);
}

void run_histograma()
{
	const Mat img_gray = imread("./images/lena.png", IMREAD_GRAYSCALE);
	Mat equalized;
	equalizeHist(img_gray, equalized);

	auto hist = histograma(img_gray);
	auto hist_equalized = histograma(equalized);

	imshow("Lena - Original", img_gray);
	imshow("Lena - Equalizado", equalized);

	imshow("Lena - Histograma", hist);
	imshow("Lena - Hist. Equalizado", hist_equalized);
}

void run_limiarizacao()
{
	const Mat img_gray = imread("./images/lena.png", IMREAD_GRAYSCALE);
	auto img_limiar = Mat(img_gray.rows, img_gray.cols, IMREAD_GRAYSCALE, Scalar(0));

	limiarizacao(&img_gray, &img_limiar, 150);

	imshow("Lena - Original", img_gray);
	imshow("Lena - limirarizado", img_limiar);
}

void run_multilimiarizacao()
{
	const Mat img_gray = imread("./images/lena.png", IMREAD_GRAYSCALE);
	auto img_limiar = Mat(img_gray.rows, img_gray.cols, IMREAD_GRAYSCALE, Scalar(0));
	multiset<tuple<int, int> > thresholds;

	thresholds.insert(make_tuple(80, 255));
	thresholds.insert(make_tuple(150, 128));
	thresholds.insert(make_tuple(255, 0));

	multilimiarizacao(&img_gray, &img_limiar, thresholds);

	imshow("Lena - Original", img_gray);
	imshow("Lena - multilimirarizado", img_limiar);
}

int main()
{
	//run_histograma();
	run_limiarizacao();
	run_multilimiarizacao();
	/*run_prewitt_filter();
	run_sobel_filter();*/

	//run_laplacian_filter();
	//run_gaussian_filter();
	//run_median_filter();
	//run_media_filter();

	waitKey();
	destroyAllWindows();
	return 0;
}