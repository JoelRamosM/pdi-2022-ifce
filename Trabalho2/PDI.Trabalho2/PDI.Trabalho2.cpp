// PDI.Trabalho2.cpp : This file contains the 'main' function. Program execution begins and ends there.
//
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>

using  namespace  cv;
using  namespace  std;

class WatershedSegmenter {
private:
	cv::Mat markers;
public:
	void setMarkers(cv::Mat& markerImage)
	{
		markerImage.convertTo(markers, CV_32SC1);
	}

	cv::Mat process(cv::Mat& image)
	{
		cv::watershed(image, markers);
		markers.convertTo(markers, CV_8U);
		return markers;
	}
};

auto read_img(string doc_name) -> Mat
{
	const Mat img = imread("./images/" + doc_name);
	return img;
}
auto open_and_save(string doc_name) -> void
{
	const Mat img = imread("./images/" + doc_name);
	const auto window_name = doc_name;
	namedWindow(doc_name, WINDOW_NORMAL);
	imshow(window_name, img);
	imwrite("./output/" + doc_name, img);
}

auto apply_sobel(Mat image)->Mat
{
	Mat  src, src_gray, grad;
	int ddepth = CV_16S;
	int ksize = 3;
	int scale = 1;
	int delta = 1;

	GaussianBlur(image, src, Size(3, 3), 0, 0, BORDER_DEFAULT);

	cvtColor(src, src_gray, COLOR_BGR2GRAY);

	Mat grad_x, grad_y;
	Mat abs_grad_x, abs_grad_y;
	Sobel(src_gray, grad_x, ddepth, 1, 0, ksize, scale, delta, BORDER_DEFAULT);
	Sobel(src_gray, grad_y, ddepth, 0, 1, ksize, scale, delta, BORDER_DEFAULT);
	convertScaleAbs(grad_x, abs_grad_x);
	convertScaleAbs(grad_y, abs_grad_y);

	addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
	return grad;
}

auto apply_laplace(Mat image)->Mat
{
	Mat  src, src_gray, grad;
	int ddepth = CV_16S;
	int ksize = 3;
	int scale = 1;
	int delta = 1;

	GaussianBlur(image, src, Size(3, 3), 0, 0, BORDER_DEFAULT);

	cvtColor(src, src_gray, COLOR_BGR2GRAY); // Convert the image to grayscale
	Mat abs_dst, abs_dst2;
	Laplacian(src_gray, grad, ddepth, ksize, scale, delta, BORDER_DEFAULT);
	// converting back to CV_8U
	convertScaleAbs(grad, abs_dst);
	convertScaleAbs(src_gray, abs_dst2);
	Mat result;
	addWeighted(abs_dst2, 0.5, abs_dst, 0.5, 0, result);
	return result;
}

auto sum(Mat image1, Mat image2)-> Mat
{
	Mat  src_gray, a, b, result;
	cvtColor(image1, src_gray, COLOR_BGR2GRAY);
	convertScaleAbs(src_gray, a);
	convertScaleAbs(image2, b);
	addWeighted(a, 0.5, b, 0.5, 0, result);
	return result;
}

auto gamma_correction(Mat image, float alpha /*< Simple contrast control */, float beta /*< Simple brightness control */)-> Mat
{
	Mat new_image = Mat::zeros(image.size(), image.type());

	for (int y = 0; y < image.rows; y++) {
		for (int x = 0; x < image.cols; x++) {
			for (int c = 0; c < image.channels(); c++) {
				new_image.at<Vec3b>(y, x)[c] =
					saturate_cast<uchar>(alpha * image.at<Vec3b>(y, x)[c] + beta);
			}
		}
	}

	return new_image;
}

auto dilate_image(const Mat src, int dilatation_size, int iterations = 1)->Mat
{
	Mat dilated, tst;
	bitwise_not(src, tst);
	const Mat element = getStructuringElement(MORPH_RECT, Size(dilatation_size, dilatation_size), Point(-1, -1));
	dilate(tst, dilated, element, Point(-1, -1), iterations);
	bitwise_not(dilated, dilated);
	return dilated;
}

auto erode_image(const Mat src, const int erosion_size)->Mat
{
	const auto erosion_type = MORPH_RECT;
	Mat eroded;
	const Mat element = getStructuringElement(erosion_type,
		Size(2 * erosion_size + 1, 2 * erosion_size + 1),
		Point(erosion_size, erosion_size));
	erode(src, eroded, element);
	return eroded;
}

auto filtro_media(const Mat image, int k_len = 3)->Mat
{
	Mat src_gray;
	Mat result = Mat::zeros(image.size(), image.type());
	cvtColor(image, src_gray, COLOR_BGR2GRAY);
	GaussianBlur(src_gray, result, Size(k_len, k_len), 0, 0, BORDER_REPLICATE);

	return result;
}

auto aplica_threshold(Mat image)-> Mat
{
	Mat result;
	adaptiveThreshold(image, result, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 5);
	//threshold(image, result, thresh, 255, THRESH_BINARY);
	return  result;
}

auto aplicar_crescimento_de_regiao(Mat image)-> Mat
{
	auto center = Point(image.size().width / 2, image.size().height / 2);
	Mat mask;
	floodFill(image, mask, center, Scalar(0), 0, Scalar(0), Scalar(0), FLOODFILL_MASK_ONLY << 8);
	bitwise_not(image, image);
	return mask;
}

auto aplicar_watershed(Mat image)->Mat
{
	Mat src = image.clone();
	src.convertTo(src, COLOR_GRAY2BGR);
	cvtColor(src, src, COLOR_GRAY2BGR);
	Mat blank(src.size(), CV_8U, Scalar(0xFF));
	Mat dest;

	Mat markers(src.size(), CV_8U, Scalar(-1));
	markers(Rect(0, 0, src.cols, 5)) = Scalar::all(1);
	//top rectangle
	markers(Rect(0, 0, src.cols, 5)) = Scalar::all(1);
	//bottom rectangle
	markers(Rect(0, src.rows - 5, src.cols, 5)) = Scalar::all(1);
	markers(Rect(0, src.rows - 15, src.cols, 5)) = Scalar::all(1);
	//left rectangle
	markers(Rect(0, 0, 5, src.rows)) = Scalar::all(1);
	//right rectangle
	markers(Rect(src.cols - 5, 0, 5, src.rows)) = Scalar::all(1);
	//centre rectangle
	int centreW = src.cols / 4;
	int centreH = src.rows / 4;
	markers(Rect((src.cols / 2) - (centreW / 2), (src.rows / 2) - (centreH / 2), centreW, centreH)) = Scalar::all(2);
	markers.convertTo(markers, CV_32SC1);
	watershed(src, markers);
	markers.convertTo(markers, CV_8U);
	Mat mask;
	convertScaleAbs(markers, mask, 1, 0);
	threshold(mask, mask, 1, 255, THRESH_BINARY);
	bitwise_and(src, src, dest, mask);

	dest.convertTo(dest, CV_8U);

	//imshow("final_result", dest);
	//imshow("markers", markers);
	return dest;
}

auto aplicar_transformacao_hough(Mat image)-> Mat
{
	Mat src, dst, cdst;
	src = image.clone();
	// Edge detection
	Canny(src, dst, 50, 200, 3);
	// Copy edges to the images that will display the results in BGR
	cvtColor(dst, cdst, COLOR_GRAY2BGR);
	auto cdstP = cdst.clone();
	// Standard Hough Line Transform
	vector<Vec2f> lines; // will hold the results of the detection
	HoughLines(dst, lines, 1, CV_PI / 180, 150, 0, 0); // runs the actual detection
	// Draw the lines
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);
	}
	vector<Vec4i> linesP; // will hold the results of the detection
	HoughLinesP(dst, linesP, 1, CV_PI / 180, 50, 50, 10); // runs the actual detection
	// Draw the lines
	for (size_t i = 0; i < linesP.size(); i++)
	{
		Vec4i l = linesP[i];
		line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
	}
	return cdstP;
}

int main()
{
	Mat src, image, blured, binary, dilated, eroded, bordas;
	auto image_name = "doc4.jpg";
	src = read_img(image_name);
	int width = 600;
	int height = 800;
	/*std::cout << "Width: " << width << endl;
	std::cout << "Height: " << height << endl;*/

	resize(src, image, Size(width, height));
	blured = filtro_media(image, 3);
	binary = aplica_threshold(blured);
	dilated = dilate_image(binary, 5, 2);
	eroded = erode_image(binary, 5);
	bitwise_not(dilated - eroded, bordas);
	bordas = dilate_image(bordas, 3);

	auto flood = bordas.clone();
	aplicar_crescimento_de_regiao(flood);
	auto watershed_result = aplicar_watershed(blured);
	auto hough = aplicar_transformacao_hough(binary);

	namedWindow("Original Image", WINDOW_AUTOSIZE);

	imshow("Original Image", image);
	imshow("Limiarização - Média", binary);
	imshow("Dilatacao", dilated);
	imshow("Bordas", bordas);
	imshow("Crescimento de	Região", flood);
	imshow("Watershed", watershed_result);
	imshow("Hough", hough);

	string output_path = "./output/";
	imwrite(output_path + "original_" + image_name, image);
	imwrite(output_path + "limiarizacao_" + image_name, binary);
	imwrite(output_path + "bordas_" + image_name, bordas);
	imwrite(output_path + "crescimento-regiao_" + image_name, flood);
	imwrite(output_path + "watershed_" + image_name, watershed_result);
	imwrite(output_path + "hough_" + image_name, hough);

	waitKey();
	destroyAllWindows();
}