#include <set>
#include <opencv2/core.hpp>

using  namespace  cv;

Mat histograma(Mat imagem) {
	int vet[256], x, y, num;
	int ma = 0;

	for (x = 0; x < 256; x++)
		vet[x] = 0;

	for (x = 0; x < imagem.rows; x++) {
		for (y = 0; y < imagem.cols; y++) {
			num = 0;
			num = imagem.at<uchar>(x, y);
			vet[num]++;
		}
	}

	for (x = 0; x < 256; x++) {
		if (ma == 0)
			ma = vet[x];
		if (vet[x] > ma)
			ma = vet[x];
	}

	int max = ma / 200;
	for (x = 0; x < 256; x++) {
		vet[x] = vet[x] / max;
	}

	Mat_<Vec3b> histogram(200, 256, CV_8UC3);
	for (x = 0; x < histogram.rows; x++) {
		for (y = 0; y < histogram.cols; y++) {
			if (x >= histogram.rows - vet[y]) {
				histogram(x, y)[0] = 0;
				histogram(x, y)[1] = 0;
				histogram(x, y)[2] = 0;
			}
			else {
				histogram(x, y)[0] = 255;
				histogram(x, y)[1] = 255;
				histogram(x, y)[2] = 255;
			}
		}
	}
	return histogram;
}

void limiarizacao(const Mat* image, Mat* result, const int threshold = 0)
{
	for (int row = 0; row < image->rows; ++row)
	{
		for (int col = 0; col < image->cols; ++col)
		{
			const auto value = image->at<unsigned char>(row, col);
			result->at<unsigned char>(row, col) = value > threshold ? value : 0;
		}
	}
}

void multilimiarizacao(const Mat* image, Mat* result, const std::multiset<std::tuple<int, int> >& multiset_of_tuples)
{
	for (int row = 0; row < image->rows; ++row)
	{
		for (int col = 0; col < image->cols; ++col)
		{
			const auto value = image->at<unsigned char>(row, col);
			auto newValue = value;
			auto prevThreshold = 0;
			for (auto x : multiset_of_tuples)
			{
				if (value >= prevThreshold && value < std::get<0>(x))
				{
					newValue = std::get<1>(x);
					break;
				}
				prevThreshold = std::get<0>(x);
			}
			result->at<unsigned char>(row, col) = newValue;
		}
	}
}