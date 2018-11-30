
#include "io_helper.hpp"
#include <opencv2/opencv.hpp>

using Eigen::MatrixXd;

MatrixXd readMatrixFromImage(const std::string &inputPath)
{
	cv::Mat img = cv::imread(inputPath, CV_LOAD_IMAGE_GRAYSCALE);
	if (img.data == nullptr)
	{
		std::cerr << "Fail to read image" << std::endl;
		throw std::runtime_error("Fail to read image");
	}
	assert(img.rows == 10);
	assert(img.cols == 10);
	Eigen::Map<Eigen::Matrix<uchar, 10, 10>> map(img.data);

	return map.matrix().cast<double>();
}

