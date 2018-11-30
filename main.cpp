
#ifndef _DEBUG
#define NDEBUG
#endif

#include "nn_base.hpp"
#include "full_connect_layer.hpp"
#include "convolution_layer.hpp"
#include "2d_to_1d_layer.hpp"
#include "activation_func.hpp"
#include "io_helper.hpp"

#include <opencv2/opencv.hpp>
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <iostream>

#include <ctime>

using std::string;
using std::vector;
using namespace Eigen;

using std::cout;
using std::endl;

void Network1(int num_loop);
void Network2(int num_loop);
void Network3(int num_loop);
void Network4(int num_loop);

inline MatrixXd readMatrixFromImage(const std::string &inputPath);
void MeasurePerformance(NerualNetwork &nn);


int main(int argc, char *argv[])
{

	//std::srand((unsigned int)time(0));

	int num_loop = 3;
	if (argc > 1)
	{
		num_loop = std::atoi(argv[1]);
	}

	Network4(num_loop);

	//MeasurePerformance(nn);

	return 0;
}

void Network2(int num_loop)
{
	constexpr int output_size = 1;
	NNFactory factory;

	factory.ConstructNN();

	factory.AddInputLayer(new InputLayer("INPUT_1", 10, 10));
	factory.AddLayer(new TwoDOneDLayer(string("2D1D1")), false, "INPUT_1");
	factory.AddLayer(new FullConnectLayer<output_size, Logistic>(string("FC1")), true, "2D1D1");
	//factory.AddLayer(new FullConnectLayer<output_size, Logistic>(string("FC2")), true, "FC1");

	NerualNetwork nn = factory.Finish();

	MatrixXd img_m = readMatrixFromImage("train_img\\001_1.png");
	cout << img_m << "\n\n" << std::endl;

	nn.TakeInput("INPUT_1", std::move(img_m));

	MatrixXd target = MatrixXd::Constant(1, output_size, 0.876);

	for (int i = 0; i < num_loop; i++)
	{
		nn.Compute();
		MatrixXd output = nn.GetOutput();

		cout << "***** OUTPUT *****\n" << output << std::endl;
		cout << "***** TARGET *****\n" << target << std::endl;
		double avg_diff = (target - output).sum() / output.rows() / output.cols();
		cout << "***** AVG_DIFF *****\n" << avg_diff << std::endl;
		cout << "\n\n";

		nn.BackProp(target);
		nn.UpdateWeight();
	}
	cout << "\n\nTraining Finished\n\n" << std::endl;
}

void Network4(int num_loop)
{
	constexpr int output_size = 1;
	NNFactory factory;

	factory.ConstructNN();

	factory.AddInputLayer(new InputLayer("INPUT_1", 10, 10));
	factory.AddLayer(new ConvolutionalLayer<3, 1>(string("CV1")), false, "INPUT_1");
	factory.AddLayer(new TwoDOneDLayer(string("2D1D1")), false, "CV1");
	factory.AddLayer(new FullConnectLayer<50, LeakyReLU<100>>(string("FC1")), false, "2D1D1");
	factory.AddLayer(new FullConnectLayer<50, HyperbolicTangent>(string("FC2")), false, "FC1");
	factory.AddLayer(new FullConnectLayer<output_size, Logistic>(string("FC3")), true, "FC2");

	NerualNetwork nn = factory.Finish();

	MatrixXd img_m = readMatrixFromImage("train_img\\001_1.png");
	cout << img_m << "\n\n" << std::endl;

	nn.TakeInput("INPUT_1", std::move(img_m));

	MatrixXd target = MatrixXd::Constant(1, output_size, 0.876);
	cout << "target: \n" << target << "\n" << std::endl;

	for (int i = 0; i < num_loop; i++)
	{
		nn.Compute();
		MatrixXd output = nn.GetOutput();

		cout << "***** OUTPUT *****\n" << output << std::endl;
		cout << "***** TARGET *****\n" << target << std::endl;
		double avg_diff = (target - output).sum() / output.rows() / output.cols();
		cout << "***** AVG_DIFF *****\n" << avg_diff << std::endl;
		//MatrixXd error = CWiseProduct((output - target).cwiseQuotient(target), 100);

		cout << "\n\n";

		nn.BackProp(target);
		nn.UpdateWeight();
	}
	cout << "\n\nTraining Finished\n\n" << std::endl;

}

void MeasurePerformance(NerualNetwork &nn)
{
	std::clock_t start;
	double duration;

	start = std::clock();
	constexpr int num = 200;
	for (int i = 0; i < num; i++)
	{
		nn.Compute();
	}

	duration = (std::clock() - start) / (double)CLOCKS_PER_SEC;

	std::cout << "time per " << num << " run: " << duration << " sec" << '\n';
	std::cout << "run per sec: " << 1.0 / (duration / num) << " run" << '\n';
}

