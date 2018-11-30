
#pragma once

#include <Eigen/Dense>
#include "nn_base.hpp"

#ifndef _DEBUG
#define NDEBUG
#endif

using Eigen::MatrixXd;


extern MatrixXd Logistic(const MatrixXd &input);

extern MatrixXd ReLU(const MatrixXd &input);

extern MatrixXd ArcTangent(const MatrixXd &input);

extern MatrixXd HyperbolicTangent(const MatrixXd &input);

template<int inverseDerivativeOfNegative>
MatrixXd LeakyReLU(const MatrixXd &input)
{
	return input.unaryExpr([](const double elem)
	{
		return elem < 0.0 ? elem / inverseDerivativeOfNegative : elem;
	});
}

extern MatrixXd SoftPlus(const MatrixXd &input);

extern MatrixXd SoftMax(const MatrixXd &input);

extern MatrixXd SolveDerivative(const MatrixXd &in, ActivationFunction func);

extern MatrixXd SolveDerivative2(const MatrixXd &in, ActivationFunction func);