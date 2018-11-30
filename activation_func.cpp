
#include "activation_func.hpp"
#include <Eigen/Dense>

MatrixXd Logistic(const MatrixXd &input)
{
	// Logisitic Function
	// f(x) = 1 / (1 + e^-x)
	MatrixXd temp = MatrixXd::Zero(input.rows(), input.cols()) - input;
	temp = temp.array().exp();
	temp.noalias() += MatrixXd::Ones(input.rows(), input.cols());
	return temp.cwiseInverse();
}

MatrixXd ReLU(const MatrixXd &input)
{
	return input.unaryExpr([](const double elem)
	{
		return elem < 0.0 ? 0.0 : elem;
	});
}

MatrixXd ArcTangent(const MatrixXd &input)
{
	// tanh(x)
	// (e^x - e^-x) / (e^x + e^-x)
	return input.array().atan();
}

MatrixXd HyperbolicTangent(const MatrixXd &input)
{
	// tanh(x)
	return input.array().tanh();
}

MatrixXd SoftPlus(const MatrixXd &input)
{
	// Soft Plus
	// log(1 + e^x)
	//return (input.array().exp() + 1).log();
	return input.unaryExpr([](const double elem)
	{
		if (elem > 40.0)
			return elem;
		else
			return log(1 + exp(elem));
	});
}

MatrixXd SoftMax(const MatrixXd &input)
{
	return (input.array().exp() / input.array().exp().sum()).matrix();
}



MatrixXd SolveDerivative(const MatrixXd &in, ActivationFunction func)
{
	// solve for the derivative of func at in
	const double divisor = 1000;
	MatrixXd diff_m = CWiseQuotient(in, divisor);

	MatrixXd derivative = (func(in + diff_m) - func(in - diff_m)).cwiseQuotient(CWiseProduct(diff_m, 2));
	return derivative;
};

MatrixXd SolveDerivative2(const MatrixXd &in, ActivationFunction func)
{
	// solve for the derivative of func at in

	//MatrixXd diff_m = MatrixXd::Constant(in.rows(), in.cols(), 0.000001);
	const double divisor = 1000;
	MatrixXd diff_m = CWiseQuotient(in, divisor);

	MatrixXd derivative = (
		func(in - CWiseProduct(diff_m, 2)) - CWiseProduct(func(in - diff_m), 8)
		+ CWiseProduct(func(in + diff_m), 8) - func(in + CWiseProduct(diff_m, 2))
		).cwiseQuotient(CWiseProduct(diff_m, 12));
	return derivative;
};