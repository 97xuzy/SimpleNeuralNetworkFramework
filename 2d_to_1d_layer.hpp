/**
 * /file 2d_to_1d_layer.hpp
 *
 */

#pragma once

#ifndef _DEBUG
#define NDEBUG
#endif

#include "nn_base.hpp"

#define input_data_ input_.at(0)->GetOutput()

/**
 * Convert 2D to 1D.<br>
 *
 * ## Input
 * This layer can only have 1 2D input
 *
 * ## Output
 * This layer has 1 output, which is the 2D input in a RowMajor layout
 *
 */
class TwoDOneDLayer : public Layer
{
protected:
	/**
	 * Since this layer can only have 1 input, calling this method the second will override the result of the first one.
	 * \param in Attempt to add layer \p in as a input to the current layer.
	 * \sa Layer::AddInputSrc
	 */
	void AddInputSrc(Layer *in) override;

	/**
	 * Initialize the layer, after setting the input. (e.g. setting up the weight for FC layer)
	 * \sa Layer::Init
	 */
	void Init() override;

public:
	/**
	 * Construcotr
	 * \param name used to initialize the Layer with this \p name
	 * \sa Layer::Layer
	 */
	TwoDOneDLayer(const std::string &name);

	/**
	 * Perform feedforward operation on the network.
	 * Assuming the compute order is generated and is correct.
	 * \sa Layer::Compute
	 */
	void Compute() override;

	/**
	 * \param d_loss_to_output Error Signal of this layer, computed in the Layer::BackProp.
	 * \sa Layer::UpdateWeights
	 */
	void UpdateWeights(const MatrixXd &error_signal, double learning_rate = 1.0) override;

	/**
	 * \return a vector which each elements represents a input to the current layer. If current layer is an instance of InputLayer, will return a empty vector
	 * \sa Layer::GetInputs
	 */
	std::vector<Layer *> GetInputs() override;

	/**
	 * Since \ref TwoDOneDLayer does not perform any computation, only transformation, it only apply the same transformation to the error signal in reverse (1D to 2D).
	 * \return \p d_loss_to_output in 2D (same dimension as the input of this layer)
	 * \sa Layer::BackPropToPrev
	 */
	MatrixXd BackPropToPrev(const MatrixXd &d_loss_to_output, Layer *prev_layer) override;

	/**
	 * \return The output Matrix of the layer, the content of the Matrix is only valid after the layer being computed
	 * \sa Layer::GetOutput
	 */
	MatrixXd GetOutput() override;
};


/*
TwoDOneDLayer(const std::string &name)
*/
TwoDOneDLayer::TwoDOneDLayer(const std::string &name) : Layer(name)
{
}

/*
void AddInputSrc(Layer *in)
*/
void TwoDOneDLayer::AddInputSrc(Layer *in)
{
	if (!this->input_.empty())
		this->input_.clear();

	this->input_.push_back(in);
}

/*
void Init()
*/
void TwoDOneDLayer::Init()
{

}

/*
void :Compute()
*/
void TwoDOneDLayer::Compute()
{

}

/*
void UpdateWeights(const MatrixXd &d_loss_to_output, double learning_rate)
*/
void TwoDOneDLayer::UpdateWeights(const MatrixXd &error_signal, double learning_rate)
{

}

/*
std::vector<Layer *> GetInputs()
*/
std::vector<Layer *> TwoDOneDLayer::GetInputs()
{
	return this->input_;
}

/*
MatrixXd BackPropToPrev(const MatrixXd &d_loss_to_output, Layer *prev_layer)
*/
MatrixXd TwoDOneDLayer::BackPropToPrev(const MatrixXd &error_signal, Layer *prev_layer)
{
	return Eigen::Map<const MatrixXd>(error_signal.data(), this->input_data_.rows(), this->input_data_.cols()).matrix();
}

/*
MatrixXd GetOutput()
*/
MatrixXd TwoDOneDLayer::GetOutput()
{
	return Eigen::Map<MatrixXd>(this->input_data_.data(), 1, this->input_data_.rows() * this->input_data_.cols()).matrix();
}

#undef input_data_
