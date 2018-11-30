/**
 * /file full_connect_layer.hpp
 *
 */
#pragma once
#ifndef _DEBUG
#define NDEBUG
#endif

#include "nn_base.hpp"

#pragma message("remove this header after debug")
#include "activation_func.hpp"


#define input_data_ input_.at(0)->GetOutput()
/**
 * Fully Connected Layer.
 * Each input element and each output element are connected.
 * An activation function is used after applying the linear weight.
 *
 * \tparam output_size The dimension of the ouput would be 1 X \p output_size
 * \tparam activa_func A function pointer to the \ref ActivationFunction
 * \sa ActivationFunction
 *
 * ## Input
 * The input must be a 1D, but in the form of a Eigen::MatrixXd, meaning input data has a shape of (1 X ?).<br>
 * The shape of the input is known and certain after calling FullConnectLayer::Init().<br>
 * Input should not be changed after calling FullConnectLayer::Init(), since the weight has been generated<br>
 *
 * ## Output
 * The output is also 1D, has a shape of (1 X \p output_size), the shape of the output is known at compile time, since it is passed in as a template argument.
 *
 * ## BackProp
 *
 */
template <int output_size, ActivationFunction activa_func>
class FullConnectLayer : public Layer
{
protected:
	/**
	 * \var weight_
	 * weight_ will have a size of (input_cols X #output_size)
	 */
	MatrixXd weight_;

	/**
	 * \var output_
	 * Output of the Layer.
	 */
	MatrixXd output_;

	/**
	 * \param in Layer that will be the input of this layer, ouput of \p in must be a 1 dimension Matrix (1 X ?)
	 * #FullConnectLayer instance can only have 1 input, calling this method the second time will overwrite the result of the first time.
	 * \sa Layer::AddInputSrc
	 */
	void AddInputSrc(Layer *in) override;

	/**
	 * Initailize \p weight_ of the Layer.
	 * \p weight_ is init according to the dimension of the output and input of this layer, thus input must be set before this method is called.
	 * \sa Layer::Init
	 */
	void Init() override;
public:
	/**
	 * \param name Name of the layer.
	 */
	FullConnectLayer(const std::string& name);

	/**
	 * Perform feedforward operation, do a matrix product of #input_ and #weight_.
	 * \sa Layer::Compute
	 */
	void Compute() override;

	/**
	 * \sa Layer::UpdateWeights
	 */
	void UpdateWeights(const MatrixXd &error_signal, double learning_rate = 1.0) override;

	/**
	 * \return A vector of size 1, or 0 (if #input_ is not set), which contains the input of this layer.
	 * \sa Layer::GetInputs
	 */
	std::vector<Layer *> GetInputs() override;

	/**
	 * \sa Layer::BackPropToPrev
	 */
	MatrixXd BackPropToPrev(const MatrixXd &d_loss_to_output, Layer *prev_layer) override;

	/**
	 * \return The output Matrix of the layer, the content of the Matrix is only valid after FullConnectLayer::Compute() of this layer being called.
	 * \sa Layer::GetOutput
	 */
	MatrixXd GetOutput() override;
};


/*
FullConnectLayer(const std::string &name)
*/
template <int output_size, ActivationFunction activa_func>
FullConnectLayer<output_size, activa_func>::FullConnectLayer(const std::string &name) : Layer(name)
{
	this->output_ = MatrixXd::Zero(1, output_size);
}


/*
void AddInputSrc(Layer *in)
*/
template <int output_size, ActivationFunction activa_func>
void FullConnectLayer<output_size, activa_func>::AddInputSrc(Layer *in)
{
	// FullConnectLayer only has 1 input.
	// So the input_ should be empty or size of 1
	ASSERT_SINGLE_INPUT();

	// FC Layer takes input of (1X?) Matrix
	assert(in->GetOutput().rows() == 1);

	// If input has already been set, overwrite the input by clear the existing input.
	if (! this->input_.empty())
		this->input_.clear();

	this->input_.push_back(in);
}

/*
void Init()
*/
template <int output_size, ActivationFunction activa_func>
void FullConnectLayer<output_size, activa_func>::Init()
{
	assert(this->input_.size() == 1);

	// Create Weights
	this->weight_ = MatrixXd::Random(this->input_data_.cols(), output_size);
	//CWiseQuotient(this->weight_, 100);
}

/*
void Compute()
*/
template <int output_size, ActivationFunction activa_func>
void FullConnectLayer<output_size, activa_func>::Compute()
{
	//std::cout << "net of " << this->name_ << "\n" << this->input_data_ * this->weight_ << std::endl;
	this->output_ = activa_func(this->input_data_ * this->weight_);
	//std::cout << "output of " << this->name_ << "\n" << this->output_ << std::endl;
}

/*
void UpdateWeights(const MatrixXd &d_loss_to_output)
*/
template <int output_size, ActivationFunction activa_func>
void FullConnectLayer<output_size, activa_func>::UpdateWeights(const MatrixXd &d_loss_to_output, double learning_rate)
{
	#pragma message("f'(output) vs f'(net)")
	// Use f'(output) may lead to faster convegence over f'(net)
	// Not sure why
	MatrixXd activa_derivative = SolveDerivative2(this->input_data_ * this->weight_, Logistic); // f'(net)
	//MatrixXd activa_derivative = SolveDerivative2(this->output_, Logistic); // f'(output)

	#pragma message("weight + delta vs - delta")
	/*
	//std::cout.setf( std::ios::fixed, std::ios::floatfield );
	//cout << "input to deriva:\n" << this->input_data_ * this->weight_ << endl;
	//cout << "activa_deriva:\n" << activa_derivative << endl;

	MatrixXd delta = CWiseProduct(this->input_data_.transpose() * d_loss_to_output.cwiseProduct(activa_derivative), learning_rate);
	cout << "delta:\n" << delta << endl;
	this->weight_.noalias() = this->weight_ + delta;
	*/
	this->weight_.noalias() = this->weight_ + CWiseProduct(this->input_data_.transpose() * d_loss_to_output.cwiseProduct(activa_derivative), learning_rate);
}


/*
std::vector<Layer *> GetInputs()
*/
template <int output_size, ActivationFunction activa_func>
std::vector<Layer *> FullConnectLayer<output_size, activa_func>::GetInputs()
{
	return this->input_;
}

/*
MatrixXd BackPropToPrev(const MatrixXd &d_loss_to_output, Layer *prev_layer = nullptr)
*/
template <int output_size, ActivationFunction activa_func>
MatrixXd FullConnectLayer<output_size, activa_func>::BackPropToPrev(const MatrixXd &d_loss_to_output, Layer *prev_layer)
{
#ifdef _DEBUG
	assert(prev_layer != nullptr);
	// Check if prev_layer is a input to this layer
	bool is_input = false;
	for (Layer * in : this->input_)
	{
		if (in == prev_layer)
		{
			is_input = true;
			break;
		}
	}
	if (!is_input)
	{
		throw std::runtime_error("prev_layer is NOT a input to this layer");
	}
#endif // _DEBUG

	// Propagate the dL/dO from this layer to the input of current layer

	return (this->weight_ * d_loss_to_output.transpose()).transpose();

	/*
	MatrixXd new_error_signal = PerColSum(PerRowCoeffWiseProduct(this->weight_, error_signal).transpose());
	//MatrixXd new_error_signal = (this->weight_ * d_loss_to_output.transpose()).transpose();
	//MatrixXd new_error_signal = d_loss_to_output;

	assert(new_error_signal.rows() == prev_layer->GetOutput().rows());
	assert(new_error_signal.cols() == prev_layer->GetOutput().cols());
	return new_error_signal;
	*/
}

/*
MatrixXd GetOutput()
*/
template <int output_size, ActivationFunction activa_func>
MatrixXd FullConnectLayer<output_size, activa_func>::GetOutput()
{
	return this->output_;
}

#undef input_data_
