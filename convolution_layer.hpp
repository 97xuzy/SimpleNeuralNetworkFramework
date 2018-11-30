/**
 * /file convolution_layer.hpp
 *
 */
#pragma once

#ifndef _DEBUG
#define NDEBUG
#endif

#include "nn_base.hpp"

#define input_data_ input_.at(0)->GetOutput()

/**
 * \class ConvolutionalLayer
 * Currently no padding is implemented.<br>
 * Input of this layer should be 2D Matrix.<br>
 * Output of this layer will be 2D Matrix.<br>
 *
 * \tparam kernel_size Size of the kernel, (e.g.kernel_size == 3, then kernel has a size of 3X3).
 * Currently kernel_size can only be odd numbers.
 * \tparam stride Steps in which the kernel moves on the input
 *
 *
 *
 */
template <int kernel_size, int stride>
class ConvolutionalLayer : public Layer
{
protected:
	
	/**
	 * \var kernel_
	 * Kernel of the \ref ConvolutionalLayer, will be a sqaure (same rows as cols).
	 */
	MatrixXd kernel_;

	/**
	 * \var output_
	 */
	MatrixXd output_;

	/**
	 * \param in Layer that will be the input of this layer, ouput of \p in must be a 1 dimension Matrix (1 X ?)
	 * \ref ConvolutionalLayer instance can only have 1 input, calling this method the second time will overwrite the result of the first time.
	 * \sa Layer::AddInputSrc
	 */
	void AddInputSrc(Layer *in) override;

	/**
	 * \sa Layer::Init
	 */
	void Init() override;
public:
	/**
	 * Constructor.<br>
	 * kernel is randomed.
	 * \param name The name of the \ref ConvolutionalLayer
	 * \sa Layer::Layer
	 */
	ConvolutionalLayer(const std::string &name);

	/**
	 * Constructor<br>
	 * \param name The name of the \ref ConvolutionalLayer
	 * \param kernel ConvolutionalLayer will use the given kernel instead of random one.
	 *  \sa Layer::Layer
	 */
	ConvolutionalLayer(const std::string &name, const MatrixXd &kernel);

	/**
	 * \sa Layer::Compute
	 */
	void Compute() override;

	/**
	 * \sa Layer::UpdateWeights
	 */
	void UpdateWeights(const MatrixXd &error_signal, double learning_rate = 1.0) override;

	/**
	 * \sa Layer::GetInputs
	 */
	std::vector<Layer *> GetInputs() override;

	/**
	 * \sa Layer::BackPropToPrev
	 */
	MatrixXd BackPropToPrev(const MatrixXd &d_loss_to_output, Layer *prev_layer) override;

	/**
	 * \sa Layer::GetOutput
	 */
	MatrixXd GetOutput() override;
};


/*
ConvolutionalLayer(const std::string &name)
*/
template <int kernel_size, int stride>
ConvolutionalLayer<kernel_size, stride>::ConvolutionalLayer(const std::string &name) : Layer(name)
{
	assert(kernel_size > 0);

	// Limit the kernel_size to odd numbers
	assert(kernel_size % 2 == 1);

	// Init kernel
	this->kernel_ = MatrixXd::Random(kernel_size, kernel_size);
	std::cout << "kernel:\n" << this->kernel_ << "\n\n";
}

/*
ConvolutionalLayer(const std::string &name, const MatrixXd &kernel)
*/
template <int kernel_size, int stride>
ConvolutionalLayer<kernel_size, stride>::ConvolutionalLayer(const std::string &name, const MatrixXd &kernel) : Layer(name)
{
	// Init kernel
	this->kernel_ = kernel;
}

/*
void AddInputSrc(Layer *in)
*/
template <int kernel_size, int stride>
void ConvolutionalLayer<kernel_size, stride>::AddInputSrc(Layer *in)
{
	// ConvolutionLayer only takes 1 input
	ASSERT_SINGLE_INPUT();

	// The input need to be larger than the kernel
	assert(in->GetOutput().rows() > kernel_size);
	assert(in->GetOutput().cols() > kernel_size);

	// If input has already been set, overwrite the input by clear the existing input.
	if (!this->input_.empty())
		this->input_.clear();

	this->input_.push_back(in);

	// Init the shape of the output_
	this->output_ = MatrixXd::Zero(in->GetOutput().rows() - kernel_size + 1, in->GetOutput().cols() - kernel_size + 1);
}

/*
void Init()
*/
template <int kernel_size, int stride>
void ConvolutionalLayer<kernel_size, stride>::Init()
{

}

/*
void Compute()
*/
template <int kernel_size, int stride>
void ConvolutionalLayer<kernel_size, stride>::Compute()
{
	int output_row = 0;
	int output_col = 0;

	// Init the shape of the output_
	this->output_ = MatrixXd::Zero(this->input_data_.rows() - kernel_size + 1, this->input_data_.cols() - kernel_size + 1);

	//std::cout << "input "<< this->name_ << ":\n" << this->input_data_ << "\n\n";
	// row and col represents the position of the top-left element of the kernel in the input_data_
	for (int row = 0, output_row = 0; row < this->input_data_.rows() - kernel_size + 1; row += stride, output_row++)
	{
		for (int col = 0, output_col = 0; col < this->input_data_.cols() - kernel_size + 1; col += stride, output_col++)
		{
			this->output_(output_row, output_col) = this->input_data_.block(row, col, kernel_size, kernel_size).cwiseProduct(this->kernel_).sum();
		}
	}
	//std::cout << "output " << this->name_ << ":\n" << this->output_ << "\n\n";
	
}

/*
void UpdateWeights(const MatrixXd &d_loss_to_output, double learning_rate = 1.0)
*/
template <int kernel_size, int stride>
void ConvolutionalLayer<kernel_size, stride>::UpdateWeights(const MatrixXd &d_loss_to_output, double learning_rate)
{
	// d_loss_to_output must have the same shape as the output of this layer
	assert(d_loss_to_output.rows() == this->output_.rows());
	assert(d_loss_to_output.cols() == this->output_.cols());

	//std::cout << "d_loss_to_output(" << this->name_ << "): \n" << d_loss_to_output << "\n==============================\n";

	// dL/dW = dL/dO *dO/dW
	// dO/dW = input

	MatrixXd delta_kernel = MatrixXd::Zero(this->kernel_.rows(), this->kernel_.cols());

	// row and col here represent the position in the output
	for (int row = 0; row < d_loss_to_output.rows(); row++)
	{
		for (int col = 0; col < d_loss_to_output.cols(); col++)
		{
			// Multiply the input data with its corresponding the elements in the d_loss_to_output (which has the same dimension as the output)
			delta_kernel += CWiseProduct(this->input_data_.block(row * stride, col * stride, kernel_size, kernel_size), d_loss_to_output(row, col));
		}
	}

	this->kernel_.noalias() = this->kernel_ + delta_kernel;
}

/*
std::vector<Layer *> GetInputs()
*/
template <int kernel_size, int stride>
std::vector<Layer *> ConvolutionalLayer<kernel_size, stride>::GetInputs()
{
	return this->input_;
}

template <int kernel_size, int stride>
MatrixXd ConvolutionalLayer<kernel_size, stride>::BackPropToPrev(const MatrixXd &d_loss_to_output, Layer *prev_layer)
{
#ifdef _DEBUG
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

	MatrixXd new_d_loss_to_output = MatrixXd::Zero(prev_layer->GetOutput().rows(), prev_layer->GetOutput().cols());

	for (int row = 0; row < this->output_.rows(); row++)
	{
		for (int col = 0; col < this->output_.cols(); col++)
		{
			new_d_loss_to_output.block(row * stride, col * stride, kernel_size, kernel_size) += CWiseProduct(this->kernel_, d_loss_to_output(row, col));
		}
	}

	return new_d_loss_to_output;
}

/*
MatrixXd GetOutput()
*/
template <int kernel_size, int stride>
MatrixXd ConvolutionalLayer<kernel_size, stride>::GetOutput()
{
	return this->output_;
}

#undef input_data_