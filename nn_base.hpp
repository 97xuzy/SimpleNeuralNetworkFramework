/**
 * /file nn_base.hpp
 *
 */
#pragma once

#ifndef _DEBUG
#define NDEBUG
#endif

#include <Eigen/Dense>
#include <vector>
#include <map>
#include <string>
#include <string_view>
#include <memory>
#include <functional>
#include <optional>

#pragma message("remove include of iostream")
#include <iostream>

class Layer;
class NerualNetwork;
class InputLayer;

using Eigen::MatrixXd;

typedef MatrixXd(*ActivationFunction)(const MatrixXd&);

#define ASSERT_SINGLE_INPUT() assert(this->input_.empty() || this->input_.size() == 1)

class NNFactory
{
	/**
     * Holds the \ref NerualNetwork object while constructing it.
     * \ref NNFactory has the ownership of the \ref NerualNetwork object during the process.
     */
	std::unique_ptr<NerualNetwork> nn_;

  public:
    /**
     *
     */
	NNFactory();

    /**
     * Starting the construting process.
     * \ref NerualNetwork object is created/allocated.
     */
	void ConstructNN();

	/**
	 * \param layer A \ref InputLayer to be added
	 */
	void AddInputLayer(InputLayer *layer);

	/**
	 * \param layer A non \ref InputLayer to be added to the network.
	 * \param input_name Name of the layer, in which the newly added layer takes input from.
	 */
    void AddLayer(Layer *layer, bool as_output, const std::string &input_name);

	/**
	 * \param layer A non \ref InputLayer to be added to the network
     * \param input_name Name of the layer, in which the newly added layer takes input from.
	 */
	template<typename ...T>
	void AddLayer(Layer *layer, bool as_output, const std::string &input_name, T&... inputs_names);

	/**
	 * Finish constructing, return/release-ownership of the \ref NerualNetwork object.
	 * \return A rvalue reference of the \ref NerualNetwork object.
	 */
	NerualNetwork&& Finish();

};

class NerualNetwork
{

private:
    /**
	 * \var all_layers_
     * All layers in the Network.
     * All layers are hold/owned by the unique)ptr in this map.
	 * std::string in the std::map are the Layer::name_ of the layer.
     */
    std::map<std::string, std::unique_ptr<Layer>> all_layers_;

    /**
	 * \var input_
     * A vector of pointers to all \ref InputLayer in this network
     */
    std::vector<Layer*> input_;

	/**
	 * \var compute_order_
	 * Order of computing layer.
	 * Layer at 0th in the vector will be compute first.
	 */
    std::vector<Layer *> compute_order_;

	/**
	 * \var output_
	 * Pointer to a Layer whose output is the output of the network.
	 */
	Layer *output_;

	/**
	 * \var d_loss_to_output_
	 * Derivative of Loss to the output of each layer (dL/dO), computed during Layer::BackProp.
	 * std::string in the std::map are the Layer::name_ of the layer.
	 */
	std::map <std::string, MatrixXd> d_loss_to_output_;

	/**
	 * \param dimen each element in the vector represent a Matrix input,
	 * the Matrix will have the dimension specified by the std::pair<int row, int col>.
	 */
	//NerualNetwork(const std::vector<std::pair<int, int>> &dimen);
	NerualNetwork();

	/**
	 * Copy Constructor, deleted.
	 * NerualNetwork instance can only be moved but not copyed.
	 */
	NerualNetwork(const NerualNetwork&) = delete;

	/**
	 * Generate Compute Order
	 */
    void GenerateComputeOrder();

  public:
	/**
	 * Move Constructor
	 */
	NerualNetwork(NerualNetwork&&);

	/**
	 * Perform a feedforward on the network.
	 * the order of which computation is performed depend on the #compute_order_
	 */
    void Compute();

	/**
	 * Perform backpropgation on the network, compute and store the Error Signal.
	 * \pararm target_output The target output to initiate the backprop
	 */
    void BackProp(const MatrixXd &target_output);

	/**
	 * Use the Error Signal to update each layer accordingly.
	 */
	void UpdateWeight();

	/**
	 * Pass input \p data to a input specified by the \p name
	 * \param name name of the input layer which takes in the data
	 * \param data Matrix data to be passed in, the Matrix need to satisify the dimension requirement of the input layer
	 */
	void TakeInput(const std::string &name, MatrixXd &&data);

	/**
	 * /return A matrix that represent the output of the network.
	 */
	MatrixXd GetOutput();

	/**
     * layer
	 * \param name \p name_ of the layer
     */
	Layer* layer(const std::string &name);

    friend NNFactory;
};

/**
 * An abstract class, inherited by other Layer.
 * Layers only have 1 output ( 1 single Matrix ).
 * Layers may have more than 1 input(e.g. \ref DepthConcatLayer), or no input (e.g. \ref InputLayer).
 * Use multiple layer to achieve depth.
 */
class Layer
{
  protected:
    /**
	 * \var name_
     * name of the layer.
     * no 2 layer can have the same name, as the name will be use as a index.
     */
    std::string name_;

	/**
	 * \var input_
	 * A vector of pointer to \ref Layer, points to \ref Layer that this layer takes input from.
	 */
    std::vector<Layer *> input_;

    /**
	 * \var next_
     * Layers that take input from this layer.
     */
    std::vector<Layer *> next_;

    /**
     * \param in Attempt to add layer \p in as a input to the current layer, behavior of this method is varied between different type of layers.
     */
    virtual void AddInputSrc(Layer *in) = 0;

    /**
     * Initialize the layer, after setting the input. (e.g. setting up the weight for FC layer)
     */
    virtual void Init() = 0;

  public:
    /**
     * Construcotr
	 * \param name used to initialize the Layer with this \p name
     */
    Layer(const std::string &name);

    /**
     * Perform feedforward operation on the network.
	 * Assuming the compute order is generated and is correct.
     */
    virtual void Compute() = 0;

	/**
	 * \param d_loss_to_output Error Signal of this layer, computed in the Layer::BackProp.
	 */
	virtual void UpdateWeights(const MatrixXd &error_signal, double learning_rate = 1.0) = 0;

	/**
	 * \return a vector which each elements represents a input to the current layer. If current layer is an instance of InputLayer, will return a empty vector
	 */
    virtual std::vector<Layer *> GetInputs() = 0;

	/**
	 * Backprop\Convert the d_loss_to_output of the calling layer to \p prev_layer.
	 *
	 * \param d_loss_to_output Partial derivative of the Loss Function to the Output of this layer, since \p Layer itself does not store the Derivative.
	 * \param prev_layer Layer to propgate to, use to disambuious which layer to propgate to, as the calling layer might have more than 1 input.
	 * This parameter must NOT be nullptr, even if the calling layer only have 1 input.
	 * \return Error Signal of the \p prev_layer.
	 */
	virtual MatrixXd BackPropToPrev(const MatrixXd &d_loss_to_output, Layer *prev_layer) = 0;

	/**
	 * \return The output Matrix of the layer, the content of the Matrix is only valid after the layer being computed 
	 */
    virtual MatrixXd GetOutput() = 0;

	/**
	 * \return the name of the layer
	 */
    std::string GetName();

    friend NNFactory;
    friend NerualNetwork;
};

/**
 * 1
 */
class TrainingLabel
{
public:

};


/**
 * Input Layer.
 * When taking MatrixXd data, the data is std::move, rather than copied
 */
class InputLayer : public Layer
{
protected:
	/**
	 * number of rows of the input data.
	 * data takes in by this layer must satisify this requirement
	 */
	int rows_;

	/**
	 * number of cols of the input data.
	 * data takes in by this layer must satisify this requirement
	 */
	int cols_;

	/**
	 * holds the data after taking in.
	 * When taking in data, the \ref Eigen::MatrixXd is being moved rather than copyed.
	 */
	MatrixXd data_;

	/**
	 * Perform no operation, Will throw exception when called.
	 * \param in
	 * \throw std::runtime_error InputLayer should NOT have any input
	 */
	void AddInputSrc(Layer *in) override { throw std::runtime_error("InputLayer should NOT have any input"); }

	/**
	 * Empty, no operation is being performed.
	 */
	void Init() override {}
public:
	/**
	 * \param name Initialize the \ref InputLayer with this \p name
	 * \param rows Initialize the number of rows with \p rows of \ref InputLayer
	 * \param cols Initialize the number of columns with \p cols of \ref InputLayer
	 */
	InputLayer(const std::string &name, int rows, int cols);

	/**
	 *
	 */
	void TakeInputData(MatrixXd &&data);

	/**
	 *
	 */
	void Compute() override {}

	/**
	 * Perform no operation
	 */
	void UpdateWeights(const MatrixXd &error_signal, double learning_rate = 1.0) override {}

	/**
	 * Should Not be called.
	 * \return Always return a empty vector
	 */
	std::vector<Layer *> GetInputs() override { return this->input_; }

	/**
	 * Perform no operation.
	 * \sa Layer::BackPropToPrev
	 * \throw std::runtime_error InputLayer does NOT have any input
	 */
	MatrixXd BackPropToPrev(const MatrixXd &error_signal, Layer *prev_layer = nullptr) override
	{
		throw std::runtime_error("InputLayer does NOT have any input");
		return MatrixXd::Zero(1, 1);
	}

	/**
	 * \return Output of the layer, which is just the input data provided by InputLayer::TakeInputData() function call
	 * \sa InputLayer::TakeInputData
	 */
	MatrixXd GetOutput() override;	// defined in source file
};

/**
 * Do a cwiseProduct between \p m_row and every row of \p m1
 * \param m1 Matrix with the same number of columns as \p m_row
 * \param m_row Matrix with only 1 row
 */
inline MatrixXd PerRowCoeffWiseProduct(const MatrixXd &m1, const MatrixXd &m_row)
{
	assert(m_row.rows() == 1);
	assert(m1.cols() == m_row.cols());

	MatrixXd m1_copy = m1;
	for (int r = 0; r < m1.rows(); r++)
	{
		auto row = m1_copy.block(r, 0, 1, m1.cols());
		row.noalias() = row.cwiseProduct(m_row);
	}
	return m1_copy;
}

/**
 * Do a cwiseProduct between \p m_col and every column of \p m1
 * \param m1 Matrix with the same number of rows as \p m_col
 * \param m_col Matrix with only 1 column
 */
inline MatrixXd PerColCoeffWiseProduct(const MatrixXd &m1, const MatrixXd &m_col)
{
	assert(m_col.cols() == 1);
	assert(m1.rows() == m_col.rows());

	MatrixXd m1_copy = m1;
	for (int c = 0; c < m1.cols(); c++)
	{
		auto col = m1_copy.block(0, c, m1.rows(), 1);
		col.noalias() = col.cwiseProduct(m_col);
	}
	return m1_copy;
}

/**
 * Sum up each row.
 * From a AXB Matrix to a AX1 Matrix
 * \param m Matrix to be per-row-sumed
 * \return Result of the per row sum.
 */
inline MatrixXd PerRowSum(const MatrixXd &m)
{
	MatrixXd result = MatrixXd::Zero(m.rows(), 1);
	for (int r = 0; r < m.rows(); r++)
	{
		auto row = m.block(r, 0, 1, m.cols());
		//result(0, c) = col.sum();
		double sum = row.sum();
		result(r, 0) = sum;
	}
	return result;
}

/**
 * Sum up each column.
 * From a AXB Matrix to a 1XB Matrix
 * \param m Matrix to be per-col-sumed
 * \return Result of the per column sum.
 */
inline MatrixXd PerColSum(const MatrixXd &m)
{
	MatrixXd result = MatrixXd::Zero(1, m.cols());
	for (int c = 0; c < m.cols(); c++)
	{
		auto col = m.block(0, c, m.rows(), 1);
		//result(0, c) = col.sum();
		double sum = col.sum();
		result(0, c) = sum;
	}
	return result;
}

/**
 * \param m Matrix to be multiplied with the \p constant
 * \param constant Constant to be multiplied with the Matrix \p m
 */
inline MatrixXd CWiseProduct(const MatrixXd &m, double constant)
{
	return m.cwiseProduct(MatrixXd::Constant(m.rows(), m.cols(), constant));
}

/**
 * \param m Matrix to be divided with the \p constant
 * \param constant Constant used to divide the Matrix \p m
 */
inline MatrixXd CWiseQuotient(const MatrixXd &m, double constant)
{
	return m.cwiseQuotient(MatrixXd::Constant(m.rows(), m.cols(), constant));
}

