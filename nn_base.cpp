/**
 * /file nn_base.cpp
 *
 */

#include "nn_base.hpp"

using std::make_unique;
using std::map;
using std::string;
using std::unique_ptr;
using std::vector;

/****************************************
 * CNNFactory
 ****************************************/

NNFactory::NNFactory()
{
}

void NNFactory::ConstructNN()
{
    // Allocate the object
	this->nn_ = std::unique_ptr<NerualNetwork>(new NerualNetwork());
}

void NNFactory::AddInputLayer(InputLayer *layer)
{
    // Create a unique_ptr for the newly added layer
    std::unique_ptr<Layer> ptr(layer);

    // Insert into NN
    this->nn_.get()->all_layers_.emplace(layer->name_, std::move(ptr));
    // Add to the input layer vector of the NN
    this->nn_.get()->input_.push_back(layer);
}

void NNFactory::AddLayer(Layer *layer, bool as_output, const std::string &input_name)
{
    assert(layer != nullptr);
    assert(!input_name.empty());

    // Create a unique_ptr for the newly added layer
    std::unique_ptr<Layer> ptr(layer);

    // Insert into NN
    this->nn_.get()->all_layers_.emplace(layer->name_, std::move(ptr));

	// Make sure input_name does exist in all_layers_
	assert(this->nn_.get()->all_layers_.count(input_name) == 1);

    Layer *input = this->nn_.get()->all_layers_.at(input_name).get();

    // Add the new layer as a 'next' layer to layer with input_name
    input->next_.push_back(layer);

    // Set the layer with input_name as input of the new layer
    layer->AddInputSrc(input);

	// Set the layer as the Output Layer of the network if specified
	if (as_output)
	{
		this->nn_.get()->output_ = layer;
	}
}

template<typename ...T>
void NNFactory::AddLayer(Layer *layer, bool as_output, const std::string &input_name, T&... inputs_names)
{
	// Set the layer as the Output Layer of the network if specified
	if (as_output)
	{
		this->nn_.get()->output_ = layer;
	}

	// Set Output operation only need to perform once, thus pass false
	this->AddLayer(layer, false, input_name);
	this->AddLayer(layer, false, inputs_names...);
}

NerualNetwork &&NNFactory::Finish()
{
    NerualNetwork *nn = this->nn_.get();

	if (nn->output_ == nullptr)
	{
		throw std::runtime_error("No Output present for the Network");
	}

	// Generate Compute Order
	nn->GenerateComputeOrder();

	// Init Layer
	for (Layer *layer : nn->compute_order_)
	{
		layer->Init();
	}

	// Release Ownership
    this->nn_.release();
    return std::move(*nn);
}

/****************************************
 * NerualNetwork
 ****************************************/
NerualNetwork::NerualNetwork()
{
}

NerualNetwork::NerualNetwork(NerualNetwork &&other)
{
    this->all_layers_ = std::move(other.all_layers_);
    this->input_ = std::move(other.input_);
    this->compute_order_ = std::move(other.compute_order_);
	this->output_ = other.output_;
	this->d_loss_to_output_ = std::move(other.d_loss_to_output_);
}

void NerualNetwork::GenerateComputeOrder()
{
	// Lamda that checks if a layer with given name has been computed
    auto layer_computed = [&](const string &name) -> bool {
		// Check if layer takes input from InputLayers
		for (Layer *layer : this->input_)
		{
			// If the input with the specified name is a InputLayer
			if (layer->name_.compare(name) == 0)
			{
				return true;
			}
		}
        for (Layer *layer : this->compute_order_)
        {
            // If the input with the specified name is already in the compute order
            if (layer->name_.compare(name) == 0)
            {
                return true;
            }
        }
        return false;
    };
	std::vector<Layer*> remaining_layer;
	for (auto &kvpair : this->all_layers_)
	{
		remaining_layer.push_back(kvpair.second.get());
	}

	// Keep running untill all that's left are InputLayers
	while (remaining_layer.size() > this->input_.size())
	{
		for (auto it = remaining_layer.begin(); it != remaining_layer.end();)
		{
			Layer *layer = *it;

			// Check if all inputs of the layer has been computed
			bool all_input_computed = true;
			for (Layer *input : layer->GetInputs())
			{
				if (!layer_computed(input->name_))
				{
					all_input_computed = false;
					break;
				}
			}

			auto is_input_layer = [](Layer *layer, std::vector<Layer*> input_layers) {
				for (Layer * input : input_layers)
				{
					if (input->GetName().compare(layer->GetName()) == 0)
					{
						return true;
					}
				}
				return false;
			};

			if (all_input_computed	// If all input of the layer has been computed, add to compute order
				&& !is_input_layer(layer, this->input_)	// If layer is a InputLayer, skip it, since InputLayer does not perform any computation
				)
			{
				this->compute_order_.push_back(layer);
				it = remaining_layer.erase(it);
			}	
			else
			{
				it++;
			}
		}
	}

}

void NerualNetwork::Compute()
{
    for (Layer *layer : this->compute_order_)
    {
        layer->Compute();
    }
}

/*
void NerualNetwork::BackProp(const MatrixXd &target_output)
{
	// Target and the Output should have the same dimension
	assert(this->output_->GetOutput().rows() == target_output.rows());
	assert(this->output_->GetOutput().cols() == target_output.cols());

	auto square_error_derivative = [](const MatrixXd &target_output, const MatrixXd &output) {
		return (target_output - output);
	};

	//
	// Compute for output layer
	//
	MatrixXd loss_dervative = square_error_derivative(target_output, this->output_->GetOutput());
	MatrixXd error_signal = this->output_->ComputeErrorSignal(loss_dervative, true, &target_output);
	std::cout << "Error Signal (output):\n" << error_signal << "\n\n";
	this->d_loss_to_output_.emplace(this->output_->GetName(), std::move(error_signal));

	//
	// Compute for hidden layers
	//
	if (this->compute_order_.size() > 1)	// If there is any hidden layer
	for (long i = this->compute_order_.size() - 2; i >= 0; i--)
	{
		Layer *layer = this->compute_order_.at(i);

		MatrixXd error_signal_from_next;

		// Propgate from all of the next layer, (multiply the Weight from next layer)

		//for (Layer *next : layer->next_)
		//{
		//	// if empty, use = rather than +=
		//	if (error_signal_from_next.rows() == 0)
		//	{
		//		error_signal_from_next = next->BackPropToPrev(this->d_loss_to_output_.at(next->GetName()), layer);
		//	}
		//	else
		//	{
		//		error_signal_from_next += next->BackPropToPrev(this->d_loss_to_output_.at(next->GetName()), layer);
		//	}
		//}

		error_signal_from_next = this->d_loss_to_output_.at(layer->next_.at(0)->GetName());

		MatrixXd error_signal = layer->ComputeErrorSignal(error_signal_from_next);
		assert(layer->GetOutput().rows() == error_signal.rows());
		assert(layer->GetOutput().cols() == error_signal.cols());
		this->d_loss_to_output_.emplace(layer->GetName(), std::move(error_signal));
	}

}
*/

void NerualNetwork::BackProp(const MatrixXd &target_output)
{
	// Target and the Output should have the same dimension
	assert(this->output_->GetOutput().rows() == target_output.rows());
	assert(this->output_->GetOutput().cols() == target_output.cols());

	auto square_error_derivative = [](const MatrixXd &target_output, const MatrixXd &output) {
		return (target_output - output);
	};

	//
	// dL/dO for Output Layer
	//
	MatrixXd d_loss_to_output = square_error_derivative(target_output, this->output_->GetOutput());
	this->d_loss_to_output_.emplace(this->output_->GetName(), d_loss_to_output);


	// If there is No hidden layer
	if (this->compute_order_.size() == 1)
	{
		return;
	}

	//
	// dL/dO for Inner Layer
	//
	for (long i = this->compute_order_.size() - 2; i >= 0; i--)
	{
		Layer *layer = this->compute_order_.at(i);

		MatrixXd d_loss_to_output_of_next;

		// Propgate from all of the next layer, (multiply the Weight from next layer)

		for (Layer *next : layer->next_)
		{
			// if empty, use = rather than +=
			if (d_loss_to_output_of_next.rows() == 0)
			{
				d_loss_to_output_of_next = next->BackPropToPrev(this->d_loss_to_output_.at(next->GetName()), layer);
			}
			else
			{
				d_loss_to_output_of_next += next->BackPropToPrev(this->d_loss_to_output_.at(next->GetName()), layer);
			}
		}

		this->d_loss_to_output_.emplace(layer->GetName(), d_loss_to_output_of_next);
	}
}

void NerualNetwork::UpdateWeight()
{
	// Update weight for all layers in compute_order_
	for(Layer *layer : this->compute_order_)
	{
		layer->UpdateWeights(this->d_loss_to_output_.at(layer->GetName()));
	}

	// Clear Error Signal, since it is no longer needed
	this->d_loss_to_output_.clear();
}

void NerualNetwork::TakeInput(const std::string &name, MatrixXd &&data)
{
    for (Layer *layer : this->input_)
    {
        if (layer->GetName().compare(name) == 0)
        {
            ((InputLayer *)layer)->TakeInputData(std::move(data));
            break;
        }
    }
}

MatrixXd NerualNetwork::GetOutput()
{
	return this->output_->GetOutput();
}

Layer *NerualNetwork::layer(const std::string &name)
{
    return this->all_layers_.at(name).get();
}

/****************************************
 * Layer
 ****************************************/
Layer::Layer(const std::string &name)
{
    this->name_ = name;
}

string Layer::GetName()
{
    return this->name_;
}

/****************************************
 * InputLayer
 ****************************************/
InputLayer::InputLayer(const std::string &name, int rows, int cols) : Layer(name)
{
    this->rows_ = rows;
    this->cols_ = cols;
	this->data_ = MatrixXd::Zero(rows, cols);
}

void InputLayer::TakeInputData(MatrixXd &&data)
{
    if (data.rows() != this->rows_ || data.cols() != this->cols_)
    {
        throw std::runtime_error("dimension of the data is incosistent with the layer");
    }
    this->data_ = std::move(data);
}

MatrixXd InputLayer::GetOutput()
{
    return this->data_;
}



