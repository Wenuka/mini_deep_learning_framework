from plot_helper import plot_data
from general_helper import NNcompute_nb_errors, generate_disc_set, analyse_model
from NN_classes import NNSequential, NNRelu, NNTanh, NNLinear

from torch import set_grad_enabled
set_grad_enabled(False)



def main() :
	"""
	Main Function: to compute performance of Two Models with different non-linear functions for different losses
	Namely, this will generate 4 graphs, for losses MSE and MAE, for each non-linear type Relu and Tanh
	:return: nothing but four graphs and write the number of errors in the terminal
	"""

	### To Set Constants and Parameters ###
	train_size = 1000
	mini_batch_size = 100
	with_one_hot_encoding = True
	label_1_in_center = True
	early_stoping = True
	epoch_count = 250
	look_back_count_early_stoping = 100
	learning_rate = 100/(epoch_count * train_size)
	loss_type_array = ['MSE','MAE']
	mini_batch_size = min(train_size,mini_batch_size)
	plot_inputs = False

	### Generate training and testing data ###
	train_input, train_target = generate_disc_set(train_size,one_hot_encoding=with_one_hot_encoding,label_1_in_center=label_1_in_center)
	test_input, test_target = generate_disc_set(train_size, one_hot_encoding=with_one_hot_encoding,label_1_in_center=label_1_in_center)
	if plot_inputs:
		plot_data(train_input, train_target, test_input, test_target, title="Train and Test Samples before Normaliation") ## uncomment to plot generated test and train inputs and targets
	
	### Normalize the input data ###
	mean, std = train_input.mean(), train_input.std()
	train_input.sub_(mean).div_(std)
	test_input.sub_(mean).div_(std)
	if plot_inputs:
		plot_data(train_input, train_target, test_input, test_target, title="Train and Test Samples after Normaliation") ## uncomment to plot noramlized test and train inputs and targets
	in_feature_size = len(train_input[0])
	out_feature_size = len(train_target[0])

	### Creating and analysing the models ###
	nnmodel_with_Tanh = NNSequential(NNLinear(in_feature_size,25),NNTanh(),NNLinear(25,25),NNTanh(),NNLinear(25,25),NNTanh(),NNLinear(25,out_feature_size) )
	nnmodel_with_Relu = NNSequential(NNLinear(in_feature_size,25),NNRelu(),NNLinear(25,25),NNRelu(),NNLinear(25,25),NNRelu(),NNLinear(25,out_feature_size) )
	model_Tanh_tupel = (nnmodel_with_Tanh, "Model with three hidden layers of 25 units with Tanh")
	model_Relu_tupel = (nnmodel_with_Relu, "Model with three hidden layers of 25 units with Relu")
	for model_iter in [model_Tanh_tupel, model_Relu_tupel]: ## for each model
		print(f'\nFor the {model_iter[1]}')
		for loss_type in loss_type_array: ## for each loss type
			nnmodel = model_iter[0]
			nnmodel.set_learning_rate(learning_rate)
			nnmodel.set_loss_function(loss_type)
			nnmodel.train_network(train_input, train_target, epoch_count= epoch_count, mini_batch_size=mini_batch_size, early_stoping=early_stoping, look_back_count_early_stoping=look_back_count_early_stoping)
			## to analyze and plot the results
			analyse_model(nnmodel,train_input,train_target,test_input,test_target, mini_batch_size,with_one_hot_encoding,title=f'Final Results with {loss_type} loss for the \n{model_iter[1]}')

if __name__ == '__main__':
    main()