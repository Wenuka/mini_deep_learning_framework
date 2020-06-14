from torch import Tensor, zeros_like
from math import pi
from torch import argmax as torch_argmax
from plot_helper import plot_prediction_data

def analyse_model(nnmodel,train_input,train_target,test_input,test_target, mini_batch_size=None,with_one_hot_encoding=True,title="Final Results"):
	"""
	Analyzing a given Model by computing number of errors and plotting them
	:param nnmodel: object of the class NNSequential
	:param train_input:
	:param train_target:
	:param test_input:
	:param test_target:
	:param mini_batch_size:
	:param with_one_hot_encoding:
	:param title:
	:return:
	"""
	## computing errors
	train_error_count = NNcompute_nb_errors(nnmodel, train_input, train_target, one_hot_encoding= with_one_hot_encoding, mini_batch_size=mini_batch_size)
	test_error_count = NNcompute_nb_errors(nnmodel, test_input, test_target, one_hot_encoding= with_one_hot_encoding, mini_batch_size=mini_batch_size)
	print (f'train error percentage: {round(train_error_count/len(train_input)*100, 2)}% (error count: {train_error_count})')
	print (f'test error percentage: {round(test_error_count/len(test_input)*100,2)}% (error count: {test_error_count})')
	## plot the final result
	test_prediction = nnmodel.forward(test_input)
	plot_prediction_data(test_input, test_prediction, test_target, title=title)

def generate_disc_set(nb, from_=0, to_=1, one_hot_encoding=False, label_1_in_center= False):
    """
    To generate data set
    :param nb:
    :param from_:
    :param to_:
    :param one_hot_encoding:
    :param label_1_in_center: True if data as in new version of the pdf (ee559-miniprojects.pdf) 
    :return:
    """
    try:
        input_ = Tensor(nb, 2).uniform_(from_, to_)
        if label_1_in_center:
            target = input_.add(-.5).pow(2).sum(1).sub( 1/ (2*pi)).sign().add(1).div(2).long()
        else:
            target = input_.pow(2).sum(1).sub( 1/ (2*pi)).sign().add(1).div(2).long()
        if one_hot_encoding:
            new_target = zeros_like(input_)
            new_target[:,1][target > 0] = 1 ## inside the circle
            new_target[:,0][target < 1] = 1 ## outside the circle
            return input_, new_target
        return input_, target
    except (RuntimeError):
        print(f"error in generate_disc_set() function, from val {from_} should be lower than to {to_}.")
        exit()

def NNcompute_nb_errors(model, data_input, data_target, one_hot_encoding=False, mini_batch_size= None):
    """
    To compute number of errors
    :param model:
    :param data_input:
    :param data_target:
    :param one_hot_encoding:
    :param mini_batch_size:
    :return:
    """
    if mini_batch_size is None:
        mini_batch_size = len(data_input)
    nb_data_errors = 0
    if one_hot_encoding:
      data_target = torch_argmax(data_target, dim=1, keepdim= False).data  #torch.argmax as torch_argmax
    for b in range(0, data_input.size(0), mini_batch_size):
        output = model.forward(data_input.narrow(0, b, mini_batch_size))
        if one_hot_encoding:
          output = torch_argmax(output, dim=1, keepdim= False).data   #torch.argmax as torch_argmax
        for k in range(mini_batch_size):
            if data_target[b + k] != output[k]:
                nb_data_errors = nb_data_errors + 1
    return nb_data_errors