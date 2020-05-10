from torch import Tensor, matmul, zeros_like
from torch import sum as torch_sum
from torch import argmax as torch_argmax
from torch import abs as torch_abs

from copy import deepcopy
from math import ceil

class NNLinear(object):
  """
  Linear object to create Fully Connected (Dense) Neural Network Object
  """
  def __init__(self,insize,outsize, lr=0.1):
    """
    initialize object with input size and output size (number of perceptrons)
    :param insize:
    :param outsize:
    :param lr:
    """
    self.insize = insize
    self.outsize = outsize
    self.weight = Tensor(outsize, insize).uniform_(-1, 1)
    self.bias = Tensor(outsize).uniform_(-1, 1)
    self.input_ = None
    self.lr = lr
    self.output = None
    self.output_grad = None
  def forward (self, input_):
    """
    forward pass for the object
    :param input_:
    :return:
    """
    output = input_.matmul(self.weight.t())
    output += self.bias
    self.input_ = input_
    self.output = output
    return output
  def backward (self, gradwrtoutput):
    """
    backward pass and update weights for the object
    :param gradwrtoutput:
    :return:
    """
    output_grad = gradwrtoutput.mm(self.weight)
    self.weight = self.weight- matmul( gradwrtoutput.view(-1,self.outsize,1), self.input_.view(-1,1,self.insize)).sum(0) * self.lr
    self.bias = self.bias - gradwrtoutput.sum(0) * self.lr
    self.output_grad = output_grad
    return output_grad
  def set_learning_rate(self, lr):
    """
    set learning rate for the object
    :param lr:
    :return:
    """
    self.lr = lr
  def param (self):
    """
    to return parameters
    :return:
    """
    return [self.output, self.output_grad]  #### <---------- check with TA

class NNTanh(object):
  """
  Neural Network object to create Tanh Non-linear layer
  """
  def __init__(self):
    """
    initialize object
    """
    self.input_ = None
  def forward (self, input_):
    """
    forward pass for the object
    :param input_:
    :return:
    """
    output = (input_.mul(2).exp() - 1)/ (input_.mul(2).exp() + 1) 
    self.input_ = input_
    return output
  def backward (self, gradwrtoutput):
    """
    backward pass for the object
    :param gradwrtoutput:
    :return:
    """
    return 4 * (self.input_.exp() + self.input_.mul(-1).exp()).pow(-2) * gradwrtoutput
  def param (self):
    """
    to return parameters
    :return:
    """
    return []

class NNRelu(object):
  """
  Neural Network object to create Relu Non-linear layer
  """
  def __init__(self):
    """
    to initialize the object
    """
    self.input_ = None
  def forward (self, input_):
    """
    forward pass for the object
    :param input_:
    :return:
    """
    output = input_.clamp(min=0)
    self.input_ = input_
    return output
  def backward (self, gradwrtoutput):
    """
    backward pass for the object
    :param gradwrtoutput:
    :return:
    """
    grad_output = gradwrtoutput.clone()
    grad_output[self.input_ < 0] = 0
    return grad_output
  def param (self):
    """
    return parameters of the object
    :return:
    """
    return []

class NNSequential(object):
  """
  Neural Network object to create Sequential Model using other given Neural Network objects
  """
  def __init__(self,*args):
    """
    initialize a Feed Forward Neural Network with the Neural Network Module sequence given in the *args
    :param args:
    """
    self.NNobject_list = args
    self.lr = 0.1
    self.out = None
    self.target = None
    self.lossval_array = []
    self.epoch_lossval_array = []
    self.batches_per_epoch = 0
    self.last_epoch_loss = 10000
    self.min_loss = 10000
    self.best_obj_array = []
    self.loss_function = self.lossMSE
    self.dloss_function = self.dlossMSE
  def forward (self, x0):
    """
    forward pass for the given module sequence given the first input x0
    :param x0:
    :return:
    """
    mid_result = self.NNobject_list[0].forward(x0)
    if len(self.NNobject_list)>1:
      for obj_idx in range(1, len(self.NNobject_list)): # for all NN objects except first one
        mid_result = self.NNobject_list[obj_idx].forward(mid_result)
    out = zeros_like(mid_result) 
    out[mid_result > 0] = 1
    self.out = out
    return out
  def backward (self):
    """
    backward pass for the given module sequence and update the weights accordingly
    :return:
    """
    mid_grad = self.dloss_function(self.out, self.target.view(-1,len(self.out[0])))
    if len(self.NNobject_list)>1:
      for obj_idx in range(len(self.NNobject_list)-1,-1,-1): # for all NN objects in reverse order
        mid_grad = self.NNobject_list[obj_idx].backward(mid_grad)
  def loss (self,target):
    """
    calculate the loss for the given module sequence given the target
    :param target:
    :return:
    """
    self.target = target   
    lossval = self.loss_function(self.out, target)
    self.lossval = lossval
    self.lossval_array.append(lossval)
    return lossval
  def param (self):
    """
    returns the parameters of the object
    :return:
    """
    return []
  def lossMSE(self, v, t):
    """
    calculate the loss for MSE
    :param v:
    :param t:
    :return:
    """
    return torch_sum((v-t).pow(2))
  def dlossMSE(self, v, t):
    """
    calculate the derivative of loss for MSE
    :param v:
    :param t:
    :return:
    """
    return 2*(v-t)
  def lossMAE(self, v, t):
    """
    calculate the loss for MAE
    :param v:
    :param t:
    :return:
    """
    return torch_sum(torch_abs(v-t))
  def dlossMAE(self, v, t):
    """
    calculate the derivative of the loss for the MAE
    :param v:
    :param t:
    :return:
    """
    out = zeros_like(v)
    out[ (v-t) < 0] = -1
    out[ (v-t) > 0] = 1
    return out
  def set_loss_function(self, lossf):
    """
    Set the loss function between MAE and MSE (default MSE)
    :param lossf:
    :return:
    """
    if (lossf=="MAE"):
      print("using the MAE as loss function.")
      self.loss_function = self.lossMAE
      self.dloss_function = self.dlossMAE
    elif (lossf=="MSE"):
      print("using the MSE as loss function.")
      self.loss_function = self.lossMSE
      self.dloss_function = self.dlossMSE
    else:
      print('given loss function is incorrect. so MSE is used as the loss function')

  def set_learning_rate(self, lr):
    """
    set learning rate for all the modules inside the given module sequence given the learning rate (lr)
    :param lr:
    :return:
    """
    self.lr = lr
    for obj_idx in range(0, len(self.NNobject_list)):
      if isinstance(self.NNobject_list[obj_idx], NNLinear):
        self.NNobject_list[obj_idx].set_learning_rate(lr)
  def train_network(self, train_input, train_target, epoch_count= None, mini_batch_size=None, early_stoping=False, look_back_count_early_stoping=None):
    """
    train the given module sequence
    :param train_input:
    :param train_target:
    :param epoch_count:
    :param mini_batch_size:
    :param early_stoping: if true, training will stop once the loss is zero or if the loss has not decreased in the last look_back_count_early_stoping epochs
    :param look_back_count_early_stoping: number of epochs to wait to see whether loss is decreasing or stop early
    :return:
    """
    # testing conditions:
    if mini_batch_size is None:
      mini_batch_size = len(train_input)
    if epoch_count is None:
      epoch_count = 10
      print (f'Since epoch_count was not defined, value {epoch_count} has been assigned.')
    if (len(train_input) != len(train_target)):
      print(f'Error: input and target lenghts are not equal')
      return 0
    if early_stoping:
      self.batches_per_epoch = ceil(len(train_input)/mini_batch_size)
      if look_back_count_early_stoping is None:
        look_back_count_early_stoping = 10
        print (f'Since look_back_count_early_stoping was not defined with early_stoping set to True, value {look_back_count_early_stoping} has been assigned.')
    # training...
    for k in range(epoch_count): ## for each epoch
      for b in range(0, train_input.size(0), mini_batch_size): ## for each batch
        output = self.forward(train_input.narrow(0, b, mini_batch_size))
        self.loss(train_target.narrow(0, b, mini_batch_size))
        if early_stoping:  ## to stop training incase we get the expected loss
          epoch_loss = sum(self.lossval_array[-self.batches_per_epoch:])
          if epoch_loss > self.last_epoch_loss: ## if new loss is higher than the previous one
            if self.min_loss > self.last_epoch_loss:
              self.min_loss = self.last_epoch_loss
              self.best_obj_array = []
              # save object correspond to the best loss
              for obj_idx in range(0, len(self.NNobject_list)):
                if isinstance(self.NNobject_list[obj_idx], NNLinear):
                  self.best_obj_array.append(deepcopy(self.NNobject_list[obj_idx]))
                else:
                  self.best_obj_array.append(self.NNobject_list[obj_idx])
          self.last_epoch_loss = epoch_loss
          # print(f'{k} - {epoch_loss}') 
        self.backward() 
      self.epoch_lossval_array.append(sum(self.lossval_array[-self.batches_per_epoch:]))
      if early_stoping:  ## to stop training incase we get the expected loss
        if (k>look_back_count_early_stoping and min(self.epoch_lossval_array[-look_back_count_early_stoping:])>= self.min_loss):
          print(f"early finishing at epoch {k} because early stopping is enabled")
          self.NNobject_list = self.best_obj_array
          return 1
        elif epoch_loss == 0: ## stop training if the loss is zero
          print(f'early finishing at epoch {k} because loss is zero...')
          return 1