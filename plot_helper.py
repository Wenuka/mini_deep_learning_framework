import matplotlib.pyplot as plt

def plot_data(train_input, train_target, test_input, test_target,title="Train and Test for each Target"):
  """
  plot train and test input data for each target 0 and 1
  :param train_input:
  :param train_target:
  :param test_input:
  :param test_target:
  :param title:
  :return:
  """
  # for train samples - label 1
  x1=(train_input[train_target[:,0]>=0.5].narrow(1,0,1).view(-1)).numpy();
  y1=(train_input[train_target[:,0]>=0.5].narrow(1,1,1).view(-1)).numpy();
  # for train samples -  0
  x0=(train_input[train_target[:,0]<0.5].narrow(1,0,1).view(-1)).numpy();
  y0=(train_input[train_target[:,0]<0.5].narrow(1,1,1).view(-1)).numpy();
  # for test sample - label 1
  x1_test=(test_input[test_target[:,0]>=0.5].narrow(1,0,1).view(-1)).numpy();
  y1_test=(test_input[test_target[:,0]>=0.5].narrow(1,1,1).view(-1)).numpy();
  # for test sample - 0
  x0_test=(test_input[test_target[:,0]<0.5].narrow(1,0,1).view(-1)).numpy();
  y0_test=(test_input[test_target[:,0]<0.5].narrow(1,1,1).view(-1)).numpy();
  ## plot the data
  plt.figure(num=None, figsize=(8, 4), dpi=80, facecolor='w', edgecolor='k')
  plt.suptitle(f'{title}')
  subplot=plt.subplot(1,2,1)
  # axes = plt.gca();
  plt.title("For label 1");
  subplot.plot(x1,y1,'g.',label="train sample");
  subplot.plot(x1_test,y1_test,'b.', label="test sample");
  plt.legend(loc="lower left");
  subplot=plt.subplot(1,2,2)
  # axes = plt.gca();
  plt.title("For label 0");
  subplot.plot(x0,y0,'g.', label="train sample");
  subplot.plot(x0_test,y0_test,'b.', label="test sample");
  plt.legend(loc="upper right");
  plt.show()

def plot_prediction_data(test_input, test_target, test_prediction, title=""):
  """
  plot the final output and mark errors
  :param test_input:
  :param test_target:
  :param test_prediction:
  :param title:
  :return:
  """
  # # for test samples - label 1
  x1=(test_input[test_prediction[:,0]>=0.5].narrow(1,0,1).view(-1)).numpy();
  y1=(test_input[test_prediction[:,0]>=0.5].narrow(1,1,1).view(-1)).numpy();
  # # for test samples - label 0
  x0=(test_input[test_prediction[:,0]<0.5].narrow(1,0,1).view(-1)).numpy();
  y0=(test_input[test_prediction[:,0]<0.5].narrow(1,1,1).view(-1)).numpy();
  # # for miss classified samples
  x_error=(test_input[test_prediction[:,0]!= test_target[:,0]].narrow(1,0,1).view(-1)).numpy();
  y_error=(test_input[test_prediction[:,0]!= test_target[:,0]].narrow(1,1,1).view(-1)).numpy();

  ## plot the data
  plt.figure(num=None, figsize=(8, 8), dpi=80, facecolor='w', edgecolor='k')
  plt.title(f'{title}')
  axes = plt.gca();
  plt.plot(x1,y1,'g.',label="prediction 1");
  plt.plot(x0,y0,'b.',label="prediction 0");
  plt.plot(x_error,y_error,'rx',label="prediction errors");
  plt.legend(loc="upper right");
  plt.show()
