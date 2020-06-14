# Mini Deep Learning Framework

## Introduction
The objective of this project is to design a mini “deep learning framework” using only pytorch’s tensor operations and the standard math library, hence in particular without using autograd or the neural-network modules.

For this project, we have implemented a Sequential Neural Network Object which can be used to create a multi-layer, fully-connected (Dense) neural network with non-linearity layers Tanh or Rectified-linear (Relu). Training the model can also be done using early stopping feature, which enable the training to stop if the loss is not decreasing.

## Packages required + versions
This code is written using python 3

1. Torch - 1.4.0

2. Matplotlib (for creating the plots) - 3.1.3

3. General Python3 modules - math, copy

## Project structure

```
proj2
│   README.md
│   general_helper.py
│   NN_classes.py
│   plot_helper.py
│   test.ipynb
│   test.py
│   report.pdf
│   
└───__pycache__/


```

## Code architecture

The code is structured in 2 different types of files described as follows:

- Jupyter Notebooks : These files details our models, and choices of implementations (and techniques)
- Python files : Submission files for computing the final models and grid search

## Usage instructions

### Run the `test.py` script as follows

   If run inside an ordinary terminal:
   ```
   python3 test.py
   ```

   If run inside an anaconda terminal with python 3 already installed:
   ```
   python test.py
   ```
This should create, train and evaluate the model for 4 iterations, Namely using two models each having three hidden layers with 25 units, one using Tanh and other using Relu as the non-linear layer. Each of those two layers were tested using two losses, namely MSE and MAE.

For each iteration results will be plotted and the results summary will get printed on the terminal.

### Other functionalities
By changing the following variables in the test.py, you can achieve the corresponding tasks,

1. label_1_in_center: to change between dataset 1 and 2 (refer the report.py)

2. train_size: number of training and testing pairs, default 1000.

3. mini_batch_size: mini batch size

4. early_stopping: stop early before all the epoch if the loss is not decreasing within the epoch count 'look_back_count_early_stoping'

5. look_back_count_early_stoping: how many epoch to look back before stopping due to early_stopping

6. learning_rate: learning rate

7. plot_inputs: plot inputs if this is set to True, by default False

## Thank you
