# Deep Learning Parameters Cheatsheet

Essential, to-the-point cheatsheet and reference for neural network parameters
and hyperparameters, including common architecture and component blueprints.

## Input to Network

* **CSV data**
  * Multilayer preceptron
* **Image**
  * `CNN` (Convolutional neural network)
* **Sequential**
  * `RNN LSTM` (Recurrent neural network, long-short-term-memory)
* **Audio**
  * `RNN LSTM`
* **Video**
  * `CNN` + `RNN` hybrid network

## Initialization

* Biases

  * `0`
    * Biases can generally be zero.

* Weights
  * `XAVIER` (aka `Glorot`)
  * Generic, not `RELU`
  * `RELU` (aka `He`)
  * `RELU` activation
  * `Leaky RELU` activation

## Activation Functions

* `Linear`
  * Regression (output)
* `Sigmoid`
  * Binary classification (output)
* `Tanh`
  * Continuous data, more than [-1, 1]
  * LSTM layers
* `Softmax`
  * Multiclass classification (output)

## Loss functions

* **Recunstruction entropy** (`RBM`, `autoencoder` (Restricted Boltzmann
  Machine))
  * Feature engineering
* **Squared loss** (output)
  * Regression
* **Cross entropy** (output)
  * Binary classification
* **Multiclass cross entropy** (aka MCXE) (output)
  * Multiclass classification
* **Root MSE** (Mean squared error) (`RBM`, `autoencoder`, output)
  * Feature engineering
  * Regression
* **Hinge loss** (output)
  * Classification
* **Negative log likelihood** (output)
  * Classification

## Learning Rates

* Strict values

  * Start with [0.1, 0.01, 0.001], 0.001 being most popular.

* Methods (try in the below order) _ **Adam** _ **Nestrov** (momentum) \*
  Momentum values: [0.5, 0.9, 0.95, 0.99], start with 0.9
  ## Optimizers

Match to networks

* `SGD` (Stochastic gradient descent)
  * CNN (+ dropout)
  * DBN (Deep belief network)
  * RNN
* `Hessian-free`
  * RNN

Properties

* `SGD`
  * Fast to converge (+)
  * Low cost (+)
  * Not as robust (-)
* `L-BFGS` (Limited memory Broydan-Fletcher-Goldfarb-Shanno)
  * Finds better local minima (+)
  * High cost and memory cost (-)
* `CG` (Conjugate gradient)
  * High cost and memory cost (-)
* `Hessian-free`
  * Automatic next step size (+)
  * Can't use on all archs (-)
  * High cost and memory cost (-)

## Batch Sizes

Larger batch sizes improves training efficiency because they ship more data to
computation units (e.g. GPU) at a time.

* Batch size
  * 32 to 1024 on GPUs. Pick numbers that are powers of two.
  * Increasing batch size by factor of N requires epoch number increase by
    factor of N to maintain number of updates.

## Regularization

Prevents overfitting and parameteres becoming too large.

* `L2`
  * Sparse models
  * More heavily penalizes large weights, but doesn’t drive small weights to 0.
* `L1`
  * Dense models
  * Has less of a penalty for large weights, but leads to many weights being
    driven to 0 (or very close to 0), meaning that the resultant weight vector
    can be sparse.
* `Max-norm`
  * Alternative to `L2`, good with large learning ratesj
  * Use with `AdaGrad`, `SGD`
* `Dropout`
  * Temporarily sets activation to 0
  * Works with all NN types
  * Avoid using on first layer, risks loosing information.
  * Increases training times x2, x3, not a good fit for millions of training
    records.
  * Use with `SGD`
  * Influences choice of momentum: 0.95 or 0.99
  * Values (per layer type)
    * Input: [0.5, 1.0)
    * Hidden: 0.5
    * Output: don't use.

## References

* [Deep learning a practitioner's approach](https://www.amazon.com/Deep-Learning-Practitioners-Josh-Patterson/dp/1491914254)

### Thanks:

To all
[Contributors](https://github.com/jondot/deep-learning-parameters-cheatsheet/graphs/contributors) -
you make this happen, thanks!
