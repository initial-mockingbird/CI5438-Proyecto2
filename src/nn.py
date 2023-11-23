from decimal import DivisionByZero
import pandas as pd
import numpy as np
import random

random.seed(5)

def cross(Y,T):
  return - (T * np.log(Y)).sum() / Y.shape[0]

def mse(Y,T):
  delta = Y-T
  return (delta.T @ delta) / Y.shape[0]

def cross_grad(Y,T):
  return (Y-T) / Y.shape[0]

def mse_grad(Y,T):
  return (Y-T) * 2 / Y.shape[0]

def logistic(X):
  return 1 / (1 + np.exp(-X))

def softmax(X):
  #np.seterr(all='raise')
  #np.seterr(all='warn')
  # under/over flow fix
  z = X - np.max(X,axis=-1,keepdims=True)
  return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

functions =\
  { "logistic": logistic
  , "softmax": softmax
  }

def logistic_derivative(X): 
  return X * (1 - X)


# Layers used in this model
class Layer:
  """Base class for the different layers.
  Defines base methods and documentation of methods."""
  
  def get_params(self):
    """Return an iterator over the parameters (if any).
    The iterator has the same order as W_grad.
    The elements returned by the iterator are editable in-place."""
    return None
  
  def W_grad(self, X = None, upstream_grad = None):
    """Return a list of gradients over the parameters.
    The list has the same order as the get_params_iter iterator.
    X is the input.
    upstream_grad is the gradient at the output of this layer.
    """
    return None
  
  def get_output(self, X = None):
    """Perform the forward step linear transformation.
    X is the input."""
    pass
  
  def X_grad(self, Y = None, upstream_grad = None, T = None):
    """Return the gradient at the inputs of this layer.
    Y is the pre-computed output of this layer (not needed in 
    this case).
    upstream_grad is the gradient at the output of this layer 
      (gradient at input of next layer).
    Output layer uses targets T to compute the gradient based on the 
      output error instead of upstream_grad"""
    pass

class OutputLayer(Layer):

  def get_params(self):
    raise NotImplementedError("get_params_iter intentionally not implemented for OutputLayer")

  def W_grad(self, X=None, upstream_grad=None):
    raise NotImplementedError("W_grad intentionally not implemented for OutputLayer")
  
  def get_cost(self,Y ,T):
    raise NotImplementedError

class Lineal_Layer(Layer):
   
  def __init__(self,input_neurons,output_neurons) -> None:
    self.W = np.array([ np.random.uniform(0,1,output_neurons) for _ in range(input_neurons)])
    self.b = np.array([0 for _ in range(output_neurons)])
  

  def get_params(self):
    return (self.W,self.b)

  def get_output(self, X : np.ndarray) -> np.ndarray:
    """Perform the forward step linear transformation."""
    return X @ self.W + self.b
      
  def W_grad(self, X : np.ndarray, upstream_grad : np.ndarray) -> tuple[np.ndarray,np.ndarray]:
    """Return a list of gradients over the parameters."""
    JW : np.ndarray = X.T @ upstream_grad
    Jb : np.ndarray = np.sum(upstream_grad, axis=0) #,keepdims=True)
    return (JW,Jb)

  def X_grad(self, Y, upstream_grad : np.ndarray,T = None) -> np.ndarray:
    """Return the gradient at the inputs of this layer."""
    return upstream_grad  @ self.W.T
  

class Logistic(Layer):
  def get_output(self, X : np.ndarray) -> np.ndarray:
    return logistic(X)
  def X_grad(self,Y : np.ndarray,upstream_grad : np.ndarray,T = None) -> np.ndarray:
    return logistic_derivative(Y) * upstream_grad

class LogisticOutput(OutputLayer):
  
  def get_output(self,X):
    return logistic(X)

  def X_grad(self, Y,T, upstream_grad = None):
    return (Y - T) / Y.shape[0]

  def get_cost(self,Y ,T):
    delta = Y - T
    return - (Y * np.log(T + 1e-7) + (1-Y)*np.log(1-T + 1e-7)).sum() / Y.shape[0]

class SoftMaxOutput(OutputLayer):
  def get_params(self):
    raise NotImplementedError("get_params_iter intentionally not implemented for OutputLayer")

  def W_grad(self, X=None, upstream_grad=None):
    raise NotImplementedError("W_grad intentionally not implemented for OutputLayer")
  
  def get_output(self,X):
    return softmax(X)

  def X_grad(self, Y,T, upstream_grad = None):
    return (Y - T) / Y.shape[0]

  def get_cost(self,Y ,T):
    return - (T * np.log(Y)).sum() / Y.shape[0]
  

class NN:
  def __init__(self,X_train,T_train,learning_rate = 1e-1, max_iter=300,epsilon=1e-7) -> None:
    self.layers = []
    self.output_layer : OutputLayer | None = None 
    self.X_train = X_train
    self.T_train = T_train
    self.learning_rate = learning_rate
    self.max_iter = max_iter
    self.validation   = 5
    #self.X_validation = X_train[:self.validation,:]
    #self.T_validation = T_train[:self.validation] #T_train[:self.validation,:]
    self.batch_size = 25
    self.epsilon = epsilon

  def add_layer(self,layer : Layer):
    self.layers.append(layer)
  def add_layers(self,layers):
    self.layers += layers
  def add_output_layer(self,output_layer : OutputLayer):
    self.output_layer = output_layer
  def get_output_layer(self) -> OutputLayer:
    if self.output_layer is None:
      raise AttributeError("no output layer provided")
    return self.output_layer

  

  def _forward(self,X):
    # X = self.input_layer
    activations = [X]
    for layer in self.layers + [self.output_layer]:
      activations.append(layer.get_output(activations[-1]))
    return activations
  
  def _backward(self,activations,T):
    W_grads = []
    Y = activations.pop()
    output_layer : OutputLayer = self.get_output_layer()
    hidden_layers : list[Layer] = self.layers

    downstream_grad  = output_layer.X_grad(Y=Y,T=T)
    # output layer (and every activation layer) does not take any W parameters, so this always return None
    W_grads.append(None)
    upstream_grad = downstream_grad

    for layer in reversed(hidden_layers):
      Y = activations.pop()
      downstream_grad = layer.X_grad(Y=Y,upstream_grad=upstream_grad)
      X = activations[-1]
      W_grads.append(layer.W_grad(X=X,upstream_grad=upstream_grad))
      upstream_grad = downstream_grad
    return list(reversed(W_grads))
  
  def _update_params(self,W_grads):
    layers = self.layers + [self.output_layer]
    for i in range(len(W_grads)):
      layer = layers[i]
      layer_backprop_grads = W_grads[i]
      if layer_backprop_grads is None:
        continue
      
      assert(isinstance(layer,Lineal_Layer))
      (dW,db) = layer_backprop_grads
      layer.W = layer.W - self.learning_rate * dW
      layer.b = layer.b - self.learning_rate * db
  
  # de aqui para abajo, robao
  def _gradient_checking(self):
    assert(self.output_layer is not None)
    activations = self._forward(self.X_validation)
    W_grads = self._backward(activations,self.T_validation)
    e = 1e-4
    for (layer,layer_Wb_grad) in zip(self.layers + [self.output_layer], W_grads):
      if layer_Wb_grad is None:
        continue
      (layer_W_grad,layer_b_grad) = layer_Wb_grad
      params = layer.get_params()
      
      (W,b) = params
      for (i,j), value in np.ndenumerate(W):
        grad_backprop = layer_W_grad[i,j]
        value += e
        plus_cost = self.output_layer.get_cost(self._forward(self.X_validation)[-1],self.T_validation)
        value -= 2*e
        min_cost = self.output_layer.get_cost(self._forward(self.X_validation)[-1],self.T_validation)
        value += e
        grad_num = (plus_cost - min_cost) / (2*e)
        if not np.isclose(grad_num, grad_backprop):
          raise ValueError((
            f'Numerical gradient of {grad_num:.6f} is '
            'not close to the backpropagation gradient '
            f'of {grad_backprop:.6f}! Fault found in W'))
      for i, value in np.ndenumerate(b):
        grad_backprop = layer_b_grad[i]
        value += e
        plus_cost = self.output_layer.get_cost(self._forward(self.X_validation)[-1],self.T_validation)
        value -= 2*e
        min_cost = self.output_layer.get_cost(self._forward(self.X_validation)[-1],self.T_validation)
        value += e
        grad_num = (plus_cost - min_cost) / (2*e)
        if not np.isclose(grad_num, grad_backprop):
          raise ValueError((
            f'Numerical gradient of {grad_num:.6f} is '
            'not close to the backpropagation gradient '
            f'of {grad_backprop:.6f}! Fault found in b'))
  
  def _minibatch(self):
    nb_of_batches = self.X_train.shape[0] // self.batch_size
    return list(zip(
      np.array_split(self.X_train, nb_of_batches, axis=0),
      np.array_split(self.T_train, nb_of_batches, axis=0)))
  
  def train(self):
    assert(self.output_layer is not None)
    batch_costs = []
    train_costs = []
    val_costs = []

    # Train for the maximum number of iterations
    for _ in range(self.max_iter):
      for X, T in self._minibatch():  # For each minibatch sub-iteration
        
        # Get the activations
        activations = self._forward(X)
        # Get cost
        batch_cost = self.output_layer.get_cost(activations[-1], T)
        batch_costs.append(batch_cost)
        # Get the gradients
        W_grads = self._backward(activations, T)
        # Update the parameters
        self._update_params(W_grads)
      # Get full training cost for future analysis (plots)
      activations = self._forward(self.X_train)
      
      train_cost = self.output_layer.get_cost(activations[-1], self.T_train)
      train_costs.append(train_cost)
      
    return (train_costs,batch_costs)
      
  def predict(self,X):
    return self._forward(X)[-1]



