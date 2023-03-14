import logging
import math
import numpy as np
import torch
from copy import deepcopy
from collections import OrderedDict

from .activations import relu, softmax, cross_entropy, stable_softmax


class AutogradNeuralNetwork:
    """Implementation that uses torch.autograd

        Neural network classifier with cross-entropy loss
        and ReLU activations
    """
    def __init__(self, shape, gpu_id=-1):
        """Initialize the network

        Args:
            shape: a list of integers that specifieds
                    the number of neurons at each layer.
            gpu_id: -1 means using cpu. 
        """
        self.shape = shape
        # declare weights and biases
        if gpu_id == -1:
            self.weights = [torch.autograd.Variable(torch.FloatTensor(j, i),
                                requires_grad=True)
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.biases = [torch.autograd.Variable(torch.FloatTensor(i, 1),
                                requires_grad=True)
                           for i in self.shape[1:]]
        else:
            self.weights = [torch.autograd.Variable(torch.randn(j, i).cuda(gpu_id),
                                requires_grad=True)
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.biases = [torch.autograd.Variable(torch.randn(i, 1).cuda(gpu_id),
                                requires_grad=True)
                           for i in self.shape[1:]]
        # initialize weights and biases
        self.init_weights()

    def init_weights(self):
        """Initialize weights and biases

            Initialize self.weights and self.biases with
            Gaussian where the std is 1 / sqrt(n_neurons)
        """
        for i in range(len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            stdv = 1. / math.sqrt(w.size(1))
            # in-place random
            w.data.uniform_(-stdv, stdv)
            b.data.uniform_(-stdv, stdv)

    def _feed_forward(self, X):
        """Forward pass

        Args:
            X: (n_neurons, n_examples)

        Returns:
            (outputs, act_outputs).

            "outputs" is a list of torch tensors. Each tensor is the Wx+b (weighted sum plus bias)
            of each layer in the shape (n_neurons, n_examples).

            "act_outputs" is also a list of torch tensors. Each tensor is the "activated" outputs
            of each layer in the shape(n_neurons, n_examples). If f(.) is the activation function,
            this should be f(ouptuts).
        """
        
        
        outputs = []
        act_outputs = []

        for i in range(len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            
            X = w.matmul(X) + b

            outputs.append(X)
            
            if i+1 == len(self.weights):
                X = stable_softmax(X)
            else:
                X = relu(X)
                
            
            act_outputs.append(X)
        
        return outputs, act_outputs

        # raise NotImplementedError

    def train_one_epoch(self, X, y, y_1hot, learning_rate):
        """Train for one epoch

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        X_t = X.t()
        y_1hot_t = y_1hot.t()

        X_t_train = X_t
        y_1hot_t_train = y_1hot_t
        
        # feed forward
        outputs, act_outputs = self._feed_forward(X_t_train)
        loss = cross_entropy(act_outputs[-1], y_1hot_t_train)

        # backward
        loss.backward()

        # update weights and biases
        for w, b in zip(self.weights, self.biases):
            w.data = w.data - (learning_rate * w.grad.data)
            b.data = b.data - (learning_rate * b.grad.data)
            w.grad.data.zero_()
            b.grad.data.zero_()
        return loss.item()
    
    def loss(self, X, y, y_1hot):
        """Compute feed forward loss

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        X_t = X.t()
        y_1hot_t = y_1hot.t()
        outputs, act_outputs = self._feed_forward(X_t)
        loss = cross_entropy(act_outputs[-1], y_1hot_t)
        return loss.item()

    def predict(self, X):
        """Predict

            Make predictions for X using the current model

        Args:
            X: (n_examples, n_features)

        Returns:
            (n_examples, )
        """
        outputs, act_outputs = self._feed_forward(X.t())
        return torch.max(act_outputs[-1], 0)[1]


class BasicNeuralNetwork:
    """Implementation using only torch.Tensor

        Neural network classifier with cross-entropy loss
        and ReLU activations
    """
    def __init__(self, shape, gpu_id=-1):
        """Initialize the network

        Args:
            shape: a list of integers that specifieds
                    the number of neurons at each layer.
            gpu_id: -1 means using cpu. 
        """
        self.shape = shape
        # declare weights and biases
        if gpu_id == -1:
            self.weights = [torch.FloatTensor(j, i)
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.biases = [torch.FloatTensor(i, 1)
                           for i in self.shape[1:]]
        else:
            self.weights = [torch.randn(j, i).cuda(gpu_id)
                            for i, j in zip(self.shape[:-1], self.shape[1:])]
            self.biases = [torch.randn(i, 1).cuda(gpu_id)
                           for i in self.shape[1:]]

        # initialize weights and biases
        self.init_weights()

    def init_weights(self):
        """Initialize weights and biases

            Initialize self.weights and self.biases with
            Gaussian where the std is 1 / sqrt(n_neurons)
        """
        for i in range(len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            stdv = 1. / math.sqrt(w.size(1))
            # in-place random
            w.uniform_(-stdv, stdv)
            b.uniform_(-stdv, stdv)

    def _feed_forward(self, X):
        """Forward pass

        Args:
            X: (n_neurons, n_examples)
        
        Returns:
            (outputs, act_outputs).

            "outputs" is a list of torch tensors. Each tensor is the Wx+b (weighted sum plus bias)
            of each layer in the shape (n_neurons, n_examples).

            "act_outputs" is also a list of torch tensors. Each tensor is the "activated" outputs
            of each layer in the shape(n_neurons, n_examples). If f(.) is the activation function,
            this should be f(ouptuts).
        """
        
        
        outputs = []
        act_outputs = []

        for i in range(len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
 
            X = w.matmul(X) + b

            outputs.append(X)
            
            if i+1 == len(self.weights):
                X = stable_softmax(X)
            else:
                X = relu(X)
            
            
            act_outputs.append(X)
        
        return outputs, act_outputs
        # raise NotImplementedError

    def _backpropagation(self, outputs, act_outputs, X, y_1hot):
        """Backward pass

        Args:
            outputs: (n_neurons, n_examples). get from _feed_forward()
            act_outputs: (n_neurons, n_examples). get from _feed_forward()
            X: (n_features, n_examples). input features
            y_1hot: (n_classes, n_examples). labels
        """
        grads_w = []
        grads_b = []
        h_i = act_outputs[-1]
        
        prev_grad = h_i -  y_1hot
        

        for i in range(len(self.weights)-2,-1,-1):
            
            
            h_i = act_outputs[i]
            
            grad_w = torch.matmul(prev_grad, h_i.T)
            grad_b = torch.sum(prev_grad,axis=1).reshape(prev_grad.shape[0],1)
            
            grads_w.append(grad_w)
            grads_b.append(grad_b)
            
            w = self.weights[i+1]
            prev_grad = torch.matmul(w.T,prev_grad)            
            prev_grad[h_i<=0] = 0
            
            
        # print(X.shape)
        # print(prev_grad.shape)
            
        grad_w = torch.matmul(prev_grad,X.T)
        grad_b = torch.sum(prev_grad,axis=1).reshape(prev_grad.shape[0],1)
        
        
        grads_w.append(grad_w)
        grads_b.append(grad_b)
        
        grads_w = grads_w[::-1]
        grads_b = grads_b[::-1]
        
#         for dw in grads_w:
#             print(dw.shape)
        
#         for db in grads_b:
#             print(db.shape)
            
#         print("============")
        
#         for i in range(len(self.weights)):
#             print(self.weights[i].shape)
            
#         for i in range(len(self.biases)):
#             print(self.biases[i].shape)
            
#         import sys
#         sys.exit(0)

        return grads_w,grads_b
        # raise NotImplementedError

    def train_one_epoch(self, X, y, y_1hot, learning_rate):
        """Train for one epoch

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        X_t = X.t()
        y_1hot_t = y_1hot.t()

        X_t_train = X_t
        y_1hot_t_train = y_1hot_t
        
        # feed forward
        outputs, act_outputs = self._feed_forward(X_t_train)
        loss = cross_entropy(act_outputs[-1], y_1hot_t_train)

        # backward
        grads_w, grads_b = self._backpropagation(outputs,act_outputs,X_t,y_1hot_t)

        # update weights and biases
        for i in range(len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            
            # print(torch.isinf(grads_w[i]).any())
            # print(torch.isinf(w[i]).any())
            
            # print("before")
            # print(torch.isnan(b[i]).any())
            # print(torch.isnan(grads_b[i]).any())
            
            w.data = w.data - (learning_rate * grads_w[i])
            b.data = b.data - (learning_rate * grads_b[i])
            
            # print("after")
            # print(torch.isnan(b[i]).any())
            # print(torch.isnan(grads_b[i]).any())
            
        # import sys
        # sys.exit(0)
        
        return loss.item()
        
        """Train for one epoch

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        X_t = X.t()
        y_1hot_t = y_1hot.t()

        X_t_train = X_t
        y_1hot_t_train = y_1hot_t
        
        # feed forward
        outputs, act_outputs = self._feed_forward(X_t_train)
        loss = cross_entropy(act_outputs[-1], y_1hot_t)

        # backward
        grads_w, grads_b = self._backpropagation(outputs,act_outputs,X_t,y_1hot_t)
        # loss.backward()

        # update weights and biases
        for i in range(len(self.weights)):
            w = self.weights[i]
            b = self.biases[i]
            
            # print("before")
            # print(b.shape)
            # print(grads_b[i].shape)
            
            # print(w.shape)
            # print(grads_w[i].shape)
            
            w.data = w.data - (learning_rate * grads_w[i])
            b.data = b.data - (learning_rate * grads_b[i])
            
            # print("after")
            # print(b.shape)
            # print(grads_b[i].shape)
            
            # print(w.shape)
            # print(grads_w[i].shape)
        
        return loss.item()

        # raise NotImplementedError

    def loss(self, X, y, y_1hot):
        """Compute feed forward loss

        Args:
            X: (n_examples, n_features)
            y: (n_examples). don't care
            y_1hot: (n_examples, n_classes)

        Returns:
            Loss for the given input
        """
        X_t = X.t()
        y_1hot_t = y_1hot.t()
        outputs, act_outputs = self._feed_forward(X_t)
        loss = cross_entropy(act_outputs[-1], y_1hot_t)
        
        return loss.item()

    def predict(self, X):
        """Predict

            Make predictions for X using the current model

        Args:
            X: (n_examples, n_features)

        Returns:
            (n_examples, )
        """
        outputs, act_outputs = self._feed_forward(X.t())
        return torch.max(act_outputs[-1], 0)[1]
