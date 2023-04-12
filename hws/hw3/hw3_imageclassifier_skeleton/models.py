#
import torch
import torch.nn as nn
import numpy as onp
from typing import List, cast
import torch.nn.functional as F


class Model(torch.nn.Module):
    R"""
    Model.
    """
    def initialize(self, rng: torch.Generator) -> None:
        R"""
        Initialize parameters.
        """
        ...


class MLP(Model):
    R"""
    MLP.
    """
    def __init__(self, /, *, size: int, shapes: List[int]) -> None:
        R"""
        Initialize the class.
        """
        #
        Model.__init__(self)

        #
        buf = []
        shapes = [size * size] + shapes
        for (num_ins, num_outs) in zip(shapes[:-1], shapes[1:]):
            #
            buf.append(torch.nn.Linear(num_ins, num_outs))
        self.linears = torch.nn.ModuleList(buf)

    def initialize(self, rng: torch.Generator) -> None:
        R"""
        Initialize parameters.
        """
        #
        for linear in self.linears:
            #
            (num_outs, num_ins) = linear.weight.data.size()
            a = onp.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a, generator=rng)
            linear.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        R"""
        Forward.
        """
        #
        x = torch.flatten(x, start_dim=1)
        for (l, linear) in enumerate(self.linears):
            #
            x = linear.forward(x)
            if l < len(self.linears) - 1:
                #
                x = torch.nn.functional.relu(x)
        return x


#
PADDING = 3


class CNN(torch.nn.Module):
    R"""
    CNN.
    """
    def __init__(
        self,
        /,
        *,
        size: int, channels: List[int], shapes: List[int],
        kernel_size_conv: int, stride_size_conv: int, kernel_size_pool: int,
        stride_size_pool: int,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        Model.__init__(self)
        
        buf_conv = []
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size_pool, stride=stride_size_pool)
        
        
        out_dim = size
        
        
        for i in range(len(channels)-1):
            conv = nn.Conv2d(in_channels=channels[i], out_channels=channels[i+1], kernel_size=kernel_size_conv, stride=stride_size_conv, padding = PADDING)
            buf_conv = buf_conv + [conv]
            
        self.convs = nn.ModuleList(buf_conv)
            
        for layer in range(len(shapes)-1):
            after_conv = (out_dim - kernel_size_conv + 2*PADDING)
            after_conv = after_conv // stride_size_conv
            after_conv = after_conv + 1
            
            after_pool = (after_conv - kernel_size_pool ) 
            after_pool = after_pool // stride_size_pool 
            after_pool = after_pool + 1
            out_dim = after_pool
        
        input_shape = out_dim*out_dim*channels[-1]
        shapes = [input_shape] + shapes
        
        buf = []
        for i in range(len(shapes)-1):
            fc = nn.Linear(shapes[i], shapes[i+1])
            buf = buf + [fc]
            
            
        self.linears = torch.nn.ModuleList(buf)
        
        # Create a list of Conv2D layers and shared max-pooling layer.
        # Input and output channles are given in `channels`.
        # ```
        # 
        # ...
        # 
        # self.pool = ...
        # ```
        # YOU SHOULD FILL IN THIS FUNCTION
        ...

        # Create a list of Linear layers.
        # Number of layer neurons are given in `shapes` except for input.
        # ```
        # 
        # ...
        # 
        # ```
        # YOU SHOULD FILL IN THIS FUNCTION
        ...

    def initialize(self, rng: torch.Generator) -> None:
        R"""
        Initialize parameters.
        """
        #
        for conv in self.convs:
            #
            (ch_outs, ch_ins, h, w) = conv.weight.data.size()
            num_ins = ch_ins * h * w
            num_outs = ch_outs * h * w
            a = onp.sqrt(6 / (num_ins + num_outs))
            conv.weight.data.uniform_(-a, a, generator=rng)
            conv.bias.data.zero_()
        for linear in self.linears:
            #
            (num_outs, num_ins) = linear.weight.data.size()
            a = onp.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a, generator=rng)
            linear.bias.data.zero_()

    def forward(self, x: torch.Tensor, /) -> torch.Tensor:
        R"""
        Forward.
        """
        for conv in self.convs:
            x = self.pool(self.relu(conv(x)))
            
        x = torch.flatten(x, start_dim=1)
        
        for i,linear in enumerate(self.linears):
            if i +1 != len(self.linears):
                x = self.relu(linear(x))
            else:
                x = linear(x)
                
        return x
        # CNN forwarding whose activation functions should all be relu.
        # YOU SHOULD FILL IN THIS FUNCTION
        ...


class CGCNN(Model):
    R"""
    CGCNN.
    """
    def __init__(
        self,
        /,
        *,
        size: int, channels: List[int], shapes: List[int],
        kernel_size_conv: int, stride_size_conv: int, kernel_size_pool: int,
        stride_size_pool: int,
    ) -> None:
        R"""
        Initialize the class.
        """
        #
        Model.__init__(self)

        # This will load precomputed eigenvectors.
        # You only need to define the proper size.
        # proper_size = ...
        # YOU SHOULD FILL IN THIS FUNCTION
        ...

        #
        proper_size = kernel_size_conv
        self.basis: torch.Tensor

        # Loaded eigenvectos are stored in `self.basis`
        with open("rf-{:d}.npy".format(proper_size), "rb") as file:
            #
            onp.load(file)
            eigenvectors = onp.load(file)
        self.register_buffer(
            "basis",
            torch.from_numpy(eigenvectors).to(torch.get_default_dtype()),
        )
        

        buf_weight = []
        buf_bias = []
        
        for i, channel in enumerate(channels[:-1]):
            
            bias_param = nn.Parameter(torch.autograd.Variable(torch.rand((channels[i+1]))),requires_grad=True)
            weight_param = nn.Parameter(torch.autograd.Variable(torch.randn((channels[i+1], channels[i], self.basis.shape[0], 1))),requires_grad=True)
            
            buf_bias = buf_bias + [bias_param]
            buf_weight = buf_weight + [weight_param]
        
        self.weights = torch.nn.ParameterList(buf_weight)
        self.biases = torch.nn.ParameterList(buf_bias)
        
        shapes = [channels[-1]] + shapes
        
        buf = []
        for i in range(len(shapes)-1):
            fc = nn.Linear(shapes[i], shapes[i+1])
            buf = buf + [fc]
            
            
        self.linears = torch.nn.ModuleList(buf)
        
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=kernel_size_pool, stride=stride_size_pool)
        
        self.channels = channels
        self.kernel_size_conv = kernel_size_conv
        self.stride = stride_size_conv
        self.padding = PADDING
        
        # Create G-invariant CNN like CNN, but is invariant to rotation and
        # flipping.
        # linear is the same as CNN.
        # You only need to create G-invariant Conv2D weights and biases.
        # ```
        # buf_weight = []
        # buf_bias = []
        # ...
        # self.weights = torch.nn.ParameterList(buf_weight)
        # self.biases = torch.nn.ParameterList(buf_bias)
        # ```
        # YOU SHOULD FILL IN THIS FUNCTION
        ...

    def initialize(self, rng: torch.Generator) -> None:
        R"""
        Initialize parameters.
        """
        #
        for (weight, bias) in zip(self.weights, self.biases):
            #
            (_, ch_ins, b1, b2) = weight.data.size()
            a = 1 / onp.sqrt(ch_ins * b1 * b2)
            weight.data.uniform_(-a, a, generator=rng)
            bias.data.zero_()
        for linear in self.linears:
            #
            (num_outs, num_ins) = linear.weight.data.size()
            a = onp.sqrt(6 / (num_ins + num_outs))
            linear.weight.data.uniform_(-a, a, generator=rng)
            linear.bias.data.zero_()

    def forward(self, x: torch.Tensor, /) -> torch.Tensor:
        R"""
        Forward.
        """
        # CG-CNN forwarding whose activation functions should all be relu.
        # Pay attention that your forwarding should be invariant to rotation
        # and flipping.
        # Thus, if you rotate x by 90 degree (see structures.py), output of
        # this function should not change.
        # YOU SHOULD FILL IN THIS FUNCTION
        
        
        if self.basis.device != x.device:
            self.basis = self.basis.to(x.device)

        new_weights = torch.mul(self.weights[0], self.basis)
        new_weights = new_weights.sum(dim=-2)
        

        new_weights = new_weights.reshape(
            (self.channels[1],
            self.channels[0],
            self.kernel_size_conv,
            self.kernel_size_conv)
        )

        x = F.conv2d(
            x, weight=new_weights, bias=self.biases[0], stride=self.stride, padding=self.padding
        )
        
        x = self.relu(x)
        x = self.pool(x)
        
        new_weights = torch.mul(self.weights[1], self.basis)
        new_weights = new_weights.sum(dim=-2)

        new_weights = new_weights.reshape(
            (self.channels[2],
            self.channels[1],
            self.kernel_size_conv,
            self.kernel_size_conv)
        )

        x = F.conv2d(
            x, weight=new_weights, bias=self.biases[1], stride=self.stride, padding=self.padding
        )
        
        x = self.relu(x)
        x = self.pool(x)
                
        x = x.sum(dim=-1).squeeze(-1)
        x = x.sum(dim=-1).squeeze(-1)
        
        x = torch.flatten(x, start_dim=1)
        
        for i,linear in enumerate(self.linears):
            if i +1 != len(self.linears):
                x = self.relu(linear(x))
            else:
                x = linear(x)
                
        return x
        