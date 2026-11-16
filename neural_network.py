import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    """
    Fully connected feedforward network for studying variance propagation.

    Computes h⁽ˡ⁾ = σ(W⁽ˡ⁾ h⁽ˡ⁻¹⁾ + b⁽ˡ⁾) for l = 1, …, L, with h⁽⁰⁾ = x,
    followed by a linear readout to the logits.

    Parameters
    ----------
    depth : int
        Number of hidden layers L.
    width : int
        Number of neurons in each hidden layer.
    activation : callable
        Elementwise activation function σ, e.g. `torch.tanh`.
    weight_init : callable
        Function that fills a layer's weight tensor of shape (n_out, n_in)
        in place, e.g. `nn.init.xavier_normal_`. Applied to every layer,
        including the readout.
    bias_init : callable
        Function that fills a layer's 1-D bias tensor in place, e.g. `nn.init.zeros_`.
    d_in : int
        Input dimension (784 for flattened MNIST).
    d_out : int
        Number of outputs (10 for MNIST).

    Attributes
    ----------
    layers : nn.ModuleList
        Hidden linear layers W⁽¹⁾, …, W⁽ᴸ⁾.
    out : nn.Linear
        Readout layer mapping h⁽ᴸ⁾ to the logits.
    h : list of torch.Tensor
        Activations h⁽¹⁾, …, h⁽ᴸ⁾ from the last forward pass, each of shape
        (batch, width). After `loss.backward()`, `h[l-1].grad` holds ∂L/∂h⁽ˡ⁾.
        Empty until the first call.
    z: list of torch.Tensor
        Pre-activations z⁽¹⁾, …, z⁽ᴸ⁾ from the last forward pass, each of shape
        (batch, width). Empty until the first call.
    """

    def __init__(self, depth, width, activation, weight_init, bias_init, d_in, d_out):
        super().__init__()

        # Create a list of layer widths
        sizes = [d_in] + [width] * depth                                    

        self.h = [] # Container for activation values
        self.z = [] # Container for pre-activation values

        # Create network layers 
        self.layers = nn.ModuleList(nn.Linear(sizes[i], sizes[i+1]) for i in range(depth))
        self.out = nn.Linear(sizes[-1], d_out)

        # Specify activation function
        self.activation = activation

        # Weight and bias initialisation
        with torch.no_grad():                        
            for layer in [*self.layers, self.out]:
                weight_init(layer.weight)
                bias_init(layer.bias)
     
    def forward(self, x):
        self.z = [] # Reset stored pre-activations          
        self.h = [] # Reset stored activations          
        h = x.flatten(1) # h(0) = x

        # Recursion over the networks hidden layers
        for layer in self.layers:
            z = layer(h)
            h = self.activation(layer(h))
            if h.requires_grad:
                h.retain_grad() # Allow measurement of gradients
            self.z.append(z)
            self.h.append(h)
        return self.out(h)