import torch
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, depth, width, activation, weight_init, bias_init, d_in, d_out):
        super().__init__()

        # Create a list of layer widths
        sizes = [d_in] + [width] * depth                                    

        # Container for activation values
        self.h = []

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
        self.h = [] # Reset stored activations          
        h = x.flatten(1) # h(0) = x

        # Recursion over the networks hidden layers
        for layer in self.layers:
            h = self.activation(layer(h))
            if h.requires_grad:
                h.retain_grad() # Allow measurement of gradients
            self.h.append(h)
        return self.out(h)