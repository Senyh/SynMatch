import torch


class ActivationExtraction(torch.nn.Module):
    """ extracting activations from targetted intermediate layers """

    def __init__(self, model, texture_layer, shape_layer):
        super(ActivationExtraction, self).__init__()
        self.model = model
        self.texture_activation = []
        self.shape_activation = []
        self.texture_grad = []
        self.shape_grad = []
        self.texture_handle = texture_layer.register_forward_hook(self.forward_texture_hook_fn)
        self.shape_handle = shape_layer.register_forward_hook(self.forward_shape_hook_fn)
        
    def forward_texture_hook_fn(self, module, input, output):
        activation = output
        self.texture_activation.append(activation.detach())

    def forward_shape_hook_fn(self, module, input, output):
        activation = output
        self.shape_activation.append(activation.detach())

    def forward(self, x):
        self.texture_activation = []
        self.shape_activation = []
        return self.model(x)

    def __del__(self):
        self.texture_handle.remove()
        self.shape_handle.remove()
           

