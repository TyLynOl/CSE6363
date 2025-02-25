import numpy as np

class LinearLayer(Layer):
    """
    Linear layer implementing a fully connected layer.

    For a given input x, weights w, and bias b, the forward pass computes:
        f(x; w) = x @ w.T + b
    where:
        - x is of shape (n, d) with n samples and d features,
        - w is of shape (h, d) with h output features,
        - b is of shape (h,).

    The backward pass computes the gradients with respect to the weights, bias,
    and input.
    """
    
    def __init__(self, in_features: int, out_features: int):
        """
        Initializes the layer with random weights and zero biases.

        Parameters:
            in_features (int): Number of input features (d).
            out_features (int): Number of output features (h).
        """
        # Initialize weights with a scaled random normal distribution
        # W should be of shape (h, d) to ensure x * w.T calculation
        self.w = np.random.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
        
        # Initialize biases as zeros
        self.b = np.zeros(out_features)
        
        # Placeholders for gradients (to be computed in backward pass)
        self.grad_w = np.zeros_like(self.w)
        self.grad_b = np.zeros_like(self.b)
        
        # Placeholder to store input for use in the backward pass
        self.input = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass for the linear layer.

        Parameters:
            input_data (np.ndarray): Input data of shape (n, d).

        Returns:
            np.ndarray: Output data of shape (n, h).
        """
        # Store input for computing gradients in the backward pass
        self.input = input_data
        
        # Compute the linear transformation: x @ w.T + b
        output = np.dot(input_data, self.w.T) + self.b
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass for the linear layer.

        Parameters:
            grad_output (np.ndarray): Gradient of the loss with respect to the output
                                      of this layer, shape (n, h).

        Returns:
            np.ndarray: Gradient of the loss with respect to the input of this layer,
                        shape (n, d).
        """
        # Compute gradient with respect to weights:
        # dL/dw = grad_output.T @ input_data, resulting in shape (h, d)
        self.grad_w = np.dot(grad_output.T, self.input)
        
        # Compute gradient with respect to biases:
        # dL/db = sum of grad_output over all samples (axis=0), shape (h,)
        self.grad_b = np.sum(grad_output, axis=0)
        
        # Compute gradient with respect to the input:
        # dL/dx = grad_output @ w, shape (n, d)
        grad_input = np.dot(grad_output, self.w)
        return grad_input
