import numpy as np

class Layer:
    """
    Abstract base class for all layers in the neural network.

    This class defines the interface for all layers by requiring the implementation
    of both forward and backward passes.
    """

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass of the layer.

        Parameters:
            input_data (np.ndarray): Input data to the layer.

        Returns:
            np.ndarray: The output after applying the layer's transformation.
        """
        raise NotImplementedError("Forward method not implemented in the base Layer class.")

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass of the layer.

        This method should calculate the gradient of the loss with respect to the input 
        of the layer, given the gradient with respect to its output.

        Parameters:
            grad_output (np.ndarray): Gradient of the loss with respect to the layer's output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the layer's input.
        """
        raise NotImplementedError("Backward method not implemented in the base Layer class.")

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
    
class Sigmoid(Layer):
    """
    Sigmoid activation layer.

    This layer applies the logistic sigmoid function element-wise:
        f(x) = 1 / (1 + exp(-x))

    During the forward pass, it computes the activation and stores it to use in the backward pass.
    """
    
    def __init__(self):
        # This will store the output from the forward pass
        self.out = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass using the sigmoid activation function.

        Parameters:
            input_data (np.ndarray): The input data.

        Returns:
            np.ndarray: The sigmoid activation of the input.
        """
        self.out = 1 / (1 + np.exp(-input_data))
        return self.out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass for the sigmoid activation.

        The derivative of the sigmoid function is:
            f'(x) = f(x) * (1 - f(x))
        Multiply this derivative with the incoming gradient to get the gradient with respect to the input.

        Parameters:
            grad_output (np.ndarray): Gradient of the loss with respect to the output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input.
        """
        grad_input = grad_output * self.out * (1 - self.out)
        return grad_input

class ReLU(Layer):
    """
    Rectified Linear Unit (ReLU) activation layer.

    Applies the ReLU activation function element-wise:
        f(x) = max(0, x)
    """
    
    def __init__(self):
        # We'll store a boolean mask indicating where the input is greater than 0.
        self.mask = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass for the ReLU activation.

        Parameters:
            input_data (np.ndarray): Input data.

        Returns:
            np.ndarray: Output data where negative values are set to zero.
        """
        # Create a mask: True where input is positive, False otherwise.
        self.mask = (input_data > 0)
        # Apply ReLU: Zero out negative values.
        output = input_data * self.mask
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass for the ReLU activation.

        The derivative of the ReLU function is:
            f'(x) = 1 if x > 0, and 0 otherwise.
        Thus, we propagate the gradient only where the input was positive.

        Parameters:
            grad_output (np.ndarray): Gradient of the loss with respect to the output.

        Returns:
            np.ndarray: Gradient of the loss with respect to the input.
        """
        # Multiply the incoming gradient element-wise with the mask.
        grad_input = grad_output * self.mask
        return grad_input

class BinaryCrossEntropyLoss(Layer):
    """
    Binary Cross-Entropy Loss layer.

    This layer computes the binary cross-entropy loss between the predicted probabilities
    and the true binary labels. For a batch of predictions p and targets y, the loss is:

        L = -1/N * sum_{i=1}^{N} [ y_i * log(p_i) + (1 - y_i) * log(1 - p_i) ]

    where:
        - p (predictions) is assumed to be output from a sigmoid and lies in (0, 1),
        - y (targets) are binary labels (0 or 1),
        - N is the number of samples in the batch.

    The backward pass computes the gradient with respect to the predictions.
    """
    def __init__(self):
        # Placeholders for storing predictions and targets during the forward pass.
        self.predictions = None
        self.targets = None
        # Small constant to avoid taking log(0)
        self.epsilon = 1e-8

    def forward(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Compute the binary cross-entropy loss.

        Parameters:
            predictions (np.ndarray): Predicted probabilities (values between 0 and 1)
            targets (np.ndarray): True binary labels (0 or 1)

        Returns:
            float: The average binary cross-entropy loss over the batch.
        """
        self.predictions = predictions
        self.targets = targets

        # Clip predictions to avoid log(0)
        predictions_clipped = np.clip(predictions, self.epsilon, 1 - self.epsilon)
        # Compute the loss for each sample and average over the batch
        loss = -np.mean(
            targets * np.log(predictions_clipped) + 
            (1 - targets) * np.log(1 - predictions_clipped)
        )
        return loss

    def backward(self, grad_output: np.ndarray = 1) -> np.ndarray:
        """
        Compute the gradient of the loss with respect to the predictions.

        The derivative of the binary cross-entropy loss for a single sample is given by:
            dL/dp = - (y/p) + ((1 - y)/(1 - p))
        Averaging over N samples gives:
            dL/dp = ( - (y/p) + ((1 - y)/(1 - p)) ) / N

        Parameters:
            grad_output (np.ndarray): Upstream gradient (typically 1, since the loss is a scalar)

        Returns:
            np.ndarray: The gradient of the loss with respect to the predictions.
        """
        N = self.targets.shape[0]
        predictions_clipped = np.clip(self.predictions, self.epsilon, 1 - self.epsilon)

        # Compute the gradient for each prediction
        grad_input = - (self.targets / predictions_clipped) + ((1 - self.targets) / (1 - predictions_clipped))
        grad_input = grad_input / N

        # Multiply by grad_output in case it's not 1 (to support chaining)
        grad_input *= grad_output
        return grad_input

class Sequential(Layer):
    """
    A Sequential container that holds a list of layers and processes data sequentially.
    It inherits from Layer, supporting both forward and backward passes, and provides
    methods for saving and loading the model weights.
    """

    def __init__(self, layers: list = None):
        """
        Initialize the Sequential model.

        Parameters:
            layers (list, optional): A list of layer objects. Defaults to an empty list.
        """
        self.layers = layers if layers is not None else []

    def add(self, layer: Layer):
        """
        Add a new layer to the model.

        Parameters:
            layer (Layer): A layer instance (e.g., LinearLayer, Sigmoid, ReLU) to add.
        """
        self.layers.append(layer)

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Perform the forward pass by sequentially passing the input through all layers.

        Parameters:
            input_data (np.ndarray): The input data to the model.

        Returns:
            np.ndarray: The output after processing through all layers.
        """
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Perform the backward pass by sequentially propagating gradients in reverse order.

        Parameters:
            grad_output (np.ndarray): The gradient of the loss with respect to the model's output.

        Returns:
            np.ndarray: The gradient of the loss with respect to the model's input.
        """
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def save_weights(self, file_path: str):
        """
        Save the weights of all layers that have trainable parameters (e.g., LinearLayer)
        to a file.

        Parameters:
            file_path (str): The path to the file where weights will be saved.
        """
        weights = {}
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "w"):
                weights[f"layer_{i}_w"] = layer.w
                weights[f"layer_{i}_b"] = layer.b
        np.savez(file_path, **weights)

    def load_weights(self, file_path: str):
        """
        Load weights from a file and assign them to the corresponding layers.

        Parameters:
            file_path (str): The path to the file from which to load the weights.
        """
        weights = np.load(file_path)
        for i, layer in enumerate(self.layers):
            if hasattr(layer, "w"):
                layer.w = weights[f"layer_{i}_w"]
                layer.b = weights[f"layer_{i}_b"]

class Tanh(Layer):
    """
    Hyperbolic Tangent (tanh) activation layer.

    Applies the tanh function element-wise:
        f(x) = tanh(x)
    """
    
    def __init__(self):
        self.out = None

    def forward(self, input_data: np.ndarray) -> np.ndarray:
        """
        Compute the forward pass using the tanh activation function.
        
        Parameters:
            input_data (np.ndarray): The input data.
        
        Returns:
            np.ndarray: The tanh activation of the input.
        """
        self.out = np.tanh(input_data)
        return self.out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Compute the backward pass for the tanh activation.

        The derivative of tanh is:
            f'(x) = 1 - tanh(x)^2

        Parameters:
            grad_output (np.ndarray): Gradient of the loss with respect to the output.
        
        Returns:
            np.ndarray: Gradient of the loss with respect to the input.
        """
        grad_input = grad_output * (1 - self.out ** 2)
        return grad_input
