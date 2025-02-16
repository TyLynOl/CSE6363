import numpy as np
from LinearRegression import LinearRegression  # Replace 'your_module' with your actual filename (without .py)

def test_weight_initialization():
    """Test if weights and bias are initialized correctly."""
    model = LinearRegression()
    X_dummy = np.random.randn(100, 5)  # 100 samples, 5 features
    y_dummy = np.random.randn(100, 1)  # 100 target values

    model.fit(X_dummy, y_dummy)

    assert model.weights.shape == (5, 1), "Weights are not initialized correctly!"
    assert isinstance(model.bias, float) or isinstance(model.bias, np.ndarray), "Bias is not correctly initialized!"
    assert model.weights is not None and model.bias is not None, "Weights or bias should not be None!"

if __name__ == "__main__":
    test_weight_initialization()
    print("All tests passed!")
