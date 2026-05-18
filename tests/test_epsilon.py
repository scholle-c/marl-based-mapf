from marl_path.model.training import (
    get_epsilon_sine
)


def test_epsilon_sine():
    """ Test the get_epsilon_sine function for correctness. """
    min_epsilon = 0.1
    max_epsilon = 1.0
    max_epochs = 100

    # Test at epoch 0
    epsilon_0 = get_epsilon_sine(0, max_epochs, min_epsilon, max_epsilon)

    # Test at epoch max_epochs / 2
    epsilon_mid = get_epsilon_sine(max_epochs // 2, max_epochs, min_epsilon, max_epsilon)

    # Test at epoch max_epochs
    epsilon_end = get_epsilon_sine(max_epochs, max_epochs, min_epsilon, max_epsilon)
    assert abs(epsilon_mid - max_epsilon) < 1e-5, f"Epoch {max_epochs // 2}: Expected {max_epsilon}, got {epsilon_mid}"
    assert abs(epsilon_0 - min_epsilon) < 1e-5, f"Epoch 0: Expected {min_epsilon}, got {epsilon_0}"
    assert abs(epsilon_end - min_epsilon) < 1e-5, f"Epoch {max_epochs}: Expected {min_epsilon}, got {epsilon_end}"