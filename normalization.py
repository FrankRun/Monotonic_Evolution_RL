import numpy as np


class RunningMeanStd:
    """
    Calculate running mean and standard deviation for state normalization
    
    Incrementally updates mean and std as new data points are added
    """
    def __init__(self, shape):
        """
        Initialize running statistics
        
        Args:
            shape: The dimension/shape of input data to normalize
        """
        self.n = 0  # Count of samples
        self.mean = np.zeros(shape)  # Mean of samples
        self.S = np.zeros(shape)  # Sum of squared deviations
        self.std = np.sqrt(self.S)  # Standard deviation

    def update(self, x):
        """
        Update statistics with a new data point
        
        Uses Welford's online algorithm for numerical stability
        
        Args:
            x: New data point(s)
        """
        x = np.array(x)
        self.n += 1
        
        if self.n == 1:
            # First sample, just set mean and std directly
            self.mean = x
            self.std = x
        else:
            # Update mean and variance using numerically stable method
            old_mean = self.mean.copy()
            self.mean = old_mean + (x - old_mean) / self.n
            self.S = self.S + (x - old_mean) * (x - self.mean)
            self.std = np.sqrt(self.S / self.n)


class Normalization:
    """
    Normalize states for improved RL training stability
    
    Standardizes inputs to zero mean and unit variance
    """
    def __init__(self, shape):
        """
        Initialize normalizer with running statistics tracker
        
        Args:
            shape: The dimension/shape of input data to normalize
        """
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        """
        Normalize input data
        
        Args:
            x: Input data to normalize
            update: Whether to update running statistics (True for training, False for evaluation)
            
        Returns:
            Normalized data with zero mean and unit variance
        """
        if update:
            self.running_ms.update(x)
        
        # Apply normalization: (x - mean) / std
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)  # Add small constant to avoid division by zero

        return x


class RewardScaling:
    """
    Scale rewards for improved RL training stability
    
    Uses a discounted return approach for normalization
    """
    def __init__(self, shape, gamma):
        """
        Initialize reward scaler
        
        Args:
            shape: The dimension/shape of rewards (typically 1)
            gamma: Discount factor for returns
        """
        self.shape = shape
        self.gamma = gamma  # Discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)  # Running discounted return

    def __call__(self, x):
        """
        Scale the input reward
        
        Args:
            x: Reward value to scale
            
        Returns:
            Normalized reward based on running statistics
        """
        # Update running discounted return
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        
        # Scale rewards by standard deviation only (preserve sign)
        x = x / (self.running_ms.std + 1e-8)
        
        return x

    def reset(self):
        """Reset the running discounted return when an episode ends"""
        self.R = np.zeros(self.shape)
