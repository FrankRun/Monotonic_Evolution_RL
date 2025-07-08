import torch
import numpy as np


class ReplayBuffer:
    """
    Experience Replay Buffer for storing and retrieving agent experiences
    
    Stores state transitions, rewards, actions and other necessary data for RL training
    """
    def __init__(self, args):
        """
        Initialize replay buffer with dimensions matching environment
        
        Args:
            args: Configuration parameters containing state and action dimensions
        """
        self.state_dim = args.state_dim
        self.action_dim = args.action_dim
        # Initialize empty lists to store experience data
        self.s = []        # States
        self.a = []        # Actions
        self.a_logprob = []  # Log probabilities of actions (for importance sampling)
        self.r = []        # Rewards
        self.s_ = []       # Next states
        self.dw = []       # Done flags with distinction of terminal vs truncated episodes
        self.done = []     # Episode termination flags
        

    def store(self, s, a, a_logprob, r, s_, dw, done):
        """
        Store a single transition in the buffer
        
        Args:
            s: Current state
            a: Action taken
            a_logprob: Log probability of the action under current policy
            r: Reward received
            s_: Next state
            dw: Boolean indicating terminal state (True if dead or win)
            done: Boolean indicating episode termination (True if terminal or max steps)
        """
        self.s.append(s)
        self.a.append(a)
        self.a_logprob.append(a_logprob)
        self.r.append(r)
        self.s_.append(s_)
        self.dw.append(dw)
        self.done.append(done)

    def clear(self):
        """Clear all stored transitions from the buffer"""
        del self.s[:]
        del self.a[:]
        del self.a_logprob[:]
        del self.r[:]
        del self.s_[:]
        del self.dw[:]
        del self.done[:]
                
    def numpy_to_tensor(self):
        """
        Convert stored NumPy arrays to PyTorch tensors for neural network processing
        
        Returns:
            Tuple of PyTorch tensors (s, a, a_logprob, r, s_, dw, done)
        """
        # Convert lists to properly shaped NumPy arrays
        s = np.array(self.s).reshape([-1, self.state_dim])
        a = np.array(self.a).reshape([-1, self.action_dim])
        a_logprob = np.array(self.a_logprob).reshape([-1, self.action_dim])
        r = np.array(self.r).reshape([-1, 1])
        s_ = np.array(self.s_).reshape([-1, self.state_dim])
        dw = np.array(self.dw).reshape([-1, 1])
        done = np.array(self.done).reshape([-1, 1])
        
        # Convert NumPy arrays to PyTorch tensors
        s = torch.tensor(s, dtype=torch.float)
        a = torch.tensor(a, dtype=torch.float)
        a_logprob = torch.tensor(a_logprob, dtype=torch.float)
        r = torch.tensor(r, dtype=torch.float)
        s_ = torch.tensor(s_, dtype=torch.float)
        dw = torch.tensor(dw, dtype=torch.float)
        done = torch.tensor(done, dtype=torch.float)

        return s, a, a_logprob, r, s_, dw, done
