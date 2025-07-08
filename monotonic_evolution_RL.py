import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Beta, Normal


# Trick 8: orthogonal initialization for better training stability
def orthogonal_init(layer, gain=1.0):
    """
    Apply orthogonal initialization to layer weights
    
    Args:
        layer: Neural network layer
        gain: Scaling factor for weights
    """
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)


class Actor_Beta(nn.Module):
    """
    Actor network using Beta distribution for action sampling
    
    Used for bounded continuous action spaces [0,1]
    """
    def __init__(self, args):
        """Initialize actor network with specified architecture"""
        super(Actor_Beta, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.alpha_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.beta_layer = nn.Linear(args.hidden_width, args.action_dim)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick 10: use tanh for better performance

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.alpha_layer, gain=0.01)
            orthogonal_init(self.beta_layer, gain=0.01)

    def forward(self, s):
        """Forward pass through the network to get alpha and beta parameters"""
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        # alpha and beta need to be larger than 1 for proper distribution shape,
        # so we use 'softplus' as the activation function and then add 1
        alpha = F.softplus(self.alpha_layer(s)) + 1.0
        beta = F.softplus(self.beta_layer(s)) + 1.0
        return alpha, beta

    def get_dist(self, s):
        """Get the Beta distribution for given state"""
        alpha, beta = self.forward(s)
        dist = Beta(alpha, beta)
        return dist

    def mean(self, s):
        """Calculate mean action (for deterministic policy execution)"""
        alpha, beta = self.forward(s)
        mean = alpha / (alpha + beta)  # Mean of the beta distribution
        return mean


class Actor_Gaussian(nn.Module):
    """
    Actor network using Gaussian (Normal) distribution for action sampling
    
    Used for unbounded continuous action spaces (later scaled/clipped as needed)
    """
    def __init__(self, args):
        """Initialize actor network with specified architecture"""
        super(Actor_Gaussian, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.mean_layer = nn.Linear(args.hidden_width, args.action_dim)
        # Log standard deviation is a trainable parameter
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim))
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick 10: use tanh for better performance

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.mean_layer, gain=0.01)

    def forward(self, s):
        """Forward pass through network to get mean action value"""
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mean = torch.tanh(self.mean_layer(s))  # Output in range [-1,1]
        return mean

    def get_dist(self, s):
        """Get the Gaussian distribution for given state"""
        mean = self.forward(s)
        # Expand log_std to match mean's dimension
        log_std = self.log_std.expand_as(mean)
        # Exponential ensures standard deviation is positive
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        return dist


class Critic(nn.Module):
    """
    Critic network for value function approximation
    
    Estimates the expected return from a given state
    """
    def __init__(self, args):
        """Initialize critic network with specified architecture"""
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_width)
        self.fc2 = nn.Linear(args.hidden_width, args.hidden_width)
        self.fc3 = nn.Linear(args.hidden_width, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick 10: use tanh for better performance

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)

    def forward(self, s):
        """Forward pass to calculate state value"""
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


class PPO_continuous():
    """
    Proximal Policy Optimization (PPO) algorithm for continuous action spaces
    
    Implements the monotonic evolution reinforcement learning approach with PPO
    """
    def __init__(self, args):
        """Initialize PPO agent with specified parameters"""
        self.policy_dist = args.policy_dist  # Distribution type (Beta or Gaussian)
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # Number of epochs per update
        self.entropy_coef = args.entropy_coef  # Entropy coefficient for exploration
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm

        # Initialize actor network based on chosen distribution type
        if self.policy_dist == "Beta":
            self.actor = Actor_Beta(args)
        else:
            self.actor = Actor_Gaussian(args)
            
        self.critic = Critic(args)

        # Initialize optimizers with optional customizations
        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5 for better numerical stability
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def evaluate(self, s):
        """
        Evaluate the policy deterministically (without sampling)
        
        Used for testing the policy after training
        
        Args:
            s: State vector
            
        Returns:
            a: Deterministic action (mean of policy distribution)
        """
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        if self.policy_dist == "Beta":
            a = self.actor.mean(s).detach().numpy().flatten()
        else:
            a = self.actor(s).detach().numpy().flatten()
        return a

    def choose_action(self, s):
        """
        Sample an action from the policy distribution
        
        Used during training for exploration
        
        Args:
            s: State vector
            
        Returns:
            a: Sampled action
            a_logprob: Log probability of the sampled action
        """
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)

        with torch.no_grad():
            dist = self.actor.get_dist(s)
            a = dist.sample()  # Sample an action according to the probability distribution
            a_logprob = dist.log_prob(a)  # Get the log probability of the sampled action
        return a.numpy().flatten(), a_logprob.numpy().flatten()
    

    def update(self, replay_buffer, total_steps):
        """
        Update policy and value networks using PPO algorithm
        
        Args:
            replay_buffer: Buffer containing experiences
            total_steps: Current training step count
        """
        # Get all training data from the buffer
        s, a, a_logprob, r, s_, dw, done = replay_buffer.numpy_to_tensor()
        
        """
        Calculate the advantage using Generalized Advantage Estimation (GAE)
        - 'dw=True' means dead or win (terminal state), no next state s'
        - 'done=True' represents the terminal of an episode (terminal state or max steps reached)
        """
        adv = []
        gae = 0
        with torch.no_grad():  # No gradient computation needed for advantage calculation
            vs = self.critic(s)
            vs_ = self.critic(s_)
            # Calculate TD errors (deltas)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            
            # Calculate advantages with GAE
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
                
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            # Calculate value targets (V-targets)
            v_target = adv + vs
            
            # Trick 1: Advantage normalization for training stability
            if self.use_adv_norm:
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # PPO update loop - optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Use mini-batches for more efficient and stable learning
            # SubsetRandomSampler ensures random sampling without repetition
            for index in BatchSampler(SubsetRandomSampler(range(s.shape[0])), self.mini_batch_size, False):
                # Get current action distribution
                dist_now = self.actor.get_dist(s[index])
                # Calculate entropy for exploration bonus
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)
                # Calculate log probabilities of actions under current policy
                a_logprob_now = dist_now.log_prob(a[index])
                
                # Calculate probability ratio p_new/p_old using log probabilities
                # For multi-dimensional actions, we sum log_probs across action dimensions
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True))
                
                # PPO clipped objective function
                surr1 = ratios * adv[index]
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                # Trick 5: Add policy entropy to encourage exploration
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy
                
                # Update actor network
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                # Trick 7: Gradient clipping for training stability
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                # Calculate critic (value) loss - MSE between predicted and target values
                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                
                # Update critic network
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                # Trick 7: Gradient clipping for training stability
                if self.use_grad_clip:
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        # Trick 6: Learning rate decay to fine-tune learning as training progresses
        if self.use_lr_decay:
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        """
        Decay learning rates based on training progress
        
        Args:
            total_steps: Current training step count
        """
        # Linear decay schedule
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        
        # Apply new learning rates to optimizers
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now
    
    def save(self, checkpoint_path, frame_id):
        """
        Save actor and critic network parameters
        
        Args:
            checkpoint_path: Directory to save models
            frame_id: Identifier for saved models (usually training step)
        """
        torch.save(self.actor.state_dict(), checkpoint_path + f'actor{frame_id}.pth')
        torch.save(self.critic.state_dict(), checkpoint_path + f'critic{frame_id}.pth')
   
    def load(self, checkpoint_path, frame_id):
        """
        Load actor and critic network parameters
        
        Args:
            checkpoint_path: Directory to load models from
            frame_id: Identifier of models to load
        """
        self.actor.load_state_dict(torch.load(checkpoint_path + f'actor{frame_id}.pth', map_location=lambda storage, loc: storage))
        self.critic.load_state_dict(torch.load(checkpoint_path + f'critic{frame_id}.pth', map_location=lambda storage, loc: storage))
