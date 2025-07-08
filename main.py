import os
import torch
import numpy as np
import argparse
from replaybuffer import ReplayBuffer
from monotonic_evolution_RL import PPO_continuous
from scipy import stats
import sys 
sys.path.append("..") 
from VissimEnvironment import VisEnv

def main(args, env_name, number, seed, vissimfile_dir, inpx, layx, confidencelevel):
    """
    Main training function for monotonic evolution RL algorithm
    
    Implements a self-improving RL agent that uses confidence bounds
    on policy performance to ensure monotonic improvement
    
    Args:
        args: Configuration parameters
        env_name: Environment name for saving models
        number: Run identifier
        seed: Random seed
        vissimfile_dir: Directory containing VISSIM files
        inpx: VISSIM network file name
        layx: VISSIM layout file name
        confidencelevel: Statistical confidence level for policy updates
    """
    # Create directory for saving trained models if it doesn't exist
    if True:
        directory = "./PPO_preTrained"
    
        directory = directory + '/' + env_name + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)    
        checkpoint_path = directory
        print("save checkpoint path : " + checkpoint_path)
    
    # Initialize VISSIM environment
    env = VisEnv(vissimfile_dir, inpx, layx)
    
    # Set random seeds for reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Configure environment parameters
    args.state_dim = env.statenum
    args.action_dim = env.actionnum
    args.max_episode_steps = env.max_episode_length
    print("env={}".format(env_name))
    print("state_dim={}".format(args.state_dim))
    print("action_dim={}".format(args.action_dim))
    print("max_episode_steps={}".format(args.max_episode_steps))
    
    # Training metrics
    evaluate_num = 0  # Record the number of evaluations
    results = []  # Record the rewards during evaluation
    total_steps = 0  # Record the total training steps

    # Initialize replay buffers for testing and training
    replay_buffer_test = ReplayBuffer(args)
    replay_buffer_train = ReplayBuffer(args)

    # Initialize agents (old_agent is the current policy, new_agent is the candidate policy)
    old_agent = PPO_continuous(args)
    new_agent = PPO_continuous(args)
    # Copy initial parameters from old to new agent
    new_agent.actor.load_state_dict(old_agent.actor.state_dict())
    new_agent.critic.load_state_dict(old_agent.critic.state_dict())
    
    # Initialize training variables
    i_episode = 0
    train_flag = 0  # Flag: 0 for test trajectories, 1 for training trajectories
    Beta = 20  # Number of trajectories to collect per iteration
    success_update_times = 0
    first_time_flag = 1  # Flag for first update (always accept)
    
    # Main training loop
    while total_steps < args.max_train_steps:
        
        # Collect trajectories for evaluation and training
        for i in range(Beta):
            # Reset environment with random seed
            s = env.reset(np.random.randint(30, 50)) 

            episode_steps = 0
            done = False
            
            # Determine if this trajectory is for training or testing
            # Assign 1/3 of trajectories for training, 2/3 for testing
            if i % 3 == 0:
                train_flag = 1  # Training trajectory
            else:
                train_flag = 0  # Testing trajectory
            
            # Collect trajectory data
            for t in range(1, args.max_episode_steps+1):
                episode_steps += 1
                
                # Select action using current policy
                a, a_logprob = old_agent.choose_action(s)                
                action = a
                
                # Execute action in environment             
                s_, r, done = env.step(action)
                
                # Distinguish between different types of termination
                # dw=True means terminal state (dead or win)
                # done=True means end of episode (terminal or max steps)
                if done and episode_steps != args.max_episode_steps:
                    dw = True
                else:
                    dw = False
                    
                # Store transition in appropriate buffer
                if train_flag:
                    replay_buffer_train.store(s, a, a_logprob, r, s_, dw, done)
                    total_steps += 1
                else:
                    replay_buffer_test.store(s, a, a_logprob, r, s_, dw, done)

                # Update current state
                s = s_
                    
                # Check for episode termination
                if done:
                    if train_flag:
                        i_episode += 1
                    break
                
        # Convert test buffer to tensor format for evaluation
        s, a, a_logprob, r, s_, dw, done = replay_buffer_test.numpy_to_tensor()
        
        # Perform multiple update attempts
        for j in range(4):
            # Update candidate policy with training data
            new_agent.update(replay_buffer_train, total_steps)
            
            # Calculate importance sampling ratios for policy evaluation
            with torch.no_grad():
                dist_now = new_agent.actor.get_dist(s)
                a_logprob_now = dist_now.log_prob(a)
                # Calculate probability ratios between new and old policies
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob.sum(1, keepdim=True))
                ratios = torch.squeeze(ratios).tolist()
           
            # Calculate rewards and importance weights for each trajectory
            rewards = []
            weights = []
            discounted_reward = 0
            cumulative_product_ratio = 0
            last_episode_flag = 1
            
            # Process rewards in reverse order (for discounting)
            for reward, ratio, is_terminal in zip(reversed(replay_buffer_test.r), 
                                                 reversed(ratios), 
                                                 reversed(replay_buffer_test.done)):
                if is_terminal:
                    if last_episode_flag:
                        last_episode_flag = 0
                    else:
                        # Store completed trajectory metrics
                        rewards.insert(0, discounted_reward)
                        weights.insert(0, cumulative_product_ratio)
                        
                    # Reset for new trajectory
                    cumulative_product_ratio = ratio
                    discounted_reward = 0
                
                # Update cumulative metrics
                cumulative_product_ratio *= ratio
                discounted_reward = reward + (args.gamma * discounted_reward)
            
            # Add final trajectory metrics
            rewards.insert(0, discounted_reward)
            weights.insert(0, cumulative_product_ratio)
            
            # Convert to numpy arrays for statistical analysis
            rewards = np.array(rewards)
            weights = np.array(weights)
            
            # Normalize rewards for stable comparison
            rewards = (rewards-50)/70  # Highway scenario normalization
            
            # Calculate mean performance of current policy
            meanreward = np.mean(rewards)
            
            # Calculate importance weighted return (for new policy evaluation)
            weighted_return = rewards * weights
    
            # Calculate confidence bounds using bootstrap
            data = (weighted_return,)
            
            # Perform bootstrap to estimate performance bounds
            bootstrap_results = stats.bootstrap(data, statistic=np.mean, 
                                           n_resamples=2000, confidence_level=(1-2*(1-confidencelevel)))    
            
            # Get lower bound of confidence interval
            low_bound, up_bound = bootstrap_results.confidence_interval
            
            print("low_bound is {}  mean reward is {}".format(low_bound, meanreward))
            
            # Policy update criterion: Accept new policy if its lower confidence bound exceeds mean performance of old policy,
            # or if this is the first update
            if (low_bound > meanreward) or first_time_flag:
                if first_time_flag:
                    first_time_flag = 0
                success_update_times += 1
                
                # Update the old policy with new policy parameters
                old_agent.actor.load_state_dict(new_agent.actor.state_dict())
                old_agent.critic.load_state_dict(new_agent.critic.state_dict())
                
                # Clear buffers for next iteration
                replay_buffer_test.clear()
                replay_buffer_train.clear()
                
                # Save successful policy
                new_agent.save(checkpoint_path, total_steps)
                break

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_train_steps", type=int, default=int(0.40e6), help="Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=1e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=1, help="Save frequency")
    parser.add_argument("--policy_dist", type=str, default="Gaussian", help="Beta or Gaussian distribution for policy")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for training")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size for PPO updates")
    parser.add_argument("--hidden_width", type=int, default=64, help="Width of hidden layers in networks")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate for actor network")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate for critic network")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor for future rewards")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter for advantage estimation")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clipping parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="Number of epochs for PPO updates")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1: Advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2: State normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3: Reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4: Reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: Policy entropy coefficient")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6: Learning rate decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clipping")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: Orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: Set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: Use tanh activation")

    args = parser.parse_args()

    # Define environment and VISSIM configuration
    env_name = "Highway_HCPI"
    vissimfile_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Vissim')
    inpx = 'highwaytest.inpx'
    layx = 'highwaytest.layx'
    confidencelevel = 0.90  # 90% confidence level for policy updates
    seed = 38  # Random seed
    
    # Start training
    main(args, env_name=env_name, number=2, seed=seed, vissimfile_dir=vissimfile_dir, inpx=inpx, layx=layx, confidencelevel=confidencelevel)
    