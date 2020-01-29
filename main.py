import argparse
import os
import numpy as np
import torch
from torch import nn, optim
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from torch.nn import functional as F
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from env import CONTROL_SUITE_ENVS, Env, GYM_ENVS, EnvBatcher
from memory import ExperienceReplay
from models import bottle, Encoder, ObservationModel, RewardModel, TransitionModel
from planner import MPCPlanner
from utils import lineplot, write_video


# Hyperparameters
parser = argparse.ArgumentParser(description='PlaNet')
parser.add_argument('--id', type=str, default='default', help='Experiment ID')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='Random seed')
parser.add_argument('--disable-cuda', action='store_true', help='Disable CUDA')
parser.add_argument('--activation-function', type=str, default='relu', choices=dir(F), help='Model activation function')
parser.add_argument('--embedding-size', type=int, default=1024, metavar='E', help='Observation embedding size')  # Note that the default encoder for visual observations outputs a 1024D vector; for other embedding sizes an additional fully-connected layer is used
parser.add_argument('--hidden-size', type=int, default=200, metavar='H', help='Hidden size')
parser.add_argument('--state-size', type=int, default=30, metavar='Z', help='State/latent size')
parser.add_argument('--batch-size', type=int, default=50, metavar='B', help='Batch size')
parser.add_argument('--chunk-size', type=int, default=50, metavar='L', help='Chunk size')
parser.add_argument('--overshooting-distance', type=int, default=50, metavar='D', help='Latent overshooting distance/latent overshooting weight for t = 1')
parser.add_argument('--overshooting-kl-beta', type=float, default=0, metavar='β>1', help='Latent overshooting KL weight for t > 1 (0 to disable)')
parser.add_argument('--overshooting-reward-scale', type=float, default=0, metavar='R>1', help='Latent overshooting reward prediction weight for t > 1 (0 to disable)')
parser.add_argument('--global-kl-beta', type=float, default=0, metavar='βg', help='Global KL weight (0 to disable)')
parser.add_argument('--free-nats', type=float, default=3, metavar='F', help='Free nats')
parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='α', help='Learning rate')
parser.add_argument('--learning-rate-schedule', type=int, default=0, metavar='αS', help='Linear learning rate schedule (optimisation steps from 0 to final learning rate; 0 to disable)') 
parser.add_argument('--adam-epsilon', type=float, default=1e-4, metavar='ε', help='Adam optimiser epsilon value') 
# Note that original has a linear learning rate decay, but it seems unlikely that this makes a significant difference
parser.add_argument('--grad-clip-norm', type=float, default=1000, metavar='C', help='Gradient clipping norm')
parser.add_argument('--checkpoint-interval', type=int, default=50, metavar='I', help='Checkpoint interval (episodes)')
parser.add_argument('--models', type=str, default='', metavar='M', help='Load model checkpoint')
args = parser.parse_args()
args.overshooting_distance = min(args.chunk_size, args.overshooting_distance)  # Overshooting distance cannot be greater than chunk size
print(' ' * 26 + 'Options')
for k, v in vars(args).items():
  print(' ' * 26 + k + ': ' + str(v))


# Setup
results_dir = os.path.join('results', args.id)
os.makedirs(results_dir, exist_ok=True)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available() and not args.disable_cuda:
  args.device = torch.device('cuda')
  torch.cuda.manual_seed(args.seed)
else:
  args.device = torch.device('cpu')
metrics = {'steps': [], 'episodes': [], 'train_rewards': [], 'test_episodes': [], 'test_rewards': [], 'observation_loss': [], 'reward_loss': [], 'kl_loss': []}


# Initialise model parameters randomly
transition_model = TransitionModel(args.state_size, env.action_size, args.hidden_size, args.embedding_size, args.activation_function).to(device=args.device)
observation_model = ObservationModel(args.state_size, args.embedding_size, args.activation_function).to(device=args.device)
encoder = Encoder(args.embedding_size, args.activation_function).to(device=args.device)
param_list = list(transition_model.parameters()) + list(observation_model.parameters()) + list(encoder.parameters())
optimiser = optim.Adam(param_list, lr=0 if args.learning_rate_schedule != 0 else args.learning_rate, eps=args.adam_epsilon)
if args.models is not '' and os.path.exists(args.models):
  model_dicts = torch.load(args.models)
  transition_model.load_state_dict(model_dicts['transition_model'])
  observation_model.load_state_dict(model_dicts['observation_model'])
  encoder.load_state_dict(model_dicts['encoder'])
  optimiser.load_state_dict(model_dicts['optimiser'])
global_prior = Normal(torch.zeros(args.batch_size, args.state_size, device=args.device), torch.ones(args.batch_size, args.state_size, device=args.device))  # Global prior N(0, I)
free_nats = torch.full((1, ), args.free_nats, device=args.device)  # Allowed deviation in KL divergence

# Training (and testing)
for episode in tqdm(range(metrics['episodes'][-1] + 1, args.episodes + 1), total=args.episodes, initial=metrics['episodes'][-1] + 1):
  # Model fitting
  losses = []
  for s in tqdm(range(args.collect_interval)):
    # Draw sequence chunks {(o_t, a_t, r_t+1, terminal_t+1)} ~ D uniformly at random from the dataset (including terminal flags)
    observations, actions, rewards, nonterminals = D.sample(args.batch_size, args.chunk_size)  # Transitions start at time t = 0
    # Create initial belief and state for time t = 0
    init_state = torch.zeros(args.batch_size, args.state_size, device=args.device)
    # Update belief/state using posterior from previous belief/state, previous action and current observation (over entire sequence at once)
    prior_states, prior_means, prior_std_devs, posterior_states, posterior_means, posterior_std_devs = transition_model(init_state, actions[:-1], bottle(encoder, (observations[1:], )))
    # Calculate observation likelihood, reward likelihood and KL losses (for t = 0 only for latent overshooting); sum over final dims, average over batch and time (original implementation, though paper seems to miss 1/T scaling?)
    observation_loss = F.mse_loss(bottle(observation_model, (posterior_states,)), observations[1:], reduction='none').sum(dim=(2, 3, 4)).mean(dim=(0, 1))
    kl_loss = torch.max(kl_divergence(Normal(posterior_means, posterior_std_devs), Normal(prior_means, prior_std_devs)).sum(dim=2), free_nats).mean(dim=(0, 1))  # Note that normalisation by overshooting distance and weighting by overshooting distance cancel out
    if args.global_kl_beta != 0:
      kl_loss += args.global_kl_beta * kl_divergence(Normal(posterior_means, posterior_std_devs), global_prior).sum(dim=2).mean(dim=(0, 1))
    # Calculate latent overshooting objective for t > 0
    if args.overshooting_kl_beta != 0:
      overshooting_vars = []  # Collect variables for overshooting to process in batch
      for t in range(1, args.chunk_size - 1):
        d = min(t + args.overshooting_distance, args.chunk_size - 1)  # Overshooting distance
        t_, d_ = t - 1, d - 1  # Use t_ and d_ to deal with different time indexing for latent states
        seq_pad = (0, 0, 0, 0, 0, t - d + args.overshooting_distance)  # Calculate sequence padding so overshooting terms can be calculated in one batch
        # Store (0) actions, (1) nonterminals, (2) rewards, (3) beliefs, (4) prior states, (5) posterior means, (6) posterior standard deviations and (7) sequence masks
        overshooting_vars.append((F.pad(actions[t:d], seq_pad), F.pad(nonterminals[t:d], seq_pad), F.pad(rewards[t:d], seq_pad[2:]), prior_states[t_], F.pad(posterior_means[t_ + 1:d_ + 1].detach(), seq_pad), F.pad(posterior_std_devs[t_ + 1:d_ + 1].detach(), seq_pad, value=1), F.pad(torch.ones(d - t, args.batch_size, args.state_size, device=args.device), seq_pad)))  # Posterior standard deviations must be padded with > 0 to prevent infinite KL divergences
      overshooting_vars = tuple(zip(*overshooting_vars))
      # Update belief/state using prior from previous belief/state and previous action (over entire sequence at once)
      prior_states, prior_means, prior_std_devs = transition_model(torch.cat(overshooting_vars[4], dim=0), torch.cat(overshooting_vars[0], dim=1), None)
      seq_mask = torch.cat(overshooting_vars[7], dim=1)
      # Calculate overshooting KL loss with sequence mask
      kl_loss += (1 / args.overshooting_distance) * args.overshooting_kl_beta * torch.max((kl_divergence(Normal(torch.cat(overshooting_vars[5], dim=1), torch.cat(overshooting_vars[6], dim=1)), Normal(prior_means, prior_std_devs)) * seq_mask).sum(dim=2), free_nats).mean(dim=(0, 1)) * (args.chunk_size - 1)  # Update KL loss (compensating for extra average over each overshooting/open loop sequence) 

    # Apply linearly ramping learning rate schedule
    if args.learning_rate_schedule != 0:
      for group in optimiser.param_groups:
        group['lr'] = min(group['lr'] + args.learning_rate / args.learning_rate_schedule, args.learning_rate)
    # Update model parameters
    optimiser.zero_grad()
    (observation_loss + kl_loss).backward()
    nn.utils.clip_grad_norm_(param_list, args.grad_clip_norm, norm_type=2)
    optimiser.step()
    # Store (0) observation loss (1) reward loss (2) KL loss
    losses.append([observation_loss.item(), kl_loss.item()])

  # Update and plot loss metrics
  losses = tuple(zip(*losses))
  metrics['observation_loss'].append(losses[0])
  metrics['kl_loss'].append(losses[2])
  lineplot(metrics['episodes'][-len(metrics['observation_loss']):], metrics['observation_loss'], 'observation_loss', results_dir)
  lineplot(metrics['episodes'][-len(metrics['kl_loss']):], metrics['kl_loss'], 'kl_loss', results_dir)

