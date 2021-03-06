from typing import Optional, List
import torch
from torch import jit, nn
from torch.nn import functional as F


# Wraps the input tuple for a function to process a time x batch x features sequence in batch x features (assumes one output)
def bottle(f, x_tuple):
    x_sizes = tuple(map(lambda x: x.size(), x_tuple))
    y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
    y_size = y.size()
    return y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])


class TransitionModel(jit.ScriptModule):
    __constants__ = ['min_std_dev']

    def __init__(self, state_size, action_size, hidden_size, embedding_size, activation_function='relu',
                 min_std_dev=0.1):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.min_std_dev = min_std_dev
        self.fc_state_prior = nn.Linear(embedding_size, 2 * state_size)
        self.fc_embed_state_action = nn.Linear(state_size + action_size, hidden_size)
        self.fc_state_prior = nn.Linear(hidden_size, 2 * state_size)
        self.fc_state_posterior = nn.Linear(hidden_size, 2 * state_size)

    # Operates over (previous) state, (previous) actions, (previous) belief, (previous) nonterminals (mask), and (current) observations
    # Diagram of expected inputs and outputs for T = 5 (-x- signifying beginning of output belief/state that gets sliced off):
    # t :  0  1  2  3  4  5
    # o :    -X--X--X--X--X-
    # a : -X--X--X--X--X-
    # n : -X--X--X--X--X-
    # pb: -X-
    # ps: -X-
    # b : -x--X--X--X--X--X-
    # s : -x--X--X--X--X--X-
    @jit.script_method
    def forward(self, prev_state: torch.Tensor, actions: torch.Tensor, observations: Optional[torch.Tensor] = None) -> \
    List[torch.Tensor]:
        # Create lists for hidden states (cannot use single tensor as buffer because autograd won't work with inplace writes)
        T = actions.size(0) + 1
        prior_states, prior_means, prior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
        posterior_states, posterior_means, posterior_std_devs = [torch.empty(0)] * T, [torch.empty(0)] * T, [torch.empty(0)] * T
        prior_states[0], posterior_states[0] = prev_state, prev_state
        # Loop over time sequence
        for t in range(T - 1):
            _state = prior_states[t] if observations is None else posterior_states[t] # Select appropriate previous state
            prior_means[t + 1], _prior_std_dev = torch.chunk(self.fc_state_prior(_state), 2, dim=1)
            prior_std_devs[t + 1] = F.softplus(_prior_std_dev) + self.min_std_dev
            prior_states[t + 1] = prior_means[t + 1] + prior_std_devs[t + 1] * torch.randn_like(prior_means[t + 1])
            if observations is not None:
                # Compute state posterior by applying transition dynamics and using current observation
                hidden = self.act_fn(self.fc_embed_state_action(torch.cat([_state, actions[t]], dim=1)))
                posterior_means[t + 1], _posterior_std_dev = torch.chunk(self.fc_state_posterior(hidden), 2, dim=1)
                posterior_std_devs[t + 1] = F.softplus(_posterior_std_dev) + self.min_std_dev
                posterior_states[t + 1] = posterior_means[t + 1] + posterior_std_devs[t + 1] * torch.randn_like(posterior_means[t + 1])
        # Return new hidden states
        hidden = [torch.stack(prior_states[1:], dim=0), torch.stack(prior_means[1:], dim=0),
                  torch.stack(prior_std_devs[1:], dim=0)]
        if observations is not None:
            hidden += [torch.stack(posterior_states[1:], dim=0), torch.stack(posterior_means[1:], dim=0),
                       torch.stack(posterior_std_devs[1:], dim=0)]
        return hidden


class VisualObservationModel(jit.ScriptModule):
    __constants__ = ['embedding_size']

    def __init__(self, state_size, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size
        self.fc1 = nn.Linear(state_size, embedding_size)
        self.conv1 = nn.ConvTranspose2d(embedding_size, 128, 5, stride=2) # 5 x 5
        self.conv2 = nn.ConvTranspose2d(128, 64, 5, stride=2)  # 13 x 13
        self.conv3 = nn.ConvTranspose2d(64, 32, 6, stride=2) # 30 x 30
        self.conv4 = nn.ConvTranspose2d(32, 3, 6, stride=2) # 64 x 64

    @jit.script_method
    def forward(self, state):
        hidden = self.fc1(state)  # No nonlinearity here
        hidden = hidden.view(-1, self.embedding_size, 1, 1)
        hidden = self.act_fn(self.conv1(hidden))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        observation = self.conv4(hidden)
        return observation


def ObservationModel(state_size, embedding_size, activation_function='relu'):
    return VisualObservationModel(state_size, embedding_size, activation_function)


class VisualEncoder(jit.ScriptModule):
    __constants__ = ['embedding_size']

    def __init__(self, embedding_size, activation_function='relu'):
        super().__init__()
        self.act_fn = getattr(F, activation_function)
        self.embedding_size = embedding_size

        self.conv1 = nn.Conv2d(3, 32, 4, stride=2)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 128, 4, stride=2)
        self.conv4 = nn.Conv2d(128, 256, 4, stride=2)
        self.fc = nn.Identity() if embedding_size == 1024 else nn.Linear(1024, embedding_size)

    @jit.script_method
    def forward(self, observation):
        hidden = self.act_fn(self.conv1(observation))
        hidden = self.act_fn(self.conv2(hidden))
        hidden = self.act_fn(self.conv3(hidden))
        hidden = self.act_fn(self.conv4(hidden))
        hidden = hidden.view(-1, 1024)
        hidden = self.fc(hidden)  # Identity if embedding size is 1024 else linear projection
        return hidden


def Encoder(embedding_size, activation_function='relu', min_std_dev=0.1):
    return VisualEncoder(embedding_size, activation_function, min_std_dev)
