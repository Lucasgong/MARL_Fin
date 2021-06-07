import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class CnnDQN(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(CnnDQN, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.features = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def feature_size(self):
        return self.features(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)


class DQN(nn.Module):
    def __init__(self,  num_days, hidden_dims=(256, 256), num_actions=3):
        super(DQN, self).__init__()
        self.input_shape = num_days
        self.num_actions = num_actions
        self.layers = nn.Sequential(OrderedDict([
            ("fc0", nn.Linear(self.input_shape, hidden_dims[0])),
            ("relu0", nn.ReLU())
        ]))
        for i in range(len(hidden_dims)-1):
            self.layers.add_module("fc"+str(i+1), nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            self.layers.add_module("relu"+str(i+1), nn.ReLU())
        self.layers.add_module("fc"+str(len(hidden_dims)), nn.Linear(hidden_dims[-1], num_actions))

    def forward(self, x):
        return self.layers(x)


class RNN(nn.Module):
    def __init__(self, input_shape, rnn_hidden_dim, n_actions):
        super(RNN, self).__init__()

        self.fc1 = nn.Linear(input_shape, rnn_hidden_dim)
        self.rnn = nn.GRUCell(rnn_hidden_dim, rnn_hidden_dim)
        self.fc2 = nn.Linear(rnn_hidden_dim, n_actions)

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        h = self.rnn(x, h_in)
        q = self.fc2(h)
        return q, h
