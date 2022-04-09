import torch
import torch.nn as nn

from AutoEncoder import MLPEncoder
from AutoEncoder import MLPDecoder


class encoding(nn.Module):
    def __init__(self, state1_dim=5):
        super().__init__()
        self.Encoder = MLPEncoder(state1_dim)

    def forward(self, state1):
        x1 = self.Encoder(state1[:,0,:])
        x2 = self.Encoder(state1[:,1,:])
        x3 = self.Encoder(state1[:,2,:])
        x4 = self.Encoder(state1[:,3,:])
        x5 = self.Encoder(state1[:,4,:])
        x6 = self.Encoder(state1[:,5,:])
        x7 = self.Encoder(state1[:,6,:])

        x = torch.concat([x1, x2, x3, x4,
                          x5, x6, x7], dim=-1)
        return x
#
class Regressor(nn.Module):
    def __init__(self, state2_dim=8, output_dim=2187):
        super().__init__()

        self.state2_dim = state2_dim
        self.output_dim = output_dim
        self.layer1 = nn.Linear(7*128+state2_dim, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, output_dim)
        self.hidden_act = nn.ReLU()
        self.out_act = nn.Identity()
        nn.init.xavier_uniform_(self.layer1.weight)
        nn.init.xavier_uniform_(self.layer2.weight)
        nn.init.xavier_uniform_(self.layer3.weight)
        nn.init.xavier_uniform_(self.layer4.weight)

    def forward(self, x, state2):
        x = torch.concat([x, state2.view(-1, self.state2_dim)], dim=-1)
        x = self.layer1(x)
        x = self.hidden_act(x)
        x = self.layer2(x)
        x = self.hidden_act(x)
        x = self.layer3(x)
        x = self.hidden_act(x)
        x = self.layer4(x)
        x = self.out_act(x)
        return x

class qnet(nn.Module):
    def __init__(self, state1_dim=5, state2_dim=8, output_dim=2187):
        super().__init__()
        self.encoding = encoding(state1_dim)
        self.regressor = Regressor(state2_dim, output_dim)

    def forward(self, state1, state2):
        x = self.encoding(state1)
        x = self.regressor(x, state2)
        return x