import numpy as np
import torch
import torch.nn as nn


class DQN(nn.Module):
    '''
    Deep Q newtork following the architecture used in the DeepMind paper
    (https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)
    '''

    # TODO implement noisy_net
    def __init__(self, input_shape, n_actions, noisy_net=False):
        super(DQN, self).__init__()

        # 3 convolutional layers. Take an image as input
        # (NB: the BatchNorm layers aren't in the paper
        # but they increase the convergence)
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        # Compute the output shape of the conv layers
        conv_out_size = self._get_conv_out(input_shape)

        # 2 fully connected layers
        if noisy_net:
            # In case of NoisyNet use noisy linear layers
            # TODO reactivate
            # self.fc = nn.Sequential(
                # NoisyLinear(conv_out_size, 512),  # noqa
                # nn.ReLU(),  # noqa
                # NoisyLinear(512, n_actions))  # noqa
            pass
        else:
            self.fc = nn.Sequential(
                nn.Linear(conv_out_size, 512),
                nn.ReLU(),
                nn.Linear(512, n_actions))

    def _get_conv_out(self, shape):
        # Compute the output shape of the conv layers
        o = self.conv(torch.zeros(1, *shape))  # apply convolution layers..
        return int(np.prod(o.size()))  # ..to obtain the output shape

    def forward(self, x):

        batch_size = x.size()[0]

        # apply convolution layers and flatten the results
        conv_out = self.conv(x).view(batch_size, -1)
        return self.fc(conv_out)  # apply fc layers
