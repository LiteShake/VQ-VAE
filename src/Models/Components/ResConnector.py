import torch.nn as nn

class ResConnector(nn.Module) :

    def __init__(self, codeblock) -> None:
        super(ResConnector, self).__init__()

        self.codeBlock = codeblock

    def forward(self, input):

        res = input
        out = self.codeBlock(input)

        out = res + out
        return out
