import torch.nn as nn


class MaxViT(nn.Module):
    """MaxVit architecture proposed in https://arxiv.org/pdf/2204.01697.pdf"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
