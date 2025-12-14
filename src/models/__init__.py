# This file makes the models directory a Python package

from .chicken_cnn import ChickenCNN
from .simple_cnn_v1 import SimpleCNNv1
from .simple_cnn_v2 import SimpleCNNv2
from .birdy_cnn import BirdyCNN
from .attr_cnn import AttrCNN
from .t_rex import TRex, train_trex, validate_trex, compute_trex_loss

__all__ = [
    'ChickenCNN',
    'SimpleCNNv1',
    'SimpleCNNv2',
    'BirdyCNN',
    'AttrCNN',
    'TRex',
    'train_trex',
    'validate_trex',
    'compute_trex_loss',
]
