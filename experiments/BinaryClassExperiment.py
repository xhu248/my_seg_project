import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from datasets.two_dim.NumpyDataLoader import NumpyDataSet
from trixi.experiment.pytorchexperiment import PytorchExperiment

class BinaryClaaExperiment(PytorchExperiment):
