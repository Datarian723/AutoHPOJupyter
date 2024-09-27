# ---------------------------
# Special Imports
# ---------------------------
from __future__ import annotations  # Enable forward references for type hints

# ---------------------------
# Standard Libraries
# ---------------------------
import os  # Operating system utilities
import gc  # Garbage collection for memory management
import logging  # Logging utilities
import sys  # System-specific parameters and functions
import unittest  # For unit-testing Python code
import copy  # For deep and shallow copying of objects
import math  # Math operations
import random  # Random number generation
import gzip  # For compressing and decompressing data
from functools import partial  # Partial function application
from collections import defaultdict  # Dictionary subclass with default values
from sortedcontainers import SortedSet  # Sorted set implementation
from unittest.mock import Mock, patch  # For mocking and patching in unit tests
from inspect import signature  # Inspecting callable signatures
from typing import Callable, Union  # Type annotations for type hints
from ast import Is  # Abstract syntax tree utilities
from abc import ABC, abstractmethod  # Abstract Base Classes for inheritance

# ---------------------------
# Scientific Libraries
# ---------------------------
import numpy as np  # Array and matrix operations
import pandas as pd  # Data manipulation and analysis
import matplotlib.pyplot as plt  # Plotting and visualization

# ---------------------------
# PyTorch and Related Modules
# ---------------------------
import torch  # Core PyTorch library for tensor operations
import torch.nn as nn  # Neural network modules
import torch.nn.functional as F  # Functional API for neural networks
import torchvision.models as models  # Pre-trained models from torchvision
from torch.multiprocessing import Pool, set_start_method  # Multiprocessing utilities for PyTorch

# ---------------------------
# FastAI Libraries
# ---------------------------
from fastai.vision.all import *  # All-in-one import for vision-specific modules
from fastai.metrics import *  # Evaluation metrics for model performance
from fastai.callback.hook import *  # Hooks for capturing intermediate model states
from fastai.callback.tracker import *  # Training callback trackers (e.g., early stopping)
from fastai.learner import Learner  # Core Learner object for model training

# ---------------------------
# Optuna for Hyperparameter Optimization
# ---------------------------
import optuna  # Core Optuna library for optimization
from optuna.integration import FastAIPruningCallback  # Pruning callback for FastAI integration
from optuna.trial import Trial, FrozenTrial  # Trial classes for defining and managing trials

# ---------------------------
# Google Colab Utilities
# ---------------------------
from google.colab import files  # File utilities for Colab
from google.colab import drive  # Mounting Google Drive in Colab
