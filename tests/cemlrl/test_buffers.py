"""Tests for the cemrl/buffers.py"""
import numpy as np
from gym import spaces

from src.cemrl.buffers import CEMRLPolicyBuffer

def test_policy_sample_valid_data():
    sut = CEMRLPolicyBuffer(1000, observation_space=)