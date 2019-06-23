#!/usr/bin/python3
# -*- coding: utf-8 -*-

from easyrst import delay
import numpy as np


def test_delay1():
    assert np.allclose(delay(0), np.array([1]))


def test_delay2():
    assert np.allclose(delay(1), np.array([0, 1]))


def test_delay3():
    assert np.allclose(delay(3), np.array([0, 0, 0, 1]))
