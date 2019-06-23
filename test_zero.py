#!/usr/bin/python3
# -*- coding: utf-8 -*-

from easyrst import zero
import numpy as np


def test_zero1():
    assert np.allclose(zero(1), np.array([1, -1]))


def test_zero2():
    assert np.allclose(zero(-1), np.array([1, 1]))


def test_zero3():
    assert np.allclose(zero(5), np.array([1, -5]))


def test_zero4():
    assert np.allclose(zero(-3), np.array([1, 3]))
