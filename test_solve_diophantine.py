#!/usr/bin/python3
# -*- coding: utf-8 -*-

from easyrst import solve_diophantine
import numpy as np


def test_solve_diophantine1():
    x, y = solve_diophantine(np.array([1, 1]), np.array([0, 1]),
                             np.array([1, 2]))
    assert np.allclose(x, np.array([1]))\
        and np.allclose(y, np.array([1]))


def test_solve_diophantine2():
    x, y = solve_diophantine(np.array([1, 1, 1]), np.array([0, 1]),
                             np.array([1, 0, 2]))
    assert np.allclose(x, np.array([1]))\
        and np.allclose(y, np.array([-1, 1]))


def test_solve_diophantine3():
    x, y = solve_diophantine(np.array([1, 1, 1]), np.array([1, 0, 1]),
                             np.array([4, 7, 6, 6]))
    assert np.allclose(x, np.array([1, 2]))\
        and np.allclose(y, np.array([3, 4]))


def test_solve_diophantine4():
    x, y = solve_diophantine(np.array([1, 1]), np.array([0, 1]),
                             np.array([1, 0]))
    assert np.allclose(x, np.array([1]))\
        and np.allclose(y, np.array([-1]))


def test_solve_diophantine5():
    x, y = solve_diophantine(np.array([1, 1]), np.array([0, 1]),
                             np.array([1]))
    assert np.allclose(x, np.array([1]))\
        and np.allclose(y, np.array([-1]))


def test_solve_diophantine6():
    x, y = solve_diophantine(np.array([1, 1]), np.array([0, 1]),
                             np.array([1, 1, 1, 1, 1]))
    assert np.allclose(x, np.array([1, 1, 0, 1]))\
        and np.allclose(y, np.array([-1]))
