#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys
from PyQt5.QtWidgets import (QWidget, QPushButton, QVBoxLayout, QApplication)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg\
                                        as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT\
                                        as NavigationToolbar
from matplotlib.figure import Figure
import control as cnt
import numpy as np
from numpy.polynomial import polynomial as P


def zero(r):
    """Return a monic polynomial with 1-r*z^(-1)"""
    return np.array([1, -r])


def delay(d):
    """Return z^(-d)"""
    z_d = np.zeros(d+1)
    z_d[-1] += 1
    return z_d


def poly2tf(p):
    """Convert a ndarray to a transfer function"""
    n_p = p.shape[0]
    z = np.zeros(n_p)
    z[0] = 1
    return cnt.tf(p, z)


def calculate_rst(b_minus, b_plus, a_minus, a_plus, a_m,
                  a0=np.ones(1), d=1, p=0, r_e=0):
    """Calculate the coefficient of the rst"""
    perturbation = P.polypow(zero(1), p)
    s2, r0 = solve_diophantine(P.polymul(perturbation, a_minus),
                               P.polymul(delay(d), b_minus), P.polymul(a_m, a0))
    b_m_plus, _ = solve_diophantine(P.polymul(delay(d), b_minus),
                                    P.polypow(zero(1), r_e+1), a_m)
    t0 = P.polymul(a0, b_m_plus)
    r = P.polymul(r0, a_plus)
    s = P.polymul(s2, P.polymul(b_plus, perturbation))
    t = P.polymul(t0, a_plus)

    print("R = ", ", ".join(map(str, r)))
    print("S = ", ", ".join(map(str, s)))
    print("T = ", ", ".join(map(str, t)))

    return r, s, t


def solve_diophantine(a, b, c):
    """Solve diophantine equation a*x+b*y=c

    a, b, c: ndarray
    """
    n_a = a.shape[0]
    n_b = b.shape[0]
    n_c = c.shape[0]
    n_x = n_b - 1
    n_y = n_a - 1
    n = n_a + n_b - 2

    if n_c < n:
        c = c.copy()
        c.resize(n)

    if n_c <= n:
        n_x = n_b - 1
        n_y = n_a - 1
    else:
        n = n_c
        n_x = n_c - n_a + 1
        n_y = n_a - 1

    s = np.zeros((n, n))

    for i in range(n_x):
        s[i:n_a+i, i] = a

    for i in range(n_y):
        s[i:n_b+i, i+n_x] = b

    res = np.linalg.solve(s, c)

    x = res[:n_x]
    y = res[n_x:]

    return x, y


class EasyRst(QWidget):
    def __init__(self):
        super().__init__()

        self.initControl()
        self.initUI()

    def initControl(self):
        self.ts = 0.005
        # Right motor
        self.k = 60
        self.tau = 0.025
        # Left motor
        # self.k = 62
        # self.tau = 0.025

        # Position control
        # self.gd = cnt.tf(self.k, [self.tau, 1, 0]).sample(self.ts)
        #
        # c = np.exp(-self.ts/self.tau)
        #
        # b_minus = self.k*np.array([self.ts-self.tau*(1-c),
        #                            self.tau*(1-c)-c*self.ts])
        # b_plus = np.ones(1)
        # a_minus = zero(1)
        # a_plus = zero(c)
        # a_m = P.polypow(zero(0.95), 2)
        # self.r, self.s, self.t =\
        #     map(poly2tf, calculate_rst(b_minus, b_plus, a_minus, a_plus, a_m,
        #                                d=1, p=1))

        # Speed control
        self.gd = cnt.tf(self.k, [self.tau, 1]).sample(self.ts)

        c = np.exp(-self.ts/self.tau)

        b_minus = np.ones(1)*self.k*(1-np.exp(-self.ts/self.tau))
        b_plus = np.ones(1)
        a_minus = np.ones(1)
        a_plus = zero(c)
        a_m = zero(0.5)
        a0 = np.ones(1)
        self.r, self.s, self.t =\
            map(poly2tf, calculate_rst(b_minus, b_plus, a_minus, a_plus, a_m,
                                       a0, d=1, p=1, r_e=1))

        self.time = np.linspace(0, 0.995, 200)

    def initUI(self):
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        self.button = QPushButton('Plot')
        self.button.clicked.connect(self.plot)

        displayLayout = QVBoxLayout()
        displayLayout.addWidget(self.toolbar)
        displayLayout.addWidget(self.canvas)
        displayLayout.addWidget(self.button)

        self.setLayout(displayLayout)

    def computeResponse(self):
        y = self.t*cnt.feedback(self.gd/self.s, self.r)
        u = self.t*cnt.feedback(1/self.s, self.gd*self.r)

        _, self.stepY = cnt.step_response(y, self.time)
        _, self.stepU = cnt.step_response(u, self.time)

    def plot(self):
        self.figure.clear()

        self.computeResponse()

        axY = self.figure.add_subplot(121)
        axY.plot(self.time, self.stepY[0], label='Y')
        axY.set_title('Y response')
        axY.legend()

        axU = self.figure.add_subplot(122)
        axU.plot(self.time, self.stepU[0], label='U')
        axU.set_title('U response')
        axU.legend()

        self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    er = EasyRst()
    er.show()
    sys.exit(app.exec_())
