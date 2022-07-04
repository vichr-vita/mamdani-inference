from __future__ import annotations
from filecmp import DEFAULT_IGNORES
import numpy as np

from typing import Collection


DEFAULT_RESOLUTION = 1000

class FuzzySet:
    """
    piecewise linear mu
    helper methods construct fuzzy numbers
    """

    def __init__(self, vertices: Collection[Collection], height: float = 1, **kwargs) -> None:
        self.vertices: np.ndarray = np.array(vertices)
        self.height = height
        self.kwargs = kwargs

    def mu(self, x: float) -> float:
        return np.interp([x], [t[0] for t in self.vertices], [t[1] for t in self.vertices])[0]

    def mu_inv(self, mu: float) -> float:
        return np.interp([mu], [t[1] for t in self.vertices], [t[0] for t in self.vertices])[0]


    def discretize(self, range: tuple[float, float], resolution: int = DEFAULT_RESOLUTION):
        return np.array([np.array([x, self.mu(x)]) for x in np.linspace(range[0], range[1], resolution)])

    @staticmethod
    def discrete(x: float) -> FuzzySet:
        return FuzzySet(vertices=(
            (x-.000000000001, 0),
            (x, 1),
            (x+.000000000001, 0)
        ))

    @staticmethod
    def triangular(a, b, c) -> FuzzySet:
        vertices = np.array([
            (-np.inf, 0),
            (a, 0),
            (b, 1),
            (c, 0),
            (np.inf, 0)
        ])
        return FuzzySet(vertices)

    @staticmethod
    def trapezoidal(a, b, c, d) -> FuzzySet:
        vertices = np.array([
            (-np.inf, 0),
            (a, 0),
            (b, 1),
            (c, 1),
            (d, 0),
            (np.inf, 0)
        ])
        return FuzzySet(vertices)

    @staticmethod
    def l_ramp(start, end) -> FuzzySet:
        return FuzzySet(np.array([
            (-np.inf, 1),
            (start, 1),
            (end, 0),
            (np.inf, 0)
        ]))

    @staticmethod
    def r_ramp(start, end) -> FuzzySet:
        return FuzzySet(np.array([
            (-np.inf, 0),
            (start, 0),
            (end, 1),
            (np.inf, 1)
        ]))

    @staticmethod
    def uniform(height: float) -> FuzzySet:
        return FuzzySet(np.array([
            (-np.inf, height),
            (0, height),
            (np.inf, height)
        ]))


    @staticmethod
    def union(f1: FuzzySet, f2:FuzzySet, range: tuple[float, float], resolution: int = DEFAULT_RESOLUTION) -> FuzzySet:
        x = np.linspace(range[0], range[1], resolution)
        vertices = np.vstack((x, np.maximum(f1.discretize(range, resolution=resolution)[:, 1], f2.discretize(range, resolution=resolution)[:, 1]))).T
        f3 = FuzzySet(vertices)
        f3.height = np.max(vertices[:, 1])
        return f3

    @staticmethod
    def intersection(f1: FuzzySet, f2:FuzzySet, range: tuple[float, float], resolution: int = DEFAULT_RESOLUTION) -> FuzzySet:
        x = np.linspace(range[0], range[1], resolution)
        vertices = np.vstack((x, np.minimum(f1.discretize(range, resolution=resolution)[:, 1], f2.discretize(range, resolution=resolution)[:, 1]))).T
        f3 = FuzzySet(vertices)
        f3.height = np.max(vertices[:, 1])
        return f3