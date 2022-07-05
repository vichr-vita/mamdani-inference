from matplotlib.axes import Axes
from fuzzy_inference.fuzzy_set import FuzzySet
from fuzzy_inference.linguistic_variable import LinguisticVariable
import matplotlib.pyplot as plt
import numpy as np


def _term_name_x_pos(vertices: np.ndarray):
    if len(vertices) == 4 + 2:  # case trapezoidal
        return (vertices[2][0] + vertices[3][0])/2
    if len(vertices) == 3 + 2:  # case triangular
        return vertices[2][0]
    if len(vertices) == 2 + 2:  # case ramp
        if vertices[0][1] > vertices[-1][1]:  # case l_ramp
            return (vertices[0][0] + vertices[1][0])/2
        else:  # case r_ramp
            return (vertices[2][0] + vertices[3][0])/2


class LVVizualizer:

    def __init__(self, lv: LinguisticVariable) -> None:
        self.lv = lv

    def vizualize(self, ax: Axes):
        for term_name, term in self.lv.terms.items():
            vertices = term.constrain_range(
                (self.lv.min, self.lv.max)).vertices
            ax.plot(vertices[:, 0], vertices[:, 1], c='k')
            ax.text(_term_name_x_pos(vertices), 1.2,
                    term_name, ha='center', va='center')
            ax.set_ylim(0, 1.4)
            ax.set_title(self.lv.name)
            ax.set_xlabel(self.lv.domain)
            ax.set_ylabel(r'$\mu$', rotation=0)
        return ax


class FuzzySetVizualizer:
    def __init__(self, f: FuzzySet, name: str = '', domain: str = 'x') -> None:
        self.f = f
        self.name = name
        self.domain = domain

    def vizualize(self, ax: Axes, range: tuple[float, float] = (-np.inf, np.inf)):
        vertices = self.f.constrain_range(
            (range[0], range[1])).vertices
        ax.plot(vertices[:, 0], vertices[:, 1], c='k')
        if self.name != '':
            ax.text(_term_name_x_pos(vertices), 1.2,
                    self.name, ha='center', va='center')
        ax.set_ylim(0, 1.4)
        ax.set_xlabel(self.domain)
        ax.set_ylabel(r'$\mu$', rotation=0)
        return ax
