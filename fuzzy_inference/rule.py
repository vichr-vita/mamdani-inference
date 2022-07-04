from __future__ import annotations
from distutils.log import Log
from enum import Enum
from typing import Collection


class Rule:

    def __init__(self) -> None:
        self.antecedents: Collection[tuple[str, str]] = []
        self.consequent: tuple[str, str] | None = None

    def IF(self, antecedents: Collection[tuple[str, str]]) -> Rule:
        self.antecedents = antecedents
        return self

    def THEN(self, consequent: tuple[str, str]) -> Rule:
        self.consequent = consequent
        return self
