from collections import Counter
from collections.abc import Iterable
from typing import Any, Self, SupportsIndex

from immutables import Map


class Record:
    """Immutable dict-like object providing attribute access."""

    def __init__(self, **kwargs):
        """Create a record with the given attributes."""
        object.__setattr__(self, "_Record__data", Map(**kwargs))

    def __eq__(self, other):
        return isinstance(other, type(self)) and self.__data == other.__data

    def __hash__(self):
        return hash(self.__data)

    def __getattr__(self, key):
        if key in self.__data:
            return self.__data[key]

        raise AttributeError(f"record has no field '{key}'")

    def __setattr__(self, key, value):
        raise NotImplementedError(f"cannot assign to field '{key}'")

    def __repr__(self):
        values = ", ".join(f"{key}={value!r}" for key, value in self.__data.items())
        return f"{type(self).__name__}({values})"


class Multiset[T](tuple[T, ...]):
    """Immutable tuple of hashable elements with no preferred ordering."""

    _counter: Counter[T]
    _hash: int

    def __new__(cls, elements: Iterable[T] = ()) -> Self:
        """Create a multiset containing the given elements."""
        self = super().__new__(cls, elements)
        self._counter = Counter(self)
        self._hash = hash(Map(self._counter))
        return self

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self)) and self._counter == other._counter

    def __ne__(self, other: Any) -> bool:
        return not isinstance(other, type(self)) or self._counter != other._counter

    def __le__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return self._counter <= other._counter

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return self._counter < other._counter

    def __ge__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return self._counter >= other._counter

    def __gt__(self, other: Any) -> bool:
        if not isinstance(other, type(self)):
            return NotImplemented

        return self._counter > other._counter

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return f"{type(self).__name__}({super().__repr__()})"

    def __add__(self, other: Any) -> Self:
        return type(self)(super().__add__(other))

    def __mul__(self, other: SupportsIndex) -> Self:
        return type(self)(super().__mul__(other))

    def __rmul__(self, other: SupportsIndex) -> Self:
        return type(self)(super().__rmul__(other))

    def count(self, value: T) -> int:
        """Count the number of occurrences of a value in the multiset."""
        return self._counter.get(value, 0)
