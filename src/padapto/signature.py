from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from typing import Self

type Comparator[T] = Callable[[T, T], bool]


@dataclass(frozen=True, slots=True)
class Signature[T]:
    """
    Abstract base class for signatures.

    All signatures have a choice function ('choose') with a neutral element ('null').
    Instances of signatures are called algebras. An algebra is valid if and only if:

    - 'null' is a neutral element for the choice function
      (i.e., choose(x, null) = choose(null, x) for all values x),

    - the choice function is commutative
      (i.e., choose(x, y) = choose(y, x) for all values x and y),

    - the choice function is associative
      (i.e., choose(x, choose(y, z)) = choose(choose(x, y), z) for all values x, y, z),

    - the choice function distributes over all algebra functions
      (i.e., choose(f(x, y), f(x, z)) = f(x, choose(y, z)) for any x, y and z, for any
      function f of the algebra, and for any argument of f).

    Constant elements (such as the 'null' field) must be declared as nullary functions
    (i.e., `Callable[[], T]` in the signature and `lambda: value` in the algebra)

    Note: The distributivity property is also referred to as the "Algebraic version of
    Bellman’s principle" in "Towards a discipline of dynamic programming" by Giegerich,
    Meyer and Steffen (2002) and related literature.

    These properties are not automatically checked in all cases; users must make sure
    that their algebras are valid.
    """

    # Neutral element for the choice function
    null: Callable[[], T]

    # Function to choose between two solutions
    choose: Callable[[T, T], T]

    def multichoose(self, *args: T) -> T:
        """Choose between any number of solutions."""
        return reduce(self.choose, args, self.null())

    def __or__[R](self, fun: Callable[[Self], R]) -> R:
        """Use this algebra as the first argument of a pipable function."""
        return fun(self)

    def natural_order(self) -> Comparator[T]:
        """
        Create a comparison function using the natural order of this algebra.

        The natural order exists if the choice function is idempotent (i.e., if
        choice(x, x) = x for any x) and is total if the algebra is conservative
        (i.e., if choice(x, y) ∈ {x, y} for any x and y).

        When it exists, this order is always monotonous with respect to other algebra
        functions, i.e., if x ⩽ y, then f(x, z) ⩽ f(y, z) for any x, y and z, for any
        function f of the algebra and for any argument of f.

        :returns: natural order comparison function
        """

        def natural_order_le(left: T, right: T) -> bool:
            return self.choose(left, right) == left

        return natural_order_le
