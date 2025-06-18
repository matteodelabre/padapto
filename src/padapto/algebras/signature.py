from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from functools import reduce, wraps
from types import GenericAlias
from typing import Any, Concatenate, Self, TypeVar, get_args, get_origin
from weakref import WeakKeyDictionary


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


def pipable[F, **P, T](
    func: Callable[Concatenate[F, P], T],
) -> Callable[P, Callable[[F], T]]:
    """
    Transform a function to be usable through an algebra pipe.

    If `func` is a function whose first argument is an algebra, the resulting wrapped
    function will accept all arguments except the first one, and wait for the algebra
    argument to be provided on the left through the | operator.

    :param func: function to wrap
    :returns wrapped function
    """

    @wraps(func)
    def rest_func(*args: P.args, **kwargs: P.kwargs) -> Callable[[F], T]:
        def first_func(first: F) -> T:
            return func(first, *args, **kwargs)

        return first_func

    return rest_func


type Comparator[T] = Callable[[T, T], bool]


def make_natural_order[T](algebra: Signature[T]) -> Comparator[T]:
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
        return algebra.choose(left, right) == left

    return natural_order_le


type OperatorWithArgTypes[T] = Callable[[tuple[Any, ...], tuple[type[Any], ...]], T]
type Operator[T] = Callable[[*tuple[Any, ...]], T]


def make_checked_operator[T](
    operator_signature: Any,
    dest_type: type[Any],
    operator: OperatorWithArgTypes[T],
) -> Operator[T]:
    """Wrap an operator to check that its arguments respect a signature on each call."""
    if get_origin(operator_signature) is not Callable:
        raise TypeError(
            f"unsupported operator type '{operator_signature}', must be 'Callable'"
        )

    # Analyze the signature of the operator to wrap
    args_types, return_type = get_args(operator_signature)
    args_types = tuple(args_types)

    if __debug__ and not isinstance(return_type, TypeVar):
        raise TypeError(f"return type of signature must be generic, not '{return_type}")

    if (
        args_types
        and isinstance(args_types[-1], GenericAlias)
        and get_origin(args_types[-1]) is tuple
        and get_args(args_types[-1])[1] == ...
    ):
        # Handle variadic operators, where the last argument can be repeated
        # an arbitrary number of times
        variadic = get_args(args_types[-1])[0]
    else:
        variadic = None

    def checked_operator(*args: Any) -> T:
        # When called, check that the given arguments respect the operator signature
        if __debug__:
            if variadic is not None:
                req_args = len(args_types) - 1

                if len(args) < req_args:
                    raise TypeError(
                        f"expected at least {req_args} arguments, got {len(args)}"
                    )

                loc_args_types = args_types[:-1] + (variadic,) * (len(args) - req_args)
            else:
                if len(args) != len(args_types):
                    raise TypeError(
                        f"expected {len(args_types)} arguments, got {len(args)}"
                    )

                loc_args_types = args_types

            for i, (arg_type, arg_value) in enumerate(
                zip(loc_args_types, args, strict=True)
            ):
                if isinstance(arg_type, TypeVar):
                    if not isinstance(arg_value, dest_type):
                        raise TypeError(
                            f"argument #{i} must be of type '{dest_type.__name__}'"
                        )
                elif not isinstance(arg_value, arg_type):
                    raise TypeError(
                        f"argument #{i} must be of type '{arg_type.__name__}'"
                    )

        # Forward to the original operator, with computed type information
        return operator(args, loc_args_types)

    return checked_operator


type CallTrace = tuple[str, tuple[Any, ...], Mapping[str, Any]]
_parent_registry: MutableMapping[Signature[Any], CallTrace] = WeakKeyDictionary()


def trace(transparent: bool = False):
    """
    Tranform an algebra-producing function to keep a record of its origin.

    When an algebra `alg` is produced by the wrapped function, the producing function’s
    name and arguments can be retrieved using the :fun:`get_algebra_parent` function.

    :param transparent: if True, assume that the first argument of the function will be
        an algebra itself, and inherit the origin of that algebra as the origin of the
        produced algebra
    """

    def tracer[S: Signature[Any], **P](func: Callable[P, S]) -> Callable[P, S]:
        @wraps(func)
        def traced_func(*args: P.args, **kwargs: P.kwargs) -> S:
            result = func(*args, **kwargs)

            if transparent:
                assert isinstance(args[0], Signature)

                if args[0] in _parent_registry:
                    _parent_registry[result] = _parent_registry[args[0]]
            else:
                _parent_registry[result] = (traced_func.__name__, args, kwargs)

            return result

        return traced_func

    return tracer


def get_algebra_parent(algebra: Signature[Any]) -> CallTrace | None:
    """Retrieve the original function used to produce an algebra, if any."""
    return _parent_registry.get(algebra)


def extract_algebra_parent(
    algebra: Signature[Any],
    maker: str,
    index: int = 0,
    kwargs: bool = False,
) -> Any:
    """
    Check that a given algebra has been produced by a function.

    If the function matches, extract the specified argument that was given.

    :param algebra: algebra to check the origin of
    :param maker: expected producing function
    :param index: index of the positional argument to extract
    :param kwargs: if True, extract all keyword arguments
    :returns: extract arguments if the function matches, None otherwise
    """
    if (parent := get_algebra_parent(algebra)) is None:
        return None

    if parent[0] != maker:
        return None

    if kwargs:
        return parent[2]
    else:
        return parent[1][index]
