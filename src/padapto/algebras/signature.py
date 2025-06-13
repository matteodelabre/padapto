from collections.abc import Callable, MutableMapping
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


_algebra_metadata_registry: MutableMapping[Signature[Any], dict[Any, Any]] = (
    WeakKeyDictionary()
)


def set_algebra_metadata[S: Signature[Any]](
    algebra: S,
    key: Any,
    value: Any,
) -> S:
    """Associate a metadata attribute to an algebra."""
    if algebra not in _algebra_metadata_registry:
        _algebra_metadata_registry[algebra] = {}

    _algebra_metadata_registry[algebra][key] = value
    return algebra


def copy_algebra_metadata[S: Signature[Any]](source: Signature[Any], target: S) -> S:
    """Copy all metadata attributes from a source algebra to a target algebra."""
    if source in _algebra_metadata_registry:
        _algebra_metadata_registry[target] = _algebra_metadata_registry[source].copy()

    return target


def get_algebra_metadata(algebra: Signature[Any], key: Any, default: Any = None) -> Any:
    """Retrieve a metadata attribute from an algebra given its key."""
    return _algebra_metadata_registry.get(algebra, {}).get(key, default)
