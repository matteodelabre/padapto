from collections.abc import Callable, Mapping, MutableMapping
from functools import wraps
from types import GenericAlias
from typing import Any, Concatenate, TypeVar, get_args, get_origin
from weakref import WeakKeyDictionary

from ..signature import Signature


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


type OperatorWithArgTypes[T] = Callable[[tuple[tuple[Any, bool], ...]], T]
type Operator[T] = Callable[[*tuple[Any, ...]], T]


def make_checked_operator[T](
    operator_signature: Any,
    dest_type: Any,
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
        arg_is_out = [True] * len(args)

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
                if arg_type == return_type:
                    arg_is_out[i] = False

                    if not isinstance(arg_value, dest_type):
                        raise TypeError(
                            f"argument #{i} must be of type '{dest_type.__name__}'"
                        )
            elif not isinstance(arg_value, arg_type):
                raise TypeError(f"argument #{i} must be of type '{arg_type.__name__}'")

        # Forward to the original operator, with computed type information
        return operator(tuple(zip(args, arg_is_out, strict=True)))

    return checked_operator


type CallTrace = tuple[str, tuple[Any, ...], Mapping[str, Any]]
_parent_registry: MutableMapping[Signature[Any], CallTrace] = WeakKeyDictionary()


def trace(transparent: bool = False):
    """
    Transform an algebra-producing function to keep a record of its origin.

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
