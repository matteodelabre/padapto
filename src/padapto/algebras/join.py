import dataclasses
from collections.abc import Iterator, Mapping
from functools import partial
from typing import Any, TypeVar

from ..collections import Record
from .signature import (
    Operator,
    Signature,
    extract_algebra_parent,
    make_checked_operator,
    trace,
)


def _joined_operator[T](
    record_type: type[T],
    suboperators: Mapping[str, Operator[Any]],
    args: tuple[Any, ...],
    args_types: tuple[type[Any], ...],
) -> T:
    return record_type(
        **{
            # Compute the value for each field using the respective subalgebra
            key: suboperator(
                *(
                    (
                        getattr(arg_value, key)
                        if isinstance(arg_type, TypeVar)
                        else arg_value
                    )
                    for arg_type, arg_value in zip(args_types, args, strict=True)
                )
            )
            for key, suboperator in suboperators.items()
        }
    )


@trace()
def join[S: Signature[Any]](
    record_type: type[Any] = Record,
    **subalgebras: S,
) -> Signature[Any]:
    """
    Compute the direct product of algebras.

    Given multiple algebras (called "subalgebras") deriving from the same signature,
    this function creates an algebra with the same signature, valued over the Cartesian
    product of all the given subalgebras. Values for each subalgebra are computed
    independently from each other using the original algebra functions.

    This algebra is valid if all the subalgebras are valid.

    Note: The return type should be 'S[record_type]', where 'record_type' is the given
    record type. Unfortunately, Python’s type system is not powerful enough to express
    this yet (see <https://github.com/python/typing/issues/548>).

    :param record_type: record type used for storing algebra values, with a field for
        each subalgebra (default: use a plain dict-like type)
    :param subalgebras: set of subalgebras with a field name for each one; this name
        will be used to fill the record type (at least one subalgebra must be provided)
    :returns: created algebra
    """
    if not subalgebras:
        raise TypeError("join: at least one subalgebra must be provided")

    signature = type(next(iter(subalgebras.values())))

    # Check that all provided subalgebras derive from the same signature
    for key, subalgebra in subalgebras.items():
        if not isinstance(subalgebra, signature):
            first_key = next(iter(subalgebras.keys()))
            raise TypeError(
                f"join: subalgebras for '{key}' and '{first_key}' derive from "
                "different signatures"
            )

    elements: dict[str, Operator[Any]] = {}

    for field in dataclasses.fields(signature):
        suboperators = {
            key: getattr(subalgebra, field.name)
            for key, subalgebra in subalgebras.items()
        }
        op = partial(_joined_operator, record_type, suboperators)
        elements[field.name] = make_checked_operator(field.type, record_type, op)

    return signature(**elements)


def get_subrecord(record: Any, field: str) -> Any:
    """Access a field inside a record using dotted notation."""
    for entry in field.split("."):
        record = getattr(record, entry)

    return record


def get_subalgebras(
    algebra: Signature[Any],
    *keys: str | tuple[str, ...],
    _prefix: tuple[str, ...] = (),
) -> Iterator[tuple[str, Signature[Any]]]:
    """
    Recursively resolve fields of a joined algebra using dotted notation.

    :param algebra: joined algebra
    :param keys: list of keys to resolve, specifying fields with dotted notation and
        optionally using stars ('*') to refer to all fields of a given depth
    :returns: iterator of pairs containing each resolved field’s complete name in dotted
        notation and the corresponding subalgebra
    """
    for key in keys:
        result = algebra

        if isinstance(key, str):
            key = tuple(key.split("."))

        for i, entry in enumerate(key):
            if (
                subalgebras := extract_algebra_parent(result, "join", kwargs=True)
            ) is None:
                location = ".".join(_prefix + key[:i])

                if location:
                    raise TypeError(f"algebra of '{location}' is not joined")
                else:
                    raise TypeError("provided algebra is not joined")

            if entry == "*":
                for name, subalgebra in subalgebras.items():
                    yield from get_subalgebras(
                        subalgebra, key[i + 1 :], _prefix=_prefix + key[:i] + (name,)
                    )

                break
            else:
                if entry not in subalgebras:
                    location = ".".join(_prefix + key[:i])

                    if location:
                        raise AttributeError(
                            f"'{entry}' is not a field of '{location}'"
                        )
                    else:
                        raise AttributeError(
                            f"'{entry}' is not a field of the provided algebra"
                        )

                result = subalgebras[entry]
        else:
            yield ".".join(_prefix + key), result
