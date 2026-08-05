"""A tiny plugin registry so every pipeline stage can be swapped from a config file.

Adding a new approach is a matter of writing a class and decorating it::

    @RETRIEVERS.register("my_retriever")
    class MyRetriever(Retriever):
        ...

and then referencing ``name: my_retriever`` in an experiment YAML.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Generic, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str) -> None:
        self.kind = kind
        self._items: dict[str, Callable[..., T]] = {}

    def register(self, name: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            if name in self._items:
                raise ValueError(f"{self.kind} '{name}' is already registered")
            self._items[name] = factory
            return factory

        return decorator

    def build(self, name: str, **kwargs: Any) -> T:
        try:
            factory = self._items[name]
        except KeyError:
            raise KeyError(
                f"Unknown {self.kind} '{name}'. Available: {', '.join(self.names())}"
            ) from None
        return factory(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._items)
