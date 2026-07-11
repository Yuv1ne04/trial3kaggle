"""A minimal, typed registry used for dependency injection from configuration.

Each component family (models, losses, ...) has its own :class:`Registry`.
Implementations register themselves with a decorator; the trainer/builder then
constructs them purely from the string ``name`` and ``params`` in the YAML, so
no source change is needed to add or switch components.
"""

from __future__ import annotations

from typing import Any, Callable, Generic, Iterable, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    """A name -> factory registry with decorator-based registration.

    Attributes:
        name: Human-readable family name (for error messages).
    """

    def __init__(self, name: str) -> None:
        """Initialise an empty registry.

        Args:
            name: The family name (e.g. ``"models"``).
        """
        self.name = name
        self._factories: dict[str, Callable[..., T]] = {}

    def register(self, key: str) -> Callable[[Callable[..., T]], Callable[..., T]]:
        """Return a decorator that registers a factory under ``key``.

        Args:
            key: The lookup name used in configuration.

        Returns:
            A decorator that registers and returns the callable unchanged.

        Raises:
            KeyError: If ``key`` is already registered.
        """
        def decorator(factory: Callable[..., T]) -> Callable[..., T]:
            lowered = key.lower()
            if lowered in self._factories:
                raise KeyError(f"{self.name}: '{key}' already registered")
            self._factories[lowered] = factory
            return factory

        return decorator

    def build(self, key: str, **kwargs: Any) -> T:
        """Instantiate a registered component.

        Args:
            key: The registered name.
            **kwargs: Keyword arguments forwarded to the factory.

        Returns:
            The constructed component.

        Raises:
            KeyError: If ``key`` is not registered.
        """
        factory = self.get(key)
        return factory(**kwargs)

    def get(self, key: str) -> Callable[..., T]:
        """Return the factory registered under ``key``.

        Args:
            key: The registered name.

        Returns:
            The factory callable.

        Raises:
            KeyError: If ``key`` is not registered (lists available keys).
        """
        lowered = key.lower()
        if lowered not in self._factories:
            raise KeyError(
                f"{self.name}: '{key}' is not registered. "
                f"Available: {sorted(self._factories)}"
            )
        return self._factories[lowered]

    def available(self) -> Iterable[str]:
        """Return the sorted registered names."""
        return sorted(self._factories)


#: Component registries, one per family.
MODELS: Registry[Any] = Registry("models")
LOSSES: Registry[Any] = Registry("losses")
METRICS: Registry[Any] = Registry("metrics")
OPTIMIZERS: Registry[Any] = Registry("optimizers")
SCHEDULERS: Registry[Any] = Registry("schedulers")
DATASETS: Registry[Any] = Registry("datasets")
CALLBACKS: Registry[Any] = Registry("callbacks")
