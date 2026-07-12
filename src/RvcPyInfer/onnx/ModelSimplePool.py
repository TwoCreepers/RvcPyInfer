from typing import Callable
from collections import OrderedDict

class ModelSimplePool[TKey, TModel]:
    def __init__(self, factory: Callable[[TKey], TModel], permanent_size: int = 1) -> None:
        self._factory = factory
        self._pools = OrderedDict[TKey, TModel]()
        self._permanent_size = permanent_size

    def remove(self, key: TKey) -> None:
        del self._pools[key]

    def clear(self) -> None:
        self._pools.clear()

    def get(self, key: TKey) -> TModel:
        model = self._pools.get(key)
        if model is not None:
            self._pools.move_to_end(key)
        else:
            model = self._factory(key)
            self._pools[key] = model
            if len(self._pools) > self._permanent_size:
                self._pools.popitem(last=False)
        return model