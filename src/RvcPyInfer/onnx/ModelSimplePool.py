import threading
import contextlib
from typing import Callable
from collections import OrderedDict

class ModelSimplePool[TKey, TModel]:
    def __init__(self, factory: Callable[[TKey], TModel], permanent_size: int = 1, max_borrow: int = 5) -> None:
        self._factory = factory
        self._pools_lock = threading.Lock()
        self._pools = OrderedDict[TKey, TModel]()
        self._permanent_size = permanent_size
        self._max_borrow = max_borrow
        self._max_borrow_sem = threading.Semaphore(self._max_borrow)

    def remove(self, key: TKey):
        with self._pools_lock:
            del self._pools[key]

    def clear(self):
        with self._pools_lock:
            self._pools.clear()
    
    @contextlib.contextmanager
    def borrow(self, key: TKey):
        self._max_borrow_sem.acquire()
        with self._pools_lock:
            model = self._pools.get(key)
            if model is not None:
                self._pools.move_to_end(key)
            else:
                model = self._factory(key)
                self._pools[key] = model
                if len(self._pools) > self._permanent_size:
                    self._pools.popitem(last=False)
        try:
            yield model
        finally:
            self._max_borrow_sem.release()