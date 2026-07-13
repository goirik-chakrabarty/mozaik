"""
This module implements the ObjectList class, which is used to peform type checks
and handle relationships within the Neo Block-Segment-Data hierarchy.
"""

import sys

from neo.core.baseneo import BaseNeo


class ObjectList:
    """
    This class behaves like a list, but has additional functionality
    to handle relationships within Neo hierarchy, and perform type checks.
    """

    def __init__(self, allowed_contents, parent=None):
        # validate allowed_contents and normalize it to a tuple
        if isinstance(allowed_contents, type) and issubclass(allowed_contents, BaseNeo):
            self.allowed_contents = (allowed_contents,)
        else:
            for item in allowed_contents:
                if not issubclass(item, BaseNeo):
                    raise TypeError("Each item in allowed_contents must be a subclass of BaseNeo")
            self.allowed_contents = tuple(allowed_contents)
        self._items = []
        self.parent = parent

    def _handle_append(self, obj):
        if not (
            isinstance(obj, self.allowed_contents)
            or (  # also allow proxy objects of the correct type
                hasattr(obj, "proxy_for") and obj.proxy_for in self.allowed_contents
            )
        ):
            raise TypeError(f"Object is a {type(obj)}. It should be one of {self.allowed_contents}.")

        if self._contains(obj):
            raise ValueError("Cannot add this object because it is already contained within the list")

        # set the child-parent relationship
        if self.parent:
            relationship_name = self.parent.__class__.__name__.lower()
            if relationship_name == "group":
                raise Exception("Objects in groups should not link to the group as their parent")
            current_parent = getattr(obj, relationship_name)
            if current_parent != self.parent:
                # use weakref here? - see https://github.com/NeuralEnsemble/python-neo/issues/684
                setattr(obj, relationship_name, self.parent)

        # PERF PATCH (26-07-13): keep the _contains id-set cache in sync in O(1). obj is appended to
        # self._items by the caller immediately after this returns, so pre-register its id and bump
        # the tracked length. (No-op unless a valid cache for the current list exists.)
        cache = self.__dict__.get("_id_cache")
        if cache is not None and cache[0] == id(self._items):
            cache[2].add(id(obj))
            self.__dict__["_id_cache"] = (cache[0], cache[1] + 1, cache[2])

    def _contains(self, obj):
        # PERF PATCH (26-07-13): identity-membership in O(1) via a cached id-set, tagged by the
        # _items list's id() + length. Original rebuilt [id(x) for x in self._items] on EVERY call,
        # making per-item append O(N) and cross-rank Segment merge (pyNN gather_blocks ->
        # container.merge) O(N^2) — the neo14 P1 get_data slowdown. The cache is rebuilt (O(N)) only
        # when the list is replaced, its length changes unexpectedly, or after unpickling (id()
        # changes); _handle_append keeps it in sync on the hot append path so appends stay O(1).
        items = self._items
        if items is None:
            return False
        cache = self.__dict__.get("_id_cache")
        if cache is None or cache[0] != id(items) or cache[1] != len(items):
            cache = (id(items), len(items), set(map(id, items)))
            self.__dict__["_id_cache"] = cache
        return id(obj) in cache[2]

    def __str__(self):
        return str(self._items)

    def __repr__(self):
        return repr(self._items)

    def __add__(self, objects):
        # todo: decision: return a list, or a new DataObjectList?
        if isinstance(objects, ObjectList):
            return self._items + objects._items
        else:
            return self._items + objects

    def __radd__(self, objects):
        if isinstance(objects, ObjectList):
            return objects._items + self._items
        else:
            return objects + self._items

    def __contains__(self, key):
        return key in self._items

    def __iadd__(self, objects):
        for obj in objects:
            self._handle_append(obj)
        self._items.extend(objects)
        return self

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, i):
        return self._items[i]

    def __len__(self):
        return len(self._items)

    def __setitem__(self, key, value):
        self._items[key] = value
        self.__dict__.pop("_id_cache", None)  # PERF PATCH: in-place replace may change membership

    def append(self, obj):
        self._handle_append(obj)
        self._items.append(obj)

    def extend(self, objects):
        for obj in objects:
            self._handle_append(obj)
        self._items.extend(objects)

    def clear(self):
        self._items = []

    def count(self, value):
        return self._items.count(value)

    def index(self, value, start=0, stop=sys.maxsize):
        return self._items.index(value, start, stop)

    def insert(self, index, obj):
        self._handle_append(obj)
        self._items.insert(index, obj)

    def pop(self, index=-1):
        return self._items.pop(index)

    def remove(self, value):
        return self._items.remove(value)

    def reverse(self):
        raise self._items.reverse()

    def sort(self, *args, key=None, reverse=False):
        self._items.sort(*args, key=key, reverse=reverse)
