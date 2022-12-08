"""
Definition of custom data types and data structures
"""
from copy import deepcopy
import numpy


class ProtectedDict(dict):
    """
    ProtectedDict is a child class of built-in type dict. It serves to protect mutable type dict
    from being modified when desired.
    """

    __protected = False

    def protect(self, protect):
        """Allow or deny modifying dictionary"""
        self.__protected = protect

    def _raise_error(self, operation, object_type):
        if self.__protected:
            raise TypeError(
                f"{operation} is not supported as the {object_type} is protected"
            )

    def __setitem__(self, key, value):
        self._raise_error("__setitem__", "dictionary")
        if isinstance(value, numpy.ndarray):
            value.setflags(write=False)
        return dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        self._raise_error("__delitem__", "dictionary")
        return dict.__delitem__(self, key)

    def update(self, e=None, **f):
        self._raise_error("update", "dictionary")
        super().update(e, **f)
        for key, value in self.items():
            if isinstance(value, numpy.ndarray):
                value.setflags(write=False)

    def pop(self, k, d=None):
        self._raise_error("pop", "dictionary")
        super().pop(k, d)

    def popitem(self):
        self._raise_error("popitem", "dictionary")
        super().popitem()

    def setdefault(self, k, d=None):
        self._raise_error("setdefault", "dictionary")
        super().setdefault(k, d)
        for key, value in self.items():
            if isinstance(value, numpy.ndarray):
                value.setflags(write=False)

    def clear(self):
        self._raise_error("clear", "dictionary")
        super().clear()

    def copy(self):
        self.__protected = False
        dict_copy = dict()
        for key, value in self.items():
            dict_copy[key] = deepcopy(value)
        self.__protected = True
        return dict_copy


class DataTypeProtector:
    """Accessor of Protected data types in order to modify the protected data types using with statement"""

    def __init__(self, protected_dtype):
        self.pd = protected_dtype

    def __enter__(self):
        self.pd.protect(False)
        return self.pd

    def __exit__(self, type, value, traceback):
        self.pd.protect(True)


class ExtendableEnum:
    """Helper class to create enumerations. In contrast to std enum.Enum, it can be extended"""

    @classmethod
    def enum_len(cls):
        """Return the actual number of enumerations"""
        return 1 + max(cls.values())

    @classmethod
    def items(cls):
        return [
            (attr, getattr(cls, attr))
            for attr in dir(cls)
            if not callable(getattr(cls, attr)) and not attr.startswith("__")
        ]

    @classmethod
    def keys(cls):
        return [item[0] for item in cls.items()]

    @classmethod
    def values(cls):
        return [item[1] for item in cls.items()]


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class classproperty(object):
    """Equivalent of @property for class rather than instance"""

    def __init__(self, f):
        self.f = classmethod(f)

    def __get__(self, *a):
        return self.f.__get__(*a)()
