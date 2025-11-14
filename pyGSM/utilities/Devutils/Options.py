"""
Provides functionality for managing large sets of options
"""

import os, inspect
from typing import Callable
from . import core

__all__ = [
    "OptionsSet",
    "OptionsMethodDispatch"
]

class OptionsSet:
    """
    Provides a helpful manager for those cases where
    there are way too many options and we need to filter
    them across subclasses and things
    """

    def __init__(self, *d, **ops):
        if len(d) > 0:
            if isinstance(d[0], dict):
                self.ops = d[0]
                self.ops.update(ops)
            else:
                self.ops = dict(d, **ops)
        else:
            self.ops = ops
    def __getitem__(self, item):
        return self.ops[item]
    def __setitem__(self, key, value):
        self.ops[key] = value
    def __delitem__(self, item):
        del self.ops[item]
    def __getattr__(self, item):
        return self.ops[item]
    def __setattr__(self, key, value):
        if key == "ops":
            super().__setattr__(key, value)
        else:
            self.ops[key] = value
    def __delattr__(self, item):
        del self.ops[item]
    def __hasattr__(self, key):
        return key in self.ops
    def update(self, **ops):
        self.ops.update(**ops)

    def keys(self):
        return self.ops.keys()
    def items(self):
        return self.ops.items()

    def save(self, file, mode=None, attribute=None):
        self.serialize(file)
    @classmethod
    def load(cls, file, mode=None, attribute=None):
        cls(cls.deserialize(file, mode=mode, attribute=attribute))

    def extract_kwarg_keys(self, obj):
        args, _, _, defaults, _, _, _  = inspect.getfullargspec(obj)
        if args is None:
            return None
        ndef = len(defaults) if defaults is not None else 0
        return tuple(args[-ndef:])
    def get_props(self, obj):
        if isinstance(obj, (list, tuple)):
            return sum(
                (self.get_props(o) for o in obj),
                ()
            )

        try:
            props = obj.__props__
        except AttributeError:
            try:
                annotations = obj.__annotations__
            except AttributeError:
                annotations = {}
            if len(annotations) == 0:
                props = self.extract_kwarg_keys(obj)
            else:
                props = tuple(annotations.keys())

        if props is None:
            raise AttributeError("{}: object {} needs props to filter against".format(
                type(self).__name__,
                self
            ))
        return props

    def bind(self, obj, props=None):
        for k,v in self.filter(obj, props=props).items():
            setattr(obj, k, self.ops[k])
    def filter(self, obj, props=None):
        if props is None:
            props = self.get_props(obj)
        ops = self.ops
        return {k:ops[k] for k in ops.keys() & set(props)}
    def exclude(self, obj, props=None):
        if props is None:
            props = self.get_props(obj)
        ops = self.ops
        return {k:ops[k] for k in ops.keys() - set(props)}
    def split(self, obj, props=None):
        if props is None:
            props = self.get_props(obj)
        return self.filter(obj, props=props), self.exclude(obj, props=props)


class OptionsMethodDispatch:
    def __init__(self,
                 methods_table:'dict|Callable[[], dict]',
                 attributes_map=None,
                 default_method=None,
                 methods_enum=None,
                 case_insensitive=True,
                 allow_custom_methods=True,
                 ignore_bad_enum_keys=False,
                 method_key='method'):
        if not hasattr(methods_table, 'items'):
            self.methods_table_generator = methods_table
            self.methods_table = {}
        else:
            self.methods_table = methods_table
            self.methods_table_generator = None
        self.attributes_map = attributes_map
        self.case_insensitive = case_insensitive
        self.method_key = method_key
        self.methods_enum = methods_enum
        self.default_method = default_method
        self.allow_custom_methods = allow_custom_methods
        self.ignore_bad_enum_keys = ignore_bad_enum_keys

    def register(self, method_name, method, base_attributes=None):
        self.methods_table[method_name] = method
        if base_attributes is not None:
            if self.attributes_map is None:
                self.attributes_map = {}
            if isinstance(base_attributes, str):
                base_attributes = (base_attributes,)
            self.attributes_map[tuple(base_attributes)] = method_name

    def load_methods_table(self):
        if self.methods_table_generator is not None:
            return dict(self.methods_table_generator(), **self.methods_table)
        else:
            return self.methods_table

    def _lookup_method(self, method, opts):
        if method is None and self.attributes_map is not None:
            for params, method_name in sorted(
                    self.attributes_map.items(),
                    key=lambda kt: -len(kt[0])
            ):
                if params is not None and all(p in opts for p in params):
                    return method_name

        return method

    def prep_method_spec(self, method_spec):
        if isinstance(method_spec, str) or core.is_default(method_spec):
            opts = {}
            method = method_spec
        elif hasattr(method_spec, 'name') and hasattr(method_spec, 'value'): # enum
            method = method_spec.value
            opts = {}
        elif core.is_dict_like(method_spec):
            opts = method_spec.copy()
            method = self._lookup_method(opts.pop(self.method_key, None), opts)
        elif callable(method_spec):
            method = method_spec
            opts = {}
        else:
            method, opts = method_spec
            method = self._lookup_method(method, opts)

        return method, opts

    def resolve(self, method_spec):
        method, opts = self.prep_method_spec(method_spec)
        if self.allow_custom_methods and callable(method):
            return method, opts

        methods_table = self.load_methods_table()
        if (
                self.methods_enum is not None
                and not core.is_default(method, allow_None=True)
                and method not in methods_table
        ):
            if self.case_insensitive and isinstance(method, str):
                try:
                    method = self.methods_enum(method)
                except ValueError:
                    if self.ignore_bad_enum_keys:
                        try:
                            method = self.methods_enum(method)
                        except ValueError:
                            ...
                    else:
                        method = self.methods_enum(method.lower())
            else:
                if self.ignore_bad_enum_keys:
                    try:
                        method = self.methods_enum(method)
                    except ValueError:
                        ...
                else:
                    method = self.methods_enum(method)

            if hasattr(method, 'value'):
                return methods_table.get(
                    method,
                    methods_table.get(method.value, methods_table.get(self.default_method))
                ), opts
            else:
                return methods_table.get(
                method,
                methods_table.get(self.default_method)
            ), opts
        else:
            if self.case_insensitive and isinstance(method, str) and not method in methods_table:
                method = method.lower()

            return methods_table.get(
                method,
                methods_table.get(self.default_method)
            ), opts