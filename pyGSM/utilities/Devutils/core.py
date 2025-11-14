"""
Provides a set of singleton objects that can declare their purpose a little bit better than None can
"""
import itertools
# import enum
import types
import numbers

__all__ = [
    "default",
    "is_default",
    "handle_default",
    "uninitialized",
    "is_uninitialized",
    "handle_uninitialized",
    "missing",
    "is_missing",
    "is_interface_like",
    "is_dict_like",
    "is_option_spec_like",
    "destructure_option_spec",
    "is_list_like",
    "is_number",
    "is_int",
    "is_atomic",
    "cached_eval",
    "merge_dicts",
    "str_comp",
    "str_is",
    "str_in",
    "str_elide",
    "resolve_key_collision",
    "merge_dicts",
    "context_wrap",
    "slice_dict",
    "dict_take"
]

class SingletonType:
    """
    A base type for singletons
    """
    __slots__ = []
    def __eq__(self, other):
        return self is other or (
                hasattr(other, '__module__')
                and type(self).__name__ == type(other).__name__
                and self.__module__ == other.__module__
        )
    def __hash__(self):
        return hash(self.__module__ +'.'+ type(self).__name__)

class DefaultType(SingletonType):
    """
    A type for declaring an argument should use its default value (for when `None` has meaning)
    """
    __is_default__ = True
default=DefaultType()

class MissingType(SingletonType):
    """
    A type for declaring a value is missing (for when `None` has meaning)
    """
    __is_missing__ = False
missing=MissingType()

class UninitializedType(SingletonType):
    """
    A type for declaring an argument should use its default value (for when `None` has meaning)
    """
    __is_uninitialized__ = True
uninitialized = UninitializedType()

def is_atomic(obj,
              interface_types=(str, bool, numbers.Number),
              exlusion_types=None,
              implementation_props=None
              ):
    return is_interface_like(obj, interface_types, exlusion_types, implementation_props)

def is_number(obj,
              interface_types=(numbers.Number,),
              exlusion_types=None,
              implementation_props=None
              ):
    return is_interface_like(obj, interface_types, exlusion_types, implementation_props)

def is_int(obj,
           interface_types=(numbers.Integral,),
           exlusion_types=None,
           implementation_props=None
           ):
    return is_interface_like(obj, interface_types, exlusion_types, implementation_props)

def is_interface_like(obj, interface_types, exlusion_types, implementation_attrs):
    return (
            (exlusion_types is None)
            or not isinstance(obj, exlusion_types)
    ) and (
            (interface_types is not None and isinstance(obj, interface_types))
            or (
                implementation_attrs is not None
                and all(hasattr(obj, a) for a in implementation_attrs)
            )
    )

def is_dict_like(obj,
                 interface_types=(dict, types.MappingProxyType),
                 exlusion_types=None,
                 implementation_props=('items',)
                 ):
    return is_interface_like(obj, interface_types, exlusion_types, implementation_props)

def is_list_like(obj,
                 interface_types=(list, tuple),
                 exlusion_types=(str, dict, type),
                 implementation_props=('__getitem__',)
                 ):
    return is_interface_like(obj, interface_types, exlusion_types, implementation_props)

def is_default(obj, allow_None=True):
    if allow_None and obj is None:
        return True

    return (
            obj is default
            or isinstance(obj, DefaultType)
            or (hasattr(obj, '__is_default__') and obj.__is_default__)
    )

def is_option_spec_like(obj, allow_enums=True):
    method, opts = destructure_option_spec(obj, allow_enums)
    return method is not None

def destructure_option_spec(spec, allow_enums=True, method_key='method'):
    if isinstance(spec, (str, bool)) or is_number(spec) or is_default(spec):
        opts = {}
        method = spec
    elif (
            allow_enums and
            hasattr(spec, 'name') and hasattr(spec, 'value')
    ):  # enum
        method = spec.value
        opts = {}
    elif is_dict_like(spec):
        opts = spec.copy()
        method = opts.pop(method_key, None)
    elif callable(spec):
        method = spec
        opts = {}
    else:
        try:
            method, opts = spec
        except TypeError:
            method = None
            opts = None

    return method, opts

def handle_default(opt, default_value, allow_None=True):
    if is_default(opt, allow_None=allow_None):
        return default_value
    else:
        return opt

def is_uninitialized(obj, allow_None=True):
    if allow_None and obj is None:
        return True

    return (
            obj is uninitialized
            or isinstance(obj, UninitializedType)
            or (hasattr(obj, '__is_uninitialized__') and obj.__is_uninitialized__)
    )

def handle_uninitialized(opt, initializer, allow_None=True, args=(), kwargs=None):
    if is_uninitialized(opt, allow_None=allow_None):
        return initializer(*args, **({} if kwargs is None else kwargs))
    else:
        return opt

def is_missing(obj, allow_None=True):
    if allow_None and obj is None:
        return True

    return (
            obj is missing
            or isinstance(obj, MissingType)
            or (hasattr(obj, '__is_missing__') and obj.__is_missing__)
    )


def cached_eval(cache, key, generator, *,
                condition=None,
                args=(),
                kwargs=None):
    condition = (condition is None or condition(key))
    if not condition:
        if kwargs is None: kwargs = {}
        return generator(*args, **kwargs)

    if key in cache:
        return cache[key]

    if kwargs is None: kwargs = {}
    val = generator(*args, **kwargs)
    cache[key] = val

    return val

def str_comp(str_val, test, test_val):
    return isinstance(str_val, str) and test(str_val, test_val)
def str_is(str_val, test_val):
    return isinstance(str_val, str) and str_val == test_val
def str_in(str_val, test_vals):
    return isinstance(str_val, str) and str_val in test_vals
def str_elide(long_str, width=80, placeholder='...'):
    l = len(long_str)
    if l > width:
        total_width = width - len(placeholder)
        l = total_width // 2 + (total_width % 2)
        r = total_width // 2
        long_str = long_str[:l] + placeholder + long_str[-r:]
    return long_str

def resolve_key_collision(a, b, k, merge_iterables=True):
    if is_dict_like(a[k]):
        if not is_dict_like(b[k]):
            return b[k]
            # raise ValueError(f"can't resolve key collision on key {k} between {a[k]} and {b[k]}")
        return merge_dicts(a[k], b[k], resolve_key_collision, merge_iterables=merge_iterables)
    elif merge_iterables and isinstance(a[k], set):
        if not isinstance(b[k], set):
            return b[k]
        a = set(a[k])
        a.update(b)
        return a
    elif merge_iterables and is_list_like(a[k]):
        if not is_list_like(b[k]):
            return type(a[k])(
                itertools.chain(a[k], b[k])
            )
        else:
            return type(a[k])(
                itertools.chain(a[k], [b[k]])
            )
    elif merge_iterables and is_list_like(b[k]):
        return type(b[k])(
            itertools.chain([a[k]], b[k])
        )
    else:
        return b[k]

def merge_dicts(a, b, collision_handler=None, merge_iterables=True):
    key_inter = a.keys() & b.keys()
    diff_a = a.keys() - key_inter
    diff_b = b.keys() - key_inter
    dd = {k: a[k] for k in diff_a}
    dd.update((k, b[k]) for k in diff_b)
    if len(key_inter) > 0:
        if collision_handler is None:
            collision_handler = lambda a, b, k:resolve_key_collision(a, b, k, merge_iterables=merge_iterables)
        dd.update(
            (k, collision_handler(a, b, k))
            for k in key_inter
        )

    return dd

class context_wrap:
    def __init__(self, obj):
        self.obj = obj
    def __enter__(self):
        if hasattr(self.obj, '__enter__'):
            return self.obj.__enter__()
        else:
            return self.obj
    def __exit__(self, exc_type, exc_val, exc_tb):
        if hasattr(self.obj, '__exit__'):
            return self.obj.__exit__(exc_type, exc_val, exc_tb)

class slice_dict:
    __slots__ = ["dict_obj"]
    def __init__(self, dict_obj:types.MappingProxyType):
        self.dict_obj = dict_obj
    def __getitem__(self, item):
        return dict_take(self.dict_obj, item)

def dict_take(dict_obj:types.MappingProxyType, spec):
    if is_number(spec):
        if spec < 0:
            spec = len(dict_obj) - spec
        for k in itertools.islice(dict_obj.keys(), spec, -1):
            return k, dict_obj[k]
    elif isinstance(spec, slice):
        return dict(
            itertools.islice(dict_obj.items(), spec.start, spec.stop, spec.step)
        )
    else:
        if not is_int(spec[0]):
            return {
                k:dict_obj[k]
                for k in spec
            }

        results = {}
        rem = set(spec)
        for n,k in enumerate(dict_obj.keys()):
            if n in rem:
                results[n] = (k, dict_obj[k])
                rem.remove(n)
                if len(rem) == 0: break
        else:
            raise IndexError("can't take elements {rem}")

        return dict(results[n] for n in spec)