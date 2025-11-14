"""
Basic layer for Schema validation that provides a superset of JSON schema validation
"""
import enum
import importlib
import re
import sys
import numbers

from . import core, Options as opts

__all__ = [
    "Schema"
]

class JSONSchemaTypes(enum.Enum):
    String = "string"
    Null = "null"
    Boolean = "boolean"
    Object = "object"
    Number = "number"
    Array = "array"

class TypeValidator:
    schema_types = JSONSchemaTypes

    def __init__(self, type_obj):
        self.types, self.validators = self.get_validators(type_obj)

    allow_imports = True
    base_type_map = {
        'str':str,
        'string':str,
        'int':int,
        'integer':int,
        'float':float,
        'real':float,
        'complex':complex,
        'bool':bool
    }
    @classmethod
    def resolve_typestr(cls, type_spec):
        if type_spec.lower() in cls.base_type_map:
            return cls.base_type_map[type_spec]
        else:
            type_spec = cls.base_type_map.get(type_spec, type_spec)
            if isinstance(type_spec, str):
                try:
                    type_spec = JSONSchemaTypes(type_spec)
                except ValueError:
                    mod_spec, type_spec = type_spec.rsplit('.')
                    if cls.allow_imports:
                        type_mod = importlib.import_module(mod_spec)
                    else:
                        if mod_spec in sys.modules:
                            type_mod = sys.modules[mod_spec]
                        else:
                            bits = mod_spec.split('.')
                            type_mod = sys.modules[bits[0]]
                            for t in bits[1:]:
                                type_mod = getattr(type_mod, t)
                    type_spec = getattr(type_mod, type_spec)
            return type_spec

    @classmethod
    def get_schema_type(cls, t):
        if t == cls.schema_types.Number:
            return (numbers.Number,)
        elif t == cls.schema_types.String:
            return (str,)
        elif t == cls.schema_types.Boolean:
            return (bool,)
        elif t == cls.schema_types.Null:
            return (type(None),)
        elif t == cls.schema_types.Object:
            return (core.is_dict_like,)
        elif t == cls.schema_types.Array:
            return (core.is_list_like,)
        else:
            raise ValueError(f"don't know what to do with {t}")

    @classmethod
    def prep_type_obj(cls, t):
        if isinstance(t, cls.schema_types):
            return cls.get_schema_type(t)
        else:
            return (t,)
    @classmethod
    def get_validators(cls, type_spec):
        if type_spec is None:
            return None, None

        if not core.is_list_like(type_spec):
            type_spec = [type_spec]

        type_spec = [
            cls.resolve_typestr(t)
                if isinstance(t, str) else
            t
            for t in type_spec
        ]

        type_spec = [
            tt
            for t in type_spec
            for tt in cls.prep_type_obj(t)
        ]

        types = tuple(
            t for t in type_spec
            if isinstance(t, type)
        )
        if len(types) == 0: types = None

        validators = tuple(
            t for t in type_spec
            if not isinstance(t, type) and t is not None
        )
        if len(validators) == 0: validators = None


        return types, validators

    def __repr__(self):
        cls = type(self)
        return f"{cls.__name__}(types={self.types}, validators={self.validators})"

    def validate(self, obj, throw=False):
        if self.types is not None and not isinstance(obj, self.types):
            return False
        if self.validators is not None and not all(
                v.validate(obj, throw=False)
                    if hasattr(v, 'validate') else
                v(obj)
                for v in self.validators
        ):
            return False
        return True

    def __call__(self, obj):
        return self.validate(obj)


class ValueValidator:
    __props__ = (
        "enum",
        "const",
        "multipleOf",
        "maximum",
        "minimum",
        "exclusiveMaximum",
        "exclusiveMinimum",
        "maxLength",
        "minLength",
        "pattern",
        "maxItems",
        "minItems",
        "uniqueItems",
        "maxProperties",
        "minProperties",
        "required",
        "validation_function"
    )
    def __init__(self,
                 enum=None,
                 const=None,
                 multipleOf=None,
                 maximum=None,
                 minimum=None,
                 exclusiveMaximum=None,
                 exclusiveMinimum=None,
                 maxLength=None,
                 minLength=None,
                 pattern=None,
                 maxItems=None,
                 minItems=None,
                 uniqueItems=None,
                 maxProperties=None,
                 minProperties=None,
                 required=None,
                 validation_function=None
                 ):
        self.tests = []
        self.enum = enum
        if self.enum is not None: self.tests.append(self._test_enum)
        self.const = const
        if self.const is not None: self.tests.append(self._test_const)
        self.multipleOf = multipleOf
        if self.multipleOf is not None: self.tests.append(self._test_multipleOf)
        self.maximum = maximum
        if self.maximum is not None: self.tests.append(self._test_maximum)
        self.minimum = minimum
        if self.minimum is not None: self.tests.append(self._test_minimum)
        self.exclusiveMaximum = exclusiveMaximum
        if self.exclusiveMaximum is not None: self.tests.append(self._test_exclusiveMaximum)
        self.exclusiveMinimum = exclusiveMinimum
        if self.exclusiveMinimum is not None: self.tests.append(self._test_exclusiveMinimum)
        self.maxLength = maxLength
        if self.maxLength is not None: self.tests.append(self._test_maxLength)
        self.minLength = minLength
        if self.minLength is not None: self.tests.append(self._test_minLength)
        self.pattern = pattern
        if self.pattern is not None: self.tests.append(self._test_pattern)
        self.maxItems = maxItems
        if self.maxItems is not None: self.tests.append(self._test_maxItems)
        self.minItems = minItems
        if self.minItems is not None: self.tests.append(self._test_minItems)
        self.uniqueItems = uniqueItems
        if self.uniqueItems is not None: self.tests.append(self._test_uniqueItems)
        self.validation_function = validation_function
        self.maxProperties = maxProperties
        if self.maxProperties is not None: self.tests.append(self._test_maxProperties)
        self.minProperties = minProperties
        if self.minProperties is not None: self.tests.append(self._test_minProperties)
        self.required = required
        if self.required is not None: self.tests.append(self._test_required)
        if self.validation_function is not None: self.tests.append(self._test_validation_function)

    def _test_validation_function(self, value): return not self.validation_function(value)
    def _test_enum(self, value): return value not in self.enum
    def _test_const(self, value): return value != self.const
    def _test_multipleOf(self, value): return (value % self.multipleOf) != 0
    def _test_maximum(self, value): return value > self.maximum
    def _test_minimum(self, value): return value < self.minimum
    def _test_exclusiveMaximum(self, value): return value >= self.exclusiveMaximum
    def _test_exclusiveMinimum(self, value): return value <= self.exclusiveMinimum
    def _test_maxLength(self, value): return len(value) > self.maxLength
    def _test_minLength(self, value): return len(value) < self.minLength
    def _test_maxItems(self, value): return len(value) > self.maxItems
    def _test_minItems(self, value): return len(value) < self.minItems
    def _test_maxProperties(self, value): return len(value.keys()) > self.maxProperties
    def _test_minProperties(self, value): return len(value.keys()) < self.minProperties
    def _test_required(self, value): return all(k in value for k in self.required)

    def _test_uniqueItems(self, value):
        for n, a in enumerate(value):
            for b in value[n + 1:]:
                if a == b: return True
        return False
    def _test_pattern(self, value):
        return not re.match(self.pattern, value)
    def validate(self, value, throw=False):
        for t in self.tests:
            if t(value): return False
        return True
    def __call__(self, obj):
        return self.validate(obj)


class Schema:
    """
    An object that represents a schema that can be used to test
    if an object matches that schema or not
    """
    type_validator = TypeValidator
    value_validator = ValueValidator

    def __init__(self, schema, optional_schema=None):
        self.schema = self.canonicalize_schema(schema, optional_schema)
        self._required = None

    @property
    def required_keys(self):
        if self._required is None:
            self._required = set(self.schema.get('required', []))
        return self._required
    # @property
    # def dependent_keys(self):
    #     return self.schema.get('dependentRequired', {})

    @classmethod
    def is_json_schema(self, schema):
        return "$schema" in schema

    @classmethod
    def _prep_schema_dict(cls, schema):
        if not core.is_dict_like(schema):
            if core.is_list_like(schema):
                schema = {
                    (
                        k
                        if len(k) != 2 or isinstance(k, str) else
                        k[0]
                    ):
                        (
                            None
                            if len(k) != 2 or isinstance(k, str) else
                            k[1]
                        )
                    for k in schema
                }
            else:
                raise ValueError(f"don't know how to canonicalize schema {schema}")
        return schema


    @classmethod
    def _prep_validators(cls, schema):
        new_schema = schema.copy()
        props = new_schema['properties'].copy()
        for k,v in props.items():
            if (not core.is_dict_like(v)) or 'type' not in v:
                v = {'type':v}
            else:
                v = v.copy()

            t = v['type']
            if core.is_dict_like(t):
                t = cls(t)

            v['type'] = cls.type_validator(t)
            if 'value' not in v:
                value_opts = opts.OptionsSet(v).filter(cls.value_validator)
                if len(value_opts) > 0:
                    v['value'] = cls.value_validator(**value_opts)

            props[k] = v
        new_schema['properties'] = props

        return new_schema


    json_schema_version = "https://json-schema.org/draft/2020-12/schema"
    @classmethod
    def canonicalize_schema(cls, schema, optional_schema=None):
        if schema is None:
            return None

        if not cls.is_json_schema(schema):
            schema = cls._prep_schema_dict(schema)

            old_schema = schema
            schema = {
                "$schema":cls.json_schema_version,
                "properties": {},
                "required": list(schema.keys())
            }
            props = schema["properties"]
            for k,v in old_schema.items():
                if v is None:
                    v = {"type":None}
                elif not core.is_dict_like(v):
                    v = {"type":v}
                props[k] = v

            if optional_schema is not None:
                optional_schema = cls._prep_schema_dict(optional_schema)
                for k,v in optional_schema.items():
                    if v is None:
                        v = {"type":None}
                    elif not core.is_dict_like(v):
                        v = {"type":v}
                    props[k] = v

        return cls._prep_validators(schema)
    @staticmethod
    def _get_prop(o, k):
        return (
            getattr(o, k)
                if not hasattr(o, '__getitem__') else
            o.__getitem__(k)
        )
    def _validate_entry(self, obj, k, v:dict, prop_getter=None, required=True, throw=False):
        match = True
        missing = False
        mistyped = False
        exc = None
        if prop_getter is None:
            prop_getter = self._get_prop
        try:
            t = prop_getter(obj, k)
        except (KeyError, AttributeError) as e:
            missing = True
            match = not required
            t = core.missing
            exc = e
        except (TypeError,) as e:
            mistyped = True
            match = False
            t = None
            exc = e
        else:
            if v is None:
                pass
            elif isinstance(v, (type, tuple)):
                match = isinstance(t, v)
            elif isinstance(v, Schema):
                match = v.validate(t, throw=throw)
            elif core.is_dict_like(v):
                validator = v['type']
                if validator is not None:
                    match = v['type'](t)
                    if match and 'value' in v:
                        match = v['value'](t)
            else:
                match = v(t)

        if not match and throw:
            if missing:
                raise KeyError("object {} doesn't match schema {}; key {} is missing".format(
                    obj, self, k
                )) from exc
            elif mistyped:
                raise KeyError("object {} doesn't match schema {}; doesn't support attributes".format(
                    obj, self
                )) from exc
            else:
                raise KeyError("object {} doesn't match schema {}; value {} doesn't match schema type {}".format(
                    obj, self, t, v
                )) from exc
        return (t, match)

    def validate(self, obj, throw=True):
        """
        Validates that `obj` matches the provided schema
        and throws an error if not

        :param obj:
        :type obj:
        :param throw:
        :type throw:
        :return:
        :rtype:
        """

        for k,v in self.schema['properties'].items():
            if not self._validate_entry(obj, k, v, required=k in self.required_keys, throw=throw)[1]:
                return False

        return True

    def to_dict(self, obj, throw=True, ignore_invalid=False):
        """
        Converts `obj` into a plain `dict` representation

        :param obj:
        :type obj:
        :return:
        :rtype:
        """

        res = {}
        for k, v in self.schema['properties'].items():
            t, m = self._validate_entry(obj, k, v, required=k in self.required_keys, throw=throw)
            if not m and not ignore_invalid:
                return None
            elif t is not core.missing:
                res[k] = t
        # if self.sub_schema is not None:
        #     for k, v in self.sub_schema.items():
        #         t, m = self._validate_entry(obj, k, v, prop_getter, throw=False)
        #         if m:
        #             res[k] = t
        return res

    def __repr__(self):
        return "{}({})".format(type(self).__name__, self.schema)