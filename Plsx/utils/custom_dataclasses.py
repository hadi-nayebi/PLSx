"""base classes for custom dataclasses"""
"""Collection of data structures and validations"""

from dataclasses import dataclass, fields, is_dataclass
from typing import Any

from dataclasses_json import Undefined, dataclass_json


@dataclass
class DefaultVal:  # -> https://stackoverflow.com/users/2128545/mikeschneeberger
    val: Any


@dataclass
class NoneRefersDefault:  # -> https://stackoverflow.com/users/2128545/mikeschneeberger
    def __post_init__(self):
        for field in fields(self):
            # if a field of this data class defines a default value of type
            # `DefaultVal`, then use its value in case the field after
            # initialization has either not changed or is None.
            if isinstance(field.default, DefaultVal):
                field_val = getattr(self, field.name)
                if isinstance(field_val, DefaultVal) or field_val is None:
                    setattr(self, field.name, field.default.val)


# decorator to wrap original __init__ -> https://www.geeksforgeeks.org/creating-nested-dataclass-objects-in-python/
def nested_deco(*args, **kwargs):
    """decorator for assigning nested dict to nested dataclass"""

    def wrapper(check_class):
        # passing class to investigate
        check_class = dataclass(check_class, **kwargs)
        o_init = check_class.__init__

        def __init__(self, *args, **kwargs):
            # getting class fields to filter extra keys
            class_fields = {f.name for f in fields(check_class)}
            for key in list(kwargs.keys()):
                if key not in class_fields:
                    del kwargs[key]
            for name, value in kwargs.items():
                # getting field type
                ft = check_class.__annotations__.get(name, None)
                if is_dataclass(ft) and isinstance(value, dict):
                    obj = ft(**value)
                    kwargs[name] = obj
                o_init(self, *args, **kwargs)

        check_class.__init__ = __init__
        return check_class

    return wrapper(args[0]) if args else wrapper
