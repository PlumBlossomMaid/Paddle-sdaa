import importlib.abc
import importlib.util
import os
import sys
import types

import paddle
from paddle.utils.cpp_extension.extension_utils import _custom_api_content

cur_dir = os.path.dirname(os.path.abspath(__file__))
so_path = os.path.join(cur_dir, "sdaa_ext_pd_.so")


def __bootstrap__():
    custom_ops = paddle.utils.cpp_extension.load_op_meta_info_and_register_op(so_path)
    module_globals = globals()

    for op_name in custom_ops:
        exec(_custom_api_content(op_name), module_globals)

    if os.name == 'nt' or sys.platform.startswith('darwin'):
        mod = types.ModuleType(__name__)
    else:
        try:
            spec = importlib.util.spec_from_file_location(__name__, so_path)
            assert spec is not None
            mod = importlib.util.module_from_spec(spec)
            assert isinstance(spec.loader, importlib.abc.Loader)
            spec.loader.exec_module(mod)
        except ImportError:
            mod = types.ModuleType(__name__)

    generated_custom_ops = set(custom_ops)
    for attr_name, attr_value in list(mod.__dict__.items()):
        if attr_name.startswith('__') and attr_name.endswith('__'):
            continue
        if attr_name in generated_custom_ops:
            continue
        module_globals[attr_name] = attr_value


__bootstrap__()
