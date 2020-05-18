#!/usr/bin/env python3
#
# Copyright (c) Bo Peng and the University of Texas MD Anderson Cancer Center
# Distributed under the terms of the 3-clause BSD License.

import pickle
from sos.utils import short_repr, env
from sos.eval import interpolate

#
# These functions will be imported by both Python2 and Python3 and cannot
# use any python3-specific syntax (e.g. f-string)
#
__init_statement__ = r'''
from collections import Sized, KeysView, Sequence
from types import ModuleType
import pydoc
import pickle
import json

def __version_info__(module):
    # return the version of Python module
    try:
        code = ("import %s; version=str(%s.__version__)" %
                (module, module))
        ns_g = ns_l = {}
        exec(compile(code, "<string>", "exec"), ns_g, ns_l)
        return ns_l["version"]
    except Exception as e:
        import pkg_resources
        try:
            return pkg_resources.require(module)[0].version
        except Exception as e:
            return 'na'


def __loaded_modules__():
    from types import ModuleType
    res = []
    for key,value in globals().items():
        if isinstance(value, ModuleType):
            res.append([value.__name__, __version_info__(value.__name__)])
    return [(x,y) for x,y in res if y != 'na']


def __short_repr(obj):
    if obj is None:
        return 'None'
    elif isinstance(obj, str) and len(obj) > 80:
        return '{}...{}'.format(obj[:60].replace('\n', '\\n'),
                                obj[-20:].replace('\n', '\\n'))
    elif isinstance(obj, (str, int, float, bool)):
        return repr(obj)
    elif hasattr(obj, '__short_repr__'):
        return obj.__short_repr__()
    elif isinstance(obj, Sequence):  # should be a list or tuple
        if len(obj) == 0:
            return '[]'
        elif len(obj) == 1:
            return __short_repr(obj[0])
        elif len(obj) == 2:
            return __short_repr(obj[0]) + ', ' + __short_repr(obj[1])
        else:
            return __short_repr(obj[0]) + ', ' + __short_repr(obj[1]) + ', ... (' + str(len(obj)) + ' items)'
    elif isinstance(obj, dict):
        if not obj:
            return ''
        elif len(obj) == 1:
            first_key = list(obj.keys())[0]
            return __short_repr(repr(first_key)) + ':' + __short_repr(obj[first_key])
        else:
            first_key = list(obj.keys())[0]
            return __short_repr(first_key) + ':' + __short_repr(obj[first_key]) + ', ... (' + str(len(obj)) + ' items)'
    elif isinstance(obj, KeysView):
        if not obj:
            return ''
        elif len(obj) == 1:
            return __short_repr(next(iter(obj)))
        else:
            return __short_repr(next(iter(obj))) + ', ... (' + str(len(obj)) + ' items)'
    else:
        ret = str(obj)
        if len(ret) > 40:
            return repr(obj)[:35] + '...'
        else:
            return ret


def __preview_var(item):
    # check if 'item' is in the subkernel
    if item not in globals():
        return '', 'Unknown variable {}'.format(item)

    obj = eval(item)

    # get the basic information of object
    txt = type(obj).__name__
    # we could potentially check the shape of data frame and matrix
    # but then we will need to import the numpy and pandas libraries
    if hasattr(obj, 'shape') and getattr(obj, 'shape') is not None:
        txt += ' of shape ' + str(getattr(obj, "shape"))
    elif isinstance(obj, Sized):
        txt += ' of length ' + str(obj.__len__())
    if isinstance(obj, ModuleType):
        return txt, ({
            'text/plain': pydoc.render_doc(obj, renderer=pydoc.plaintext)
        }, {})
    elif callable(obj):
        return txt, ({
            'text/plain': pydoc.render_doc(obj, renderer=pydoc.plaintext)
        }, {})
    elif hasattr(obj, 'to_html') and getattr(obj, 'to_html') is not None:
        try:
            html = obj.to_html()
            return txt, {'text/html': html}
        except Exception as e:
            return txt, __short_repr(obj)
    else:
        return txt, __short_repr(obj)

def __repr_var(item):
    if item not in globals():
        raise ValueError('Undefined variable {}'.format(item))
    return repr(eval(item))

'''


class sos_Python:
    supported_kernels = {'Python3': ['python3'], 'Python2': ['python2']}
    background_color = {'Python2': '#FFF177', 'Python3': '#FFD91A'}
    options = {
        'variable_pattern': r'^\s*[_A-Za-z0-9\.]+\s*$',
        'assignment_pattern': r'^\s*([_A-Za-z0-9\.]+)\s*=.*$',
        'indentation_aware': True
    }
    cd_command = 'import os;os.chdir({dir!r})'

    def __init__(self, sos_kernel, kernel_name='python3'):
        self.sos_kernel = sos_kernel
        self.kernel_name = kernel_name
        self.init_statements = __init_statement__

    def get_vars(self, names):
        for name in names:
            if self.kernel_name == 'python3':
                stmt = "globals().update(pickle.loads({!r}))\n".format(
                    pickle.dumps({name: env.sos_dict[name]}))
            else:
                stmt = "globals().update(pickle.loads({!r}))\n".format(
                    pickle.dumps({name: env.sos_dict[name]},
                                 protocol=2,
                                 fix_imports=True))
            self.sos_kernel.run_cell(
                stmt,
                True,
                False,
                on_error='Failed to get variable {} from SoS to {}'.format(
                    name, self.kernel_name))

    def load_pickled(self, item):
        if isinstance(item, bytes):
            return pickle.loads(item)
        elif isinstance(item, str):
            return pickle.loads(item.encode('utf-8'))
        else:
            self.sos_kernel.warn(
                'Cannot restore from result of pickle.dumps: {}'.format(
                    short_repr(item)))
            return {}

    def put_vars(self, items, to_kernel=None, as_type=None):
        stmt = '__vars__={{ {} }}\n__vars__.update({{x:y for x,y in locals().items() if x.startswith("sos")}})\npickle.dumps(__vars__)'.format(
            ','.join('"{0}":{0}'.format(x) for x in items))
        try:
            # sometimes python2 kernel would fail to send a execute_result and lead to an error
            response = self.sos_kernel.get_response(stmt,
                                                    ['execute_result'])[-1][1]
        except:
            return {}

        # Python3 -> Python3
        if (self.kernel_name == 'python3' and to_kernel == 'Python3') or \
                (self.kernel_name == 'python2' and to_kernel == 'Python2'):
            # to self, this should allow all variables to be passed
            return 'import pickle;globals().update(pickle.loads({}))'.format(
                response['data']['text/plain'])
        try:
            ret = self.load_pickled(eval(response['data']['text/plain']))
            if self.sos_kernel._debug_mode:
                self.sos_kernel.warn('Get: {}'.format(ret))
            return ret
        except Exception as e:
            self.sos_kernel.warn('Failed to import variables {}: {}'.format(
                items, e))
            return {}

    def expand(self, text, sigil):
        if sigil != '{ }':
            from sos.parser import replace_sigil
            text = replace_sigil(text, sigil)

        try:
            from sos.utils import as_fstring
            response = self.sos_kernel.get_response(
                as_fstring(text), ['execute_result'])[-1][1]
            return eval(response['data']['text/plain'])
        except Exception as e:
            err_msg = self.sos_kernel.get_response(
                as_fstring(text), ('error',), name=('evalue',))[0][1]['evalue']
            self.sos_kernel.warn(f'Failed to expand "{text}": {err_msg}')
            return text

    def preview(self, item):
        try:
            response = self.sos_kernel.get_response(
                f'pickle.dumps(__preview_var("{item}"))',
                ['execute_result'])[-1][1]
            return self.load_pickled(eval(response['data']['text/plain']))
        except Exception as e:
            env.log_to_file('PREVIEW', f'Failed to preview {item}: {e}')
            return '', f'No preview is available {e}'

    def sessioninfo(self):
        modules = self.sos_kernel.get_response(
            'import pickle;import sys;res=[("Version", sys.version)];res.extend(__loaded_modules__());pickle.dumps(res)',
            ['execute_result'])[0][1]
        return self.load_pickled(eval(modules['data']['text/plain']))
