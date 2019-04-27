#!/usr/bin/env python3
#
# Copyright (c) Bo Peng and the University of Texas MD Anderson Cancer Center
# Distributed under the terms of the 3-clause BSD License.

from sos_notebook.test_utils import NotebookTest
import random


class TestDataExchange(NotebookTest):

    #
    # Python 2
    #
    def _var_name(self):
        if not hasattr(self, '_var_idx'):
            self._var_idx = 0
        self._var_idx += 1
        return f'var{self._var_idx}'

    def py2_get_from_SoS(self, notebook, sos_expr):
        var_name = self._var_name()
        notebook.call(f'{var_name} = {sos_expr}', kernel='SoS')
        return notebook.check_output(
            f'''\
            %get {var_name}
            print(repr({var_name}))
            ''',
            kernel='Python2')

    def py2_put_to_SoS(self, notebook, py2_expr):
        var_name = self._var_name()
        notebook.call(
            f'''\
            %put {var_name}
            {var_name} = {py2_expr}
            ''',
            kernel='Python2')
        return notebook.check_output(f'print(repr({var_name}))', kernel='SoS')

    def test_py2_get_none(self, notebook):
        assert 'None' == self.py2_get_from_SoS(notebook, 'None')

    def test_py2_put_none(self, notebook):
        assert 'None' == self.py2_put_to_SoS(notebook, 'None')

    def test_py2_get_int(self, notebook):
        assert 123 == int(self.py2_get_from_SoS(notebook, '123'))
        assert 1234567891234 == int(
            self.py2_get_from_SoS(notebook, '1234567891234').rstrip('L'))
        assert 123456789123456789 == int(
            self.py2_get_from_SoS(notebook, '123456789123456789').rstrip('L'))

    def test_py2_put_int(self, notebook):
        assert 123 == int(self.py2_put_to_SoS(notebook, '123'))
        assert 1234567891234 == int(
            self.py2_put_to_SoS(notebook, '1234567891234'))
        assert 123456789123456789 == int(
            self.py2_put_to_SoS(notebook, '123456789123456789'))

    def test_py2_get_double(self, notebook):
        val = str(random.random())
        assert abs(float(val) -
                   float(self.py2_get_from_SoS(notebook, val))) < 1e-10

    def test_py2_put_double(self, notebook):
        val = str(random.random())
        assert abs(float(val) -
                   float(self.py2_put_to_SoS(notebook, val))) < 1e-10

    def test_py2_get_logic(self, notebook):
        assert 'True' == self.py2_get_from_SoS(notebook, 'True')
        assert 'False' == self.py2_get_from_SoS(notebook, 'False')

    def test_py2_put_logic(self, notebook):
        assert 'True' == self.py2_put_to_SoS(notebook, 'True')
        assert 'False' == self.py2_put_to_SoS(notebook, 'False')

    def test_py2_get_num_array(self, notebook):
        assert '[1]' == self.py2_get_from_SoS(notebook, '[1]')
        assert '[1, 2]' == self.py2_get_from_SoS(notebook, '[1, 2]')
        #
        assert '[1.23]' == self.py2_get_from_SoS(notebook, '[1.23]')
        assert '[1.4, 2]' == self.py2_get_from_SoS(notebook, '[1.4, 2]')

    def test_py2_put_num_array(self, notebook):
        # Note that single element numeric array is treated as single value
        assert '[1]' == self.py2_put_to_SoS(notebook, '[1]')
        assert '[1, 2]' == self.py2_put_to_SoS(notebook, '[1, 2]')
        #
        assert '[1.23]' == self.py2_put_to_SoS(notebook, '[1.23]')
        assert '[1.4, 2]' == self.py2_put_to_SoS(notebook, '[1.4, 2]')

    def test_py2_get_logic_array(self, notebook):
        assert '[True]' == self.py2_get_from_SoS(notebook, '[True]')
        assert '[True, False, True]' == self.py2_get_from_SoS(
            notebook, '[True, False, True]')

    def test_py2_put_logic_array(self, notebook):
        # Note that single element numeric array is treated as single value
        assert '[True]' == self.py2_put_to_SoS(notebook, '[True]')
        assert '[True, False, True]' == self.py2_put_to_SoS(
            notebook, '[True, False, True]')

    def test_py2_get_str(self, notebook):
        assert "u'ab c d'" == self.py2_get_from_SoS(notebook, "'ab c d'")
        assert "u'ab\\td'" == self.py2_get_from_SoS(notebook, r"'ab\td'")

    def test_py2_put_str(self, notebook):
        assert "'ab c d'" == self.py2_put_to_SoS(notebook, "'ab c d'")
        assert "'ab\\td'" == self.py2_put_to_SoS(notebook, r"'ab\td'")

    def test_py2_get_mixed_list(self, notebook):
        assert "[1.4, True, u'asd']" == self.py2_get_from_SoS(
            notebook, '[1.4, True, "asd"]')

    def test_py2_put_mixed_list(self, notebook):
        # R does not have mixed list, it just convert everything to string.
        assert "[1.4, True, 'asd']" == self.py2_put_to_SoS(
            notebook, '[1.4, True, "asd"]')

    def test_py2_get_dict(self, notebook):
        # Python does not have named ordered list, so get dictionary
        output = self.py2_get_from_SoS(notebook, "dict(a=1, b='2')")
        assert "{u'a': 1, u'b': u'2'}" == output or "{u'b': u'2', u'a': 1}" == output

    def test_py2_put_dict(self, notebook):
        output = self.py2_put_to_SoS(notebook, "dict(a=1, b='2')")
        assert "{'a': 1, 'b': '2'}" == output or "{'b': '2', 'a': 1}" == output

    def test_py2_get_set(self, notebook):
        output = self.py2_get_from_SoS(notebook, "{1.5, 'abc'}")
        assert "set([1.5, u'abc'])" == output or "set([u'abc', 1.5])" == output

    def test_py2_get_complex(self, notebook):
        assert "(1+2.2j)" == self.py2_get_from_SoS(notebook, "complex(1, 2.2)")

    def test_py2_put_complex(self, notebook):
        assert "(1+2.2j)" == self.py2_put_to_SoS(notebook, "complex(1, 2.2)")

    def test_py2_get_recursive(self, notebook):
        output = self.py2_get_from_SoS(
            notebook, "{'a': 1, 'b': {'c': 3, 'd': 'whatever'}}")
        assert "u'a': 1" in output and "u'b':" in output and "u'c': 3" in output and "u'd': u'whatever'" in output

    def test_py2_put_recursive(self, notebook):
        output = self.py2_put_to_SoS(
            notebook, "{'a': 1, 'b': {'c': 3, 'd': 'whatever'}}")
        assert "'a': 1" in output and "'b':" in output and "'c': 3" in output and "'d': 'whatever'" in output

    #
    # Python 3
    #
    def py3_get_from_SoS(self, notebook, sos_expr):
        var_name = self._var_name()
        notebook.call(f'{var_name} = {sos_expr}', kernel='SoS')
        return notebook.check_output(
            f'''\
            %get {var_name}
            print(repr({var_name}))
            ''',
            kernel='Python3')

    def py3_put_to_SoS(self, notebook, py3_expr):
        var_name = self._var_name()
        notebook.call(
            f'''\
            %put {var_name}
            {var_name} = {py3_expr}
            ''',
            kernel='Python3')
        return notebook.check_output(f'print(repr({var_name}))', kernel='SoS')

    def test_py3_get_none(self, notebook):
        assert 'None' == self.py3_get_from_SoS(notebook, 'None')

    def test_py3_put_none(self, notebook):
        assert 'None' == self.py3_put_to_SoS(notebook, 'None')

    def test_py3_get_int(self, notebook):
        assert 123 == int(self.py3_get_from_SoS(notebook, '123'))
        assert 1234567891234 == int(
            self.py3_get_from_SoS(notebook, '1234567891234'))
        assert 123456789123456789 == int(
            self.py3_get_from_SoS(notebook, '123456789123456789'))

    def test_py3_put_int(self, notebook):
        assert 123 == int(self.py3_put_to_SoS(notebook, '123'))
        assert 1234567891234 == int(
            self.py3_put_to_SoS(notebook, '1234567891234'))
        assert 123456789123456789 == int(
            self.py3_put_to_SoS(notebook, '123456789123456789'))

    def test_py3_get_double(self, notebook):
        val = str(random.random())
        assert abs(float(val) -
                   float(self.py3_get_from_SoS(notebook, val))) < 1e-10

    def test_py3_put_double(self, notebook):
        val = str(random.random())
        assert abs(float(val) -
                   float(self.py3_put_to_SoS(notebook, val))) < 1e-10

    def test_py3_get_logic(self, notebook):
        assert 'True' == self.py3_get_from_SoS(notebook, 'True')
        assert 'False' == self.py3_get_from_SoS(notebook, 'False')

    def test_py3_put_logic(self, notebook):
        assert 'True' == self.py3_put_to_SoS(notebook, 'True')
        assert 'False' == self.py3_put_to_SoS(notebook, 'False')

    def test_py3_get_num_array(self, notebook):
        assert '[1]' == self.py3_get_from_SoS(notebook, '[1]')
        assert '[1, 2]' == self.py3_get_from_SoS(notebook, '[1, 2]')
        #
        assert '[1.23]' == self.py3_get_from_SoS(notebook, '[1.23]')
        assert '[1.4, 2]' == self.py3_get_from_SoS(notebook, '[1.4, 2]')

    def test_py3_put_num_array(self, notebook):
        # Note that single element numeric array is treated as single value
        assert '[1]' == self.py3_put_to_SoS(notebook, '[1]')
        assert '[1, 2]' == self.py3_put_to_SoS(notebook, '[1, 2]')
        #
        assert '[1.23]' == self.py3_put_to_SoS(notebook, '[1.23]')
        assert '[1.4, 2]' == self.py3_put_to_SoS(notebook, '[1.4, 2]')

    def test_py3_get_logic_array(self, notebook):
        assert '[True]' == self.py3_get_from_SoS(notebook, '[True]')
        assert '[True, False, True]' == self.py3_get_from_SoS(
            notebook, '[True, False, True]')

    def test_py3_put_logic_array(self, notebook):
        # Note that single element numeric array is treated as single value
        assert '[True]' == self.py3_put_to_SoS(notebook, '[True]')
        assert '[True, False, True]' == self.py3_put_to_SoS(
            notebook, '[True, False, True]')

    def test_py3_get_str(self, notebook):
        assert "'ab c d'" == self.py3_get_from_SoS(notebook, "'ab c d'")
        assert "'ab\\td'" == self.py3_get_from_SoS(notebook, r"'ab\td'")

    def test_py3_put_str(self, notebook):
        assert "'ab c d'" == self.py3_put_to_SoS(notebook, "'ab c d'")
        assert "'ab\\td'" == self.py3_put_to_SoS(notebook, r"'ab\td'")

    def test_py3_get_mixed_list(self, notebook):
        assert "[1.4, True, 'asd']" == self.py3_get_from_SoS(
            notebook, '[1.4, True, "asd"]')

    def test_py3_put_mixed_list(self, notebook):
        # R does not have mixed list, it just convert everything to string.
        assert "[1.4, True, 'asd']" == self.py3_put_to_SoS(
            notebook, '[1.4, True, "asd"]')

    def test_py3_get_dict(self, notebook):
        # Python does not have named ordered list, so get dictionary
        output = self.py3_get_from_SoS(notebook, "dict(a=1, b='2')")
        assert "{'a': 1, 'b': '2'}" == output or "{'b': '2', 'a': 1}" == output

    def test_py3_put_dict(self, notebook):
        output = self.py3_put_to_SoS(notebook, "dict(a=1, b='2')")
        assert "{'a': 1, 'b': '2'}" == output or "{'b': '2', 'a': 1}" == output

    def test_py3_get_set(self, notebook):
        output = self.py3_get_from_SoS(notebook, "{1.5, 'abc'}")
        assert "{1.5, 'abc'}" == output or "{'abc', 1.5}" == output

    def test_py3_get_complex(self, notebook):
        assert "(1+2.2j)" == self.py3_get_from_SoS(notebook, "complex(1, 2.2)")

    def test_py3_put_complex(self, notebook):
        assert "(1+2.2j)" == self.py3_put_to_SoS(notebook, "complex(1, 2.2)")

    def test_py3_get_recursive(self, notebook):
        output = self.py3_get_from_SoS(
            notebook, "{'a': 1, 'b': {'c': 3, 'd': 'whatever'}}")
        assert "'a': 1" in output and "'b':" in output and "'c': 3" in output and "'d': 'whatever'" in output

    def test_py3_put_recursive(self, notebook):
        output = self.py3_put_to_SoS(
            notebook, "{'a': 1, 'b': {'c': 3, 'd': 'whatever'}}")
        assert "'a': 1" in output and "'b':" in output and "'c': 3" in output and "'d': 'whatever'" in output