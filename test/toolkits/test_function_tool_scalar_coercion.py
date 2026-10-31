# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ========= Copyright 2023-2026 @ CAMEL-AI.org. All Rights Reserved. =========
"""Tests for scalar (int/float/bool) string coercion in
:func:`camel.toolkits.function_tool._coerce_arg`.

LLM providers frequently deliver numeric and boolean tool-call arguments
as JSON strings even when the declared schema says ``integer``/``boolean``.
The wrapped function previously received ``str`` values for scalar
annotations, causing ``TypeError`` on arithmetic.
"""

from camel.toolkits import FunctionTool


def compute_total(quantity: int, price: float, tax_included: bool) -> str:
    total = quantity * price
    return f"total={total}, tax_included={tax_included}"


def test_string_numeric_args_are_coerced():
    tool = FunctionTool(compute_total)
    # Previously raised: can't multiply sequence by non-int of type 'str'
    result = tool(**{"quantity": "42", "price": "9.5", "tax_included": "true"})
    assert "total=399.0" in result
    assert "tax_included=True" in result


def test_correct_types_unchanged():
    tool = FunctionTool(compute_total)
    result = tool(**{"quantity": 42, "price": 9.5, "tax_included": True})
    assert "total=399.0" in result
    assert "tax_included=True" in result


def test_int_string_coercion():
    def grab(n: int) -> str:
        return f"{type(n).__name__}:{n!r}"

    tool = FunctionTool(grab)
    assert tool(**{"n": "42"}) == "int:42"
    # Whitespace is tolerated
    assert tool(**{"n": " 42 "}) == "int:42"
    # Unparseable strings pass through unchanged
    assert tool(**{"n": "42abc"}) == "str:'42abc'"
    # Non-integral string is not silently truncated
    assert tool(**{"n": "9.5"}) == "str:'9.5'"


def test_float_string_coercion():
    def grab(x: float) -> str:
        return f"{type(x).__name__}:{x!r}"

    tool = FunctionTool(grab)
    assert tool(**{"x": "9.5"}) == "float:9.5"
    # Integral-looking strings coerce to float
    assert tool(**{"x": "42"}) == "float:42.0"


def test_bool_string_coercion():
    def grab(on: bool) -> str:
        return f"{on!r}"

    tool = FunctionTool(grab)
    assert tool(**{"on": "true"}) == "True"
    assert tool(**{"on": "True"}) == "True"
    assert tool(**{"on": "yes"}) == "True"
    assert tool(**{"on": "1"}) == "True"
    assert tool(**{"on": "false"}) == "False"
    assert tool(**{"on": "FALSE"}) == "False"
    assert tool(**{"on": "no"}) == "False"
    assert tool(**{"on": "0"}) == "False"
    # Unrecognized strings pass through unchanged
    assert tool(**{"on": "maybe"}) == "'maybe'"


def test_non_string_values_unchanged():
    def grab(n: int, x: float, on: bool) -> str:
        return f"{type(n).__name__},{type(x).__name__},{type(on).__name__}"

    tool = FunctionTool(grab)
    result = tool(**{"n": 7, "x": 1.5, "on": False})
    assert result == "int,float,bool"
