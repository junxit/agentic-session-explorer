"""Oh My Pi's ``.env`` files, read as the Bun runtime omp runs on reads them.

omp fills unset variables from its ``.env`` files with ``util.parseEnv`` — Bun's,
not Node's: ``parse_env`` in Bun's ``node_util_binding.rs`` runs the parser in
``src/dotenv/env_loader.rs``. ``sx`` reads every file with a port of that
parser, and with its port of Node's (:mod:`tests.test_omp_dotenv`); a variable
``sx`` follows that the two read differently makes every lease unknown
(:mod:`tests.test_omp_bun_leases`). The expected values below were read off
Bun's source and the assertions in its own ``test/js/bun/util/env.test.ts``;
Bun is never run, nor is ``omp``.
"""

from __future__ import annotations

import pytest

from sx.adapters.omp import _bun_parse_env as bun
from sx.adapters.omp import _parse_dotenv


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # From Bun's own tests of util.parseEnv and of .env files.
        ("A=x\u0920\nB=\u00a0x\u00a0\nC=\u00a0\nD=  x  \n",
         {"A": "x\u0920", "B": "\u00a0x\u00a0", "C": "\u00a0", "D": "x"}),
        ("FOO=\rBAR='bar\r\rbaz'\r\nMOO=moo\r", {"FOO": "", "BAR": "bar\n\nbaz", "MOO": "moo"}),
        ('KEY="a\\nb"', {"KEY": "a\nb"}),
        ("VARNAME=A B", {"VARNAME": "A B"}),
        ('A="a$t"\nC=`c\\$v`', {"A": "a$t", "C": "c\\$v"}),
        ("A='hello' junk\nB='x'\nC=3\n", {"A": "hello", "B": "x", "C": "3"}),
        ('A="hello\nworld" junk\nB=2\n', {"A": "hello\nworld", "B": "2"}),
        ("\ufeffA=1\nB=2\n", {"A": "1", "B": "2"}),
        ("\ufeffexport FOO=bar\n", {"FOO": "bar"}),
        ("\ufeff", {}),
        # Where Bun and Node part.
        ("KEY: value\n", {"KEY": "value"}),
        ("KEY:value\n", {}),
        ("KEY:\nvalue\n", {"KEY": "value"}),
        ("KEY\n=value\n", {"KEY": "value"}),
        ("A=1\rB=2\r", {"A": "1", "B": "2"}),
        ("export\tKEY=v\n", {"KEY": "v"}),
        ("export A=\nB=2", {"A": "", "B": "2"}),
        ("=KEY=v\n", {}),
        ("KEY=\nvalue\n", {"KEY": ""}),
        ('KEY=\n"quoted"\nB=2\n', {"KEY": "quoted", "B": "2"}),
        ("A=\fx\f\n", {"A": "x"}),
        ('A="x\\"y"\n', {"A": 'x\\"y'}),
        ('A="x\\ry"\n', {"A": "x\ry"}),
        ('A="x\ry"\n', {"A": "x\ny"}),
        ('A="x\\\ry"\n', {"A": "x\\\ry"}),
        ("A='a\\'b'\n", {"A": "a\\'b"}),
        ("A='abc\\'\n", {"A": "'abc\\'"}),
        ('A="abc # x\n', {"A": '"abc'}),
        ('A="unclosed\\', {"A": '"unclosed\\'}),
        # Where they agree.
        ("export =1\nexport=2\n", {"export": "2"}),
        ("A=x#y\n  # c\nB=1 # c\n", {"A": "x", "B": "1"}),
        ("A=1\nA=2\n", {"A": "2"}),
        ("A", {}),
        ("A=", {"A": ""}),
        ("A=1\x00x\nB=2\n", {"A": "1\x00x", "B": "2"}),
    ],
)
def test_dot_env_text_is_parsed_as_bun_parses_it(text, expected):
    """Every rule of Bun's ``Parser`` is kept, quirks included."""
    assert bun(text) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("A=x\n" * 150_000, {"A": "x"}),
        ("A='x\n" + "B=y\n" * 150_000, {"A": "'x", "B": "y"}),
        ('A="x\n' * 150_000, {"A": "x\nA="}),
        ('A="' + "\\x" * 300_000, {"A": '"' + "\\x" * 300_000}),
    ],
    ids=["plain", "one-unclosed-quote", "quotes-closing-across-lines", "escapes-never-closing"],
)
def test_a_large_file_is_parsed_whole_and_quickly_as_bun_would(text, expected):
    """Every step searches on from where the last stopped: the cost grows with the file."""
    assert bun(text) == expected


@pytest.mark.parametrize(
    ("text", "values", "unsure"),
    [
        ("XDG_STATE_HOME=/s\n", {"XDG_STATE_HOME": "/s"}, ()),
        ("XDG_STATE_HOME: /s\n", {"XDG_STATE_HOME": "/s"}, ("XDG_STATE_HOME",)),
        ("=XDG_STATE_HOME=/s\n", {}, ("XDG_STATE_HOME",)),
        ("XDG_STATE_HOME=/s\f\n", {"XDG_STATE_HOME": "/s"}, ("XDG_STATE_HOME",)),
        (
            "OMP_CONFIG_DIR: alt\n",
            {"OMP_CONFIG_DIR": "alt", "PI_CONFIG_DIR": "alt"},
            ("OMP_CONFIG_DIR",),
        ),
        (
            'PI_TEST_SESSION_OWNERS_DIR="/o\\"x"\n',
            {"PI_TEST_SESSION_OWNERS_DIR": '/o\\"x'},
            ("PI_TEST_SESSION_OWNERS_DIR",),
        ),
        ("OTHER: x\nPI_PROFILE=work\n", {"OTHER": "x", "PI_PROFILE": "work"}, ()),
    ],
    ids=["agree", "colon", "empty-key", "form-feed", "omp-spelling", "escaped-quote", "unfollowed"],
)
def test_a_followed_variable_the_two_parsers_read_differently_is_unsure(text, values, unsure):
    """Bun's reading is the one kept; Node's only says which variables to doubt."""
    dotenv = _parse_dotenv(text)
    assert dotenv.values == values
    assert dotenv.unsure == unsure
