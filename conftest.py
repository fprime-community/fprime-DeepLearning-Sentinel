"""Keep the F' toolchain out of pytest's collection.

The framework checkout under `fprime/lib/` vendors googletest, which ships its
own pytest-shaped Python files (`gtest_xml_outfiles_test.py` and eleven more).
Collecting them fails on imports that belong to a different project, so the suite
would be red for a reason that has nothing to do with this repository.

Scoped to the two gitignored, script-rebuilt subtrees, for the same reason
`tests/test_no_local_persistence.py` exempts them: `fprime/`'s own sources are
still collected, so a test written beside the component runs like any other.
"""

collect_ignore_glob = ["fprime/lib/*", "fprime/fprime-venv/*"]
