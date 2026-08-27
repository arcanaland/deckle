from __future__ import annotations

import pytest


@pytest.fixture
def library_less_suite(pytester: pytest.Pytester, monkeypatch) -> pytest.Pytester:
    """The real gate, in a two-test suite, with no scan library reachable."""
    monkeypatch.setenv("DECKLE_SCANS", "")
    pytester.copy_example("conftest.py")
    pytester.makeini("""
        [pytest]
        markers = scans: needs the DECKLE_SCANS library
    """)
    pytester.makepyfile("""
        import pytest

        @pytest.mark.scans
        def test_needs_a_scan():
            pass

        def test_needs_nothing():
            pass
    """)
    return pytester


def test_selecting_a_scan_test_without_a_library_errors(library_less_suite):
    result = library_less_suite.runpytest_subprocess("-m", "scans")
    assert result.ret != pytest.ExitCode.OK
    result.stderr.fnmatch_lines(["*scan test(s) need a library*"])


def test_it_is_an_error_and_not_a_skip(library_less_suite):
    result = library_less_suite.runpytest_subprocess("-m", "scans")
    result.assert_outcomes(skipped=0, passed=0, failed=0)


def test_deselecting_them_runs_clean(library_less_suite):
    result = library_less_suite.runpytest_subprocess("-m", "not scans")
    result.assert_outcomes(passed=1, skipped=0)
    assert result.ret == pytest.ExitCode.OK


def test_a_bare_run_collects_them_and_so_errors(library_less_suite):
    """No -m still counts as selecting them."""
    result = library_less_suite.runpytest_subprocess()
    assert result.ret != pytest.ExitCode.OK
