# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""One loud failure per missing implementation.

Every other module in this suite skips when the script it exercises does not
exist, which keeps the run readable while the harness is being written.  This
module is what makes those skips safe: if a script is missing, exactly one test
fails and names it, so a green run can never mean "nothing was tested".
"""

import pytest

import harness_support as support


@pytest.mark.parametrize("name", support.IMPLEMENTATIONS)
def test_implementation_exists(name):
    path = support.script_path(name)
    assert path.is_file(), (
        f"{path} does not exist. Every test module that exercises it skips, so "
        f"this is the only signal that the coverage is absent rather than passing."
    )


@pytest.mark.parametrize("name", support.IMPLEMENTATIONS)
def test_implementation_carries_reuse_metadata(name):
    path = support.script_path(name)
    if not path.is_file():
        pytest.skip("covered by test_implementation_exists")
    head = path.read_text().splitlines()[:6]
    assert any("SPDX-FileCopyrightText: The Eigen Authors" in line for line in head), path
    assert any("SPDX-License-Identifier: MPL-2.0" in line for line in head), path


@pytest.mark.parametrize("name", support.IMPLEMENTATIONS)
def test_implementation_help_exits_zero_and_writes_stdout(name):
    proc = support.run_cli(name, ["--help"])
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip(), f"{name} --help wrote nothing to stdout"


@pytest.mark.parametrize("name", support.IMPLEMENTATIONS)
def test_a_usage_error_exits_one_not_argparse_two(name):
    """CONTRACTS.md section 4 assigns exit 1 to a usage error; argparse's own
    default is 2, which every tool here uses for a configuration or input error."""
    proc = support.run_cli(name, ["--no-such-option"])
    assert proc.returncode == 1, (
        f"{name} exited {proc.returncode} on an unrecognised option; 2 collides with the "
        f"configuration/input-error code\n{proc.stderr[-500:]}"
    )


def test_contract_documents_are_present():
    for name in ("CONTRACTS.md", "ops.toml", "result_schema.json"):
        assert (support.COMPARISON_DIR / name).is_file(), name


def test_fixture_tree_is_complete():
    expected = [
        "raw/gbench_gemm.json",
        "raw/benchmark_list_tests.txt",
        "machines/testmachine.toml",
        "stub_benchmark.py",
        "merged/gemm_merged.json",
        "merged/one_cell.json",
        "results/gemm_eigen_accelerate.json",
    ]
    for relative in expected:
        assert (support.FIXTURES / relative).is_file(), f"missing fixture {relative}; run tests/regenerate.py"
