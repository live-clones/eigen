# SPDX-FileCopyrightText: The Eigen Authors
# SPDX-License-Identifier: MPL-2.0

"""Session fixtures for the comparison-harness tests.

Run the suite with a bare `pytest` from `benchmarks/comparison/`.  It needs no
network, no built benchmark binary, and no configured CMake build directory.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import harness_support as support  # noqa: E402


def pytest_addoption(parser):
    parser.addoption(
        "--regen-golden",
        action="store_true",
        default=False,
        help="Rewrite the committed golden files from the current renderers "
        "instead of comparing against them. Equivalent to "
        "`python3 tests/regenerate.py --goldens`.",
    )


@pytest.fixture(scope="session")
def ops():
    return support.load_ops()


@pytest.fixture(scope="session")
def schema():
    return support.load_schema()


@pytest.fixture(scope="session")
def validator(schema):
    return support.schema_validator(schema)


@pytest.fixture(scope="session")
def fixtures_dir():
    return support.FIXTURES


@pytest.fixture(scope="session")
def raw_gbench():
    return json.loads((support.FIXTURES / "raw" / "gbench_gemm.json").read_text())


@pytest.fixture(scope="session")
def listed_benchmarks():
    text = (support.FIXTURES / "raw" / "benchmark_list_tests.txt").read_text()
    return [line.strip() for line in text.splitlines() if line.strip()]


@pytest.fixture(scope="session")
def merged_gemm():
    return support.read_json(support.FIXTURES / "merged" / "gemm_merged.json")


@pytest.fixture(scope="session")
def merged_one_cell():
    return support.read_json(support.FIXTURES / "merged" / "one_cell.json")


@pytest.fixture
def result_files(fixtures_dir):
    """Mapping of fixture stem to path for every synthetic result file."""
    return {p.stem: p for p in sorted((fixtures_dir / "results").glob("*.json"))}


@pytest.fixture
def regen_golden(request):
    return request.config.getoption("--regen-golden")
