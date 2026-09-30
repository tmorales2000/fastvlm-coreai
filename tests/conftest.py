"""
tests/conftest.py — Shared pytest configuration for fastvlm-coreai tests.
"""
import pytest
from pathlib import Path

LLM_RUNNER_DEFAULT = Path.home() / "git/apple/coreai-models/.build/out/Products/Debug/llm-runner"


def pytest_addoption(parser):
    parser.addoption("--variant", default="0.5b", choices=["0.5b", "1.5b", "7b"],
                     help="FastVLM variant to test (default: 0.5b)")
    parser.addoption("--llm-runner", type=Path, default=LLM_RUNNER_DEFAULT,
                     help="Path to llm-runner binary")


@pytest.fixture
def variant(pytestconfig):
    return pytestconfig.getoption("--variant")


@pytest.fixture
def llm_runner(pytestconfig):
    return pytestconfig.getoption("--llm-runner")
