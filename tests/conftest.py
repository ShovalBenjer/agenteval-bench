import pytest


@pytest.fixture
def _write_yaml(tmp_path):
    """Factory fixture for writing temporary YAML files."""

    def _write(content: str) -> str:
        path = tmp_path / "test.yaml"
        path.write_text(content)
        return str(path)

    return _write


@pytest.fixture
def _dummy_agent():
    """Create an agent function that always returns the given response."""

    def _dummy_agent(response: str):
        return lambda _input: response

    return _dummy_agent
