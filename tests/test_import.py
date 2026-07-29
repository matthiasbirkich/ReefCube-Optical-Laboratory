"""Basic tests for the ReefCube package."""

import reefcube


def test_package_import() -> None:
    """Verify that the ReefCube package can be imported."""
    assert reefcube is not None


def test_package_version() -> None:
    """Verify that the package exposes a non-empty version string."""
    assert hasattr(reefcube, "__version__")
    assert isinstance(reefcube.__version__, str)
    assert reefcube.__version__
