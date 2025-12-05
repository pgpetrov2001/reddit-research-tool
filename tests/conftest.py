from __future__ import annotations

import sys
from pathlib import Path
import pytest

# Add parent directory to path so we can import arctic_shift
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))


# Configure pytest-asyncio
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "asyncio: mark test as an asyncio test"
    )


# Set asyncio mode to auto
@pytest.fixture(scope="session")
def event_loop_policy():
    """Set event loop policy for async tests."""
    import asyncio
    return asyncio.get_event_loop_policy()

