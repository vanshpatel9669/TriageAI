"""Session-wide test setup, loaded before tests/unit/ or tests/integration/
conftest.py regardless of which subdirectory pytest targets.
"""

from __future__ import annotations

import os

# Must be set before the first create_app() call anywhere in the session:
# the OTel SDK's global TracerProvider can only be configured once per
# process, and the default console exporter would otherwise dump a full
# JSON span for every request every test makes across the whole suite.
os.environ.setdefault("TRIAGEAI_OTEL_CONSOLE_EXPORT", "false")
