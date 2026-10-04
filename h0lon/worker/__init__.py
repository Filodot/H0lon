"""Speech recognition worker for Google Colab and for this PC (docs/ARCHITECTURE.md, M6).

`python -m h0lon.worker --port 8790` starts it; `create_app` builds the FastAPI application
(`h0lon.worker.server`). The client is `h0lon.extract.asr` (`compute.asr = "colab"`).
"""

from h0lon.worker.server import APP_NAME, PROTOCOL, TOKEN_ENV, create_app

__all__ = ["APP_NAME", "PROTOCOL", "TOKEN_ENV", "create_app"]
