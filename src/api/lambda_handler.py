"""
AWS Lambda entry point. This is the only file that knows it is on Lambda.

    Lambda handler:  api.lambda_handler.handler

WHY A SEPARATE FILE
-------------------
`main.py` stays a plain FastAPI app that runs under uvicorn exactly as before.
Everything Lambda-specific lives here, so local development and the deployed
service run the same application object and cannot drift apart.

THE COLD-START DETAIL THAT MATTERS
----------------------------------
Mangum runs the ASGI lifespan cycle around EVERY invocation, so leaving lifespan
on would re-load five joblib artefacts plus the dataset on every single request
(~1s each). That would not be visibly broken — just silently, badly slow.

So: lifespan is off, and `load_models()` is called here at MODULE IMPORT time.
Lambda imports the handler module once per container and then reuses it for
every subsequent invocation, which gives us the same "load once, serve many"
behaviour the uvicorn lifespan hook gives locally.

Cold start is therefore ~3-6s (import + model load) and warm invocations are
single-digit ms plus SHAP. Give the function 1024 MB and a 30s timeout: memory
on Lambda also scales CPU, and xgboost/shap import time drops sharply above
1 GB, so 1024 MB is genuinely cheaper per request than 512 MB despite the
higher per-ms rate.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Same path setup main.py does, so `predict`, `features`, `rules_engine` and the
# agent package resolve identically whether we came in via uvicorn or Lambda.
HERE = Path(__file__).resolve().parent
SRC = HERE.parent
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "agent"))
sys.path.insert(0, str(HERE))

from mangum import Mangum

from main import app, load_models

# --- one-time, per-container init -----------------------------------------
_t0 = time.time()
load_models()
print(f"[lambda] cold start init finished in {time.time() - _t0:.2f}s")

# lifespan="off" is load-bearing — see module docstring.
handler = Mangum(app, lifespan="off")
