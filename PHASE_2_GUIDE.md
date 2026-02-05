# 🚀 Phase 2 & Running the Pipeline

## 1. Fix for ImportError
You encountered an error because `pipeline.py` uses relative imports (e.g., `from .config import...`) which require it to be run as a module or from the package root.

**Do NOT run:**
`python c:/Users/SATWIK/Documents/Phishing/phishing_pipeline/pipeline.py`

**DO run (from root directory):**
```powershell
python main_controller.py
```

## 2. Phase 2 Architecture (Sequential Processing)

Phase 2 runs sequentially (one domain at a time) to handle WHOIS rate limits correctly.

| Component | File | Lines | Description |
|-----------|------|-------|-------------|
| **Core Loop** | `pipeline.py` | 442-568 | Iterates through `features_enriched.csv` rows |
| **WHOIS** | `pipeline.py` | 460-496 | **Blocking** call suitable for rate limiting (now fixed with retry) |
| **DNS** | `pipeline.py` | 512-525 | Performs A, NS, MX lookups |
| **Classification** | `pipeline.py` | 546 (call) | Calls `reclassify_label` |
| **Logic** | `pipeline.py` | 323-356 | `reclassify_label` definition (Brand checks, defaults) |

**Key Files:**
- `phishing_pipeline/pipeline.py`: Contains **all** the logic for Phase 2.
- `main_controller.py`: The entry point that properly initializes the package and triggers `run_pipeline`.
