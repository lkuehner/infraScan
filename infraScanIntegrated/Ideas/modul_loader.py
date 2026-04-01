import sys
import os
from pathlib import Path
from typing import Any


MODULE_ROOT = Path(__file__).resolve().parent
CODE_ROOT = MODULE_ROOT.parent

ROAD_ROOT = CODE_ROOT / 'infraScanRoad'
RAIL_ROOT = CODE_ROOT / 'infraScanRail'

def load_rail_modules() -> dict[str, Any]:
    """Load rail modules and expose key handles for notebooks/scripts."""

    rail_root = str(RAIL_ROOT)
    if rail_root not in sys.path:
        sys.path.insert(0, rail_root)

    os.chdir(CODE_ROOT)

    import infraScanRail.settings as rail_settings
    import infraScanRail.pipeline as rail_pipeline

    return {
        "settings": rail_settings,
        "pipeline": rail_pipeline,
        "RAIL_ROOT": RAIL_ROOT,
    }
