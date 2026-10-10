"""Render every dashboard page headlessly with streamlit.testing (run as a subprocess so that
MARVIS_OUTPUTS / MARVIS_DATA env overrides take effect). Prints a JSON report."""
import json
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "dashboard.py")
report = {}
at = AppTest.from_file(APP, default_timeout=120).run()
pages = at.sidebar.radio(key="page").options
for p in pages:
    at.sidebar.radio(key="page").set_value(p).run()
    report[p] = {"exceptions": [str(e.value) for e in at.exception],
                 "warnings": [w.value for w in at.warning], "errors": [e.value for e in at.error],
                 "n_dataframes": len(at.dataframe)}
print(json.dumps(report))
