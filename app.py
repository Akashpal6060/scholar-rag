"""HuggingFace Spaces entrypoint.

Includes a monkeypatch for a known gradio_client bug where api_info schema
generation crashes on boolean JSON-schema values ("argument of type 'bool' is
not iterable"). We patch the helper to treat bool schemas as "Any" before the
app loads. This is version-independent and harmless.
"""
import os
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")

# --- Patch gradio_client bug: bool schema crashes json_schema_to_python_type ---
import gradio_client.utils as _gcu

_orig_get_type = _gcu.get_type
def _safe_get_type(schema):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_get_type(schema)
_gcu.get_type = _safe_get_type

_orig_js2pt = _gcu._json_schema_to_python_type
def _safe_js2pt(schema, defs=None):
    if not isinstance(schema, dict):
        return "Any"
    return _orig_js2pt(schema, defs)
_gcu._json_schema_to_python_type = _safe_js2pt
# --- end patch ---

from src.api.ui import demo

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7860,
        ssr_mode=False,
        show_api=False,
    )
