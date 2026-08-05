"""Offline evaluation harness for the barcode-scanner pipeline.

Runs ``analyze_image()`` on the ground-truth dataset (``dataset.json``) and
scores results with LangSmith ``evaluate()``.

See ``runner.py`` for the harness. ``test_eval.py`` runs a deterministic
offline version that does NOT call LangSmith or Gemini — it asserts the
dataset loads and the target/evaluators work end-to-end with a stub.
"""
