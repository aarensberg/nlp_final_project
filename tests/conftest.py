import pytest

from nlp_final_project.config import ARTIFACTS

HAVE_MODELS = (ARTIFACTS / "classifier" / "model.safetensors").exists() and (ARTIFACTS / "baseline.joblib").exists()
HAVE_CONFIG = (ARTIFACTS / "inference_config.json").exists()

needs_models = pytest.mark.skipif(not (HAVE_MODELS and HAVE_CONFIG),
                                  reason="trained models / inference config not in artifacts/")
