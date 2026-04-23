import os
import importlib.util
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import matplotlib
import numpy as np

_MAIN_PATH = Path(__file__).resolve().parents[1] / "main.py"
_SPEC = importlib.util.spec_from_file_location("main", _MAIN_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Failed to load module spec from {_MAIN_PATH}")
_MAIN = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MAIN)
FirnModel = _MAIN.FirnModel


matplotlib.use("Agg")


def configured_model() -> FirnModel:
    model = FirnModel()
    model.setup(
        dz=0.2,
        simDuration=0.2,
        scaleDuration=False,
        print_messages=False,
        interp_on_reg_z=False,
    )
    return model


def test_setup_populates_expected_state() -> None:
    model = configured_model()

    assert "y0" in model.p
    assert "z_h" in model.p
    assert model.p["N"] == len(model.p["z_h"])
    assert model.p["y0"].shape == (model.p["N"] * 4 + 1,)
    assert np.all(model.p["z_h"] >= 0.0)
    assert np.all(model.p["z_h"] <= 1.0)


def test_run_creates_expected_dataset_variables() -> None:
    model = configured_model()
    result = model.run()

    expected_vars = {
        "phi",
        "r2",
        "rho",
        "A",
        "T",
        "w",
        "h",
        "M",
        "FAC",
        "z830",
        "nu",
    }
    assert expected_vars.issubset(set(result.data_vars))
    assert result.sizes["z_h"] == model.p["N"]
    assert result.sizes["t"] > 1


def test_interp_regular_z_adds_regular_grid_variable() -> None:
    model = configured_model()
    model.run()
    model.interp_regular_z(var_name="phi")

    assert "phi_r" in model.results
    assert "z_r" in model.results.coords
    assert model.results["phi_r"].shape[1] == model.results.sizes["t"]


def test_upwind_difference_matrix_has_expected_shape_and_zero_row_sum() -> None:
    model = FirnModel()
    d_pos = model.upwind_difference_matrix(0.0, 1.0, 6, 1.0)
    d_neg = model.upwind_difference_matrix(0.0, 1.0, 6, -1.0)

    assert d_pos.shape == (6, 6)
    assert d_neg.shape == (6, 6)
    assert np.allclose(d_pos.sum(axis=1), 0.0)
    assert np.allclose(d_neg.sum(axis=1), 0.0)
