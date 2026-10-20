import base64

import pytest
from fastapi.testclient import TestClient

from api.main import app
from api.service import MODEL_TAG
from data.df_import import DATA_DIR
from training.train import OUT_DIR, covariate_stats_path

MODEL_FILES = [OUT_DIR / f"best_{MODEL_TAG}.pt", covariate_stats_path(MODEL_TAG)]
DICOM = next(DATA_DIR.rglob("*.[Dd][Cc][Mm]"), None)

pytestmark = pytest.mark.skipif(
    not all(f.exists() for f in MODEL_FILES) or DICOM is None,
    reason="Faltan el modelo entrenado, su JSON de covariables o algún DICOM de ejemplo",
)

FORM = {"age": "65", "gender": "female", "bmi": "24.5"}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def post(client, form=FORM, file=None):
    files = {"dicom": ("radiografia.dcm", file if file is not None else DICOM.read_bytes())}
    return client.post("/predict", data=form, files=files)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model": MODEL_FILES[0].name}


def test_predict_returns_class_probabilities_and_bmd(client):
    r = post(client)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] in {"Sano", "Osteopenia", "Osteoporosis"}
    assert set(body["probabilities"]) == {"Sano", "Osteopenia", "Osteoporosis"}
    assert sum(body["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert body["probabilities"][body["predicted_class"]] == max(body["probabilities"].values())
    assert isinstance(body["bmd_estimated"], float)
    assert "heatmap_png_base64" not in body


def test_predict_is_deterministic(client):
    assert post(client).json() == post(client).json()


def test_heatmap_is_a_png_when_requested(client):
    r = post(client, {**FORM, "heatmap": "true"})
    assert r.status_code == 200
    assert base64.b64decode(r.json()["heatmap_png_base64"]).startswith(b"\x89PNG")


def test_gender_accepts_only_female_or_male(client):
    assert post(client, {**FORM, "gender": "male"}).status_code == 200
    assert post(client, {**FORM, "gender": "x"}).status_code == 422


@pytest.mark.parametrize("field,value", [("age", "200"), ("age", "-1"), ("bmi", "0"), ("bmi", "81")])
def test_out_of_range_covariates_are_rejected(client, field, value):
    assert post(client, {**FORM, field: value}).status_code == 422


@pytest.mark.parametrize("missing", ["age", "gender", "bmi"])
def test_missing_covariate_is_rejected(client, missing):
    form = {k: v for k, v in FORM.items() if k != missing}
    assert post(client, form).status_code == 422


def test_missing_file_is_rejected(client):
    assert client.post("/predict", data=FORM).status_code == 422


def test_non_dicom_file_is_rejected(client):
    r = post(client, file=b"esto no es un DICOM")
    assert r.status_code == 422
    assert "DICOM" in r.json()["detail"]
