import pytest
from fastapi.testclient import TestClient

import api.web
from test_api import pytestmark  # noqa: F401  (mismos requisitos: modelo y un DICOM)


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    dist = tmp_path_factory.mktemp("dist")
    (dist / "index.html").write_text("<!doctype html><title>LumosNet</title>")
    (dist / "favicon.svg").write_text("<svg/>")
    (dist.parent / "secreto.txt").write_text("no")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(api.web, "FRONTEND_DIR", dist)
        with TestClient(api.web.app) as c:
            yield c


def test_api_is_mounted_under_api_and_loads_the_model(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"
    assert r.headers["cache-control"] == "no-store"


def test_react_routes_fall_back_to_index(client):
    for path in ("/", "/proyecto/el-modelo"):
        r = client.get(path)
        assert r.status_code == 200
        assert "LumosNet" in r.text
        assert r.headers["cache-control"] == "no-cache"


def test_static_files_are_served(client):
    assert client.get("/favicon.svg").text == "<svg/>"


def test_paths_outside_the_build_are_not_served(client):
    r = client.get("/..%2Fsecreto.txt")
    assert "LumosNet" in r.text
