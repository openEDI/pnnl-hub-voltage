import importlib
import json
import os
import sys
from unittest.mock import MagicMock, patch, mock_open

import pytest


# ---------------------------------------------------------------------------
# Helpers: import server module with hub_federate mocked out
# ---------------------------------------------------------------------------

@pytest.fixture
def server_mod(mock_helics):
    """Import server.py after mocking helics and hub_federate."""
    # server.py does `from hub_federate import run_simulator` at module level,
    # so we need hub_federate importable before importing server.
    fake_hub = MagicMock()
    sys.modules["hub_federate"] = fake_hub

    # Remove cached server module
    for key in list(sys.modules):
        if key == "server" or "server" in key and "pnnl" in key:
            sys.modules.pop(key, None)

    spec = importlib.util.spec_from_file_location(
        "server",
        str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src" / "pnnl-hub-voltage" / "server.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["server"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def client(server_mod):
    """Return a Starlette TestClient for the FastAPI app."""
    from starlette.testclient import TestClient

    return TestClient(server_mod.app)


# ===========================================================================
# TestKubernetesService
# ===========================================================================

class TestKubernetesService:
    def test_returns_k8s_service_name(self, server_mod, monkeypatch):
        # Clear the lru_cache / functools.cache
        server_mod.kubernetes_service.cache_clear()
        monkeypatch.setenv("KUBERNETES_SERVICE_NAME", "my-k8s-svc")
        monkeypatch.setenv("SERVICE_NAME", "fallback-svc")
        assert server_mod.kubernetes_service() == "my-k8s-svc"
        server_mod.kubernetes_service.cache_clear()

    def test_falls_back_to_service_name(self, server_mod, monkeypatch):
        server_mod.kubernetes_service.cache_clear()
        monkeypatch.delenv("KUBERNETES_SERVICE_NAME", raising=False)
        monkeypatch.setenv("SERVICE_NAME", "fallback-svc")
        assert server_mod.kubernetes_service() == "fallback-svc"
        server_mod.kubernetes_service.cache_clear()

    def test_returns_none_when_unset(self, server_mod, monkeypatch):
        server_mod.kubernetes_service.cache_clear()
        monkeypatch.delenv("KUBERNETES_SERVICE_NAME", raising=False)
        monkeypatch.delenv("SERVICE_NAME", raising=False)
        assert server_mod.kubernetes_service() is None
        server_mod.kubernetes_service.cache_clear()


# ===========================================================================
# TestBuildUrl
# ===========================================================================

class TestBuildUrl:
    def test_builds_url_without_k8s(self, server_mod, monkeypatch):
        server_mod.kubernetes_service.cache_clear()
        monkeypatch.delenv("KUBERNETES_SERVICE_NAME", raising=False)
        monkeypatch.delenv("SERVICE_NAME", raising=False)
        url = server_mod.build_url("myhost", 8080, ["api", "data"])
        assert url == "http://myhost:8080/api/data"
        server_mod.kubernetes_service.cache_clear()

    def test_builds_url_with_k8s(self, server_mod, monkeypatch):
        server_mod.kubernetes_service.cache_clear()
        monkeypatch.setenv("KUBERNETES_SERVICE_NAME", "k8s-ns")
        url = server_mod.build_url("myhost", 8080, ["api", "data"])
        assert url == "http://myhost.k8s-ns:8080/api/data"
        server_mod.kubernetes_service.cache_clear()

    @pytest.mark.xfail(reason="Log messages are swapped — logs 'docker-compose' for k8s and vice versa")
    def test_log_messages_correct(self, server_mod, monkeypatch, caplog):
        import logging

        # With k8s service set, should log "kubernetes" not "docker-compose"
        server_mod.kubernetes_service.cache_clear()
        monkeypatch.setenv("KUBERNETES_SERVICE_NAME", "k8s-ns")
        with caplog.at_level(logging.INFO):
            server_mod.build_url("host", 80, ["path"])
        assert "kubernetes" in caplog.text.lower()
        assert "docker-compose" not in caplog.text.lower()
        server_mod.kubernetes_service.cache_clear()


# ===========================================================================
# TestReadRoot
# ===========================================================================

class TestReadRoot:
    def test_health_check_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "hostname" in data
        assert "host_ip" in data


# ===========================================================================
# TestRunModel
# ===========================================================================

class TestRunModel:
    def test_successful_run(self, client, server_mod, monkeypatch):
        server_mod.kubernetes_service.cache_clear()
        monkeypatch.delenv("KUBERNETES_SERVICE_NAME", raising=False)
        monkeypatch.delenv("SERVICE_NAME", raising=False)

        sensor_data = [{"id": "sensor1", "value": 42}]

        mock_response = MagicMock()
        mock_response.json.return_value = sensor_data

        with patch.object(server_mod.requests, "get", return_value=mock_response) as mock_get, \
             patch.object(server_mod, "run_simulator") as mock_run_sim, \
             patch("builtins.open", mock_open()) as mocked_file:

            response = client.post("/run", json={
                "broker_ip": "10.0.0.1",
                "broker_port": 23404,
                "feeder_host": "feeder1",
                "feeder_port": 9090,
            })

        assert response.status_code == 200
        data = response.json()
        assert "detail" in data
        server_mod.kubernetes_service.cache_clear()


# ===========================================================================
# TestConfigure
# ===========================================================================

class TestConfigure:
    def test_writes_config_files(self, client, server_mod):
        payload = {
            "component": {
                "name": "hub_voltage",
                "type": "HubVoltage",
                "parameters": {"max_itr": 5, "number_of_timesteps": 10},
            },
            "links": [
                {
                    "source": "feeder0",
                    "source_port": "pub_v0",
                    "target": "hub",
                    "target_port": "sub_v0",
                },
                {
                    "source": "feeder1",
                    "source_port": "pub_v1",
                    "target": "hub",
                    "target_port": "sub_v1",
                },
            ],
        }

        with patch("builtins.open", mock_open()) as mocked_file:
            response = client.post("/configure", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "detail" in data

        # Verify open was called with the correct filenames
        opened_files = [c.args[0] for c in mocked_file.call_args_list]
        assert "input_mapping.json" in opened_files
        assert "static_inputs.json" in opened_files
