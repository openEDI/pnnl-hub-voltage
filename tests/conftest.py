import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, PropertyMock

import pytest
import xarray as xr
from oedisi.types.data_types import (
    EquipmentNodeArray,
    MeasurementArray,
    VoltagesMagnitude,
)


# ---------------------------------------------------------------------------
# Sample data fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_voltages_magnitude():
    return VoltagesMagnitude(
        ids=["bus1", "bus2", "bus3"],
        values=[1.01, 0.99, 1.02],
        units="kV",
        time=0,
    )


@pytest.fixture
def sample_equipment_node_array():
    return EquipmentNodeArray(
        ids=["node1", "node2", "node3"],
        values=[10.0, 20.0, 30.0],
        equipment_ids=["eq1", "eq2", "eq3"],
        units="kW",
    )


@pytest.fixture
def sample_measurement_array():
    return MeasurementArray(
        ids=["m1", "m2", "m3"],
        values=[100.0, 200.0, 300.0],
        units="V",
    )


# ---------------------------------------------------------------------------
# JSON config fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def static_inputs_json():
    return {"name": "test_hub", "max_itr": 5, "number_of_timesteps": 10}


@pytest.fixture
def input_mapping_json():
    return {
        "sub_v0": "feeder0/pub_v0",
        "sub_v1": "feeder1/pub_v1",
        "sub_v2": "feeder2/pub_v2",
        "sub_v3": "feeder3/pub_v3",
        "sub_v4": "feeder4/pub_v4",
    }


@pytest.fixture
def component_definition_json():
    return {
        "directory": "hub_voltage",
        "execute_function": "python hub_federate.py",
        "static_inputs": [{"type": "", "port_id": "max_itr"}],
        "dynamic_inputs": [
            {"type": "VoltagesMagnitude", "port_id": f"sub_v{i}", "optional": True}
            for i in range(5)
        ],
        "dynamic_outputs": [
            {"type": "VoltagesMagnitude", "port_id": f"pub_v{i}", "optional": True}
            for i in range(5)
        ],
    }


# ---------------------------------------------------------------------------
# Mock helics module
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_helics():
    """Patch the entire helics module with a MagicMock before hub_federate is imported."""
    helics = MagicMock()

    # Constants used by production code
    helics.HELICS_CORE_TYPE_ZMQ = 0
    helics.HELICS_PROPERTY_TIME_PERIOD = 140
    helics.HELICS_DATA_TYPE_STRING = "string"
    helics.helics_iteration_request_iterate_if_needed = "iterate_if_needed"
    helics.helics_iteration_request_no_iteration = "no_iteration"
    helics.helics_iteration_result_next_step = "next_step"

    # helicsCreateFederateInfo -> returns a mock info object
    mock_info = MagicMock()
    helics.helicsCreateFederateInfo.return_value = mock_info

    # helicsCreateValueFederate -> returns a mock federate
    mock_fed = MagicMock()
    helics.helicsCreateValueFederate.return_value = mock_fed

    # fed.register_subscription returns mock subscription objects
    def make_sub(topic, _units):
        sub = MagicMock()
        sub.is_updated.return_value = False
        type(sub).json = PropertyMock(return_value="{}")
        return sub

    mock_fed.register_subscription.side_effect = make_sub

    # fed.register_publication returns mock publication objects
    mock_fed.register_publication.side_effect = lambda name, dtype, units: MagicMock()

    # helicsFederateGetTimeProperty -> 1 (second)
    helics.helicsFederateGetTimeProperty.return_value = 1

    sys.modules["helics"] = helics
    yield helics
    # Restore: remove the fake helics so it doesn't leak between test files
    sys.modules.pop("helics", None)


# ---------------------------------------------------------------------------
# Temporary package directory with JSON config files
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_package_dir(tmp_path, monkeypatch, static_inputs_json, input_mapping_json, component_definition_json):
    """Write config JSON files to tmp_path and patch Path in hub_federate module."""
    for name, data in [
        ("static_inputs.json", static_inputs_json),
        ("input_mapping.json", input_mapping_json),
        ("component_definition.json", component_definition_json),
    ]:
        (tmp_path / name).write_text(json.dumps(data))

    _original_path = Path

    class _RedirectPath(_original_path):
        _flavour = _original_path._flavour

        def __new__(cls, *args, **kwargs):
            return _original_path.__new__(cls, *args, **kwargs)

        @property
        def parent(self):
            real_parent = _original_path.parent.fget(self)
            if "hub_federate" in str(self):
                return tmp_path
            return real_parent

    # Patch Path in the hub_federate module (already imported by hub_mod fixture)
    if "hub_federate" in sys.modules:
        monkeypatch.setattr(sys.modules["hub_federate"], "Path", _RedirectPath)

    return tmp_path
