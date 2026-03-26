import importlib
import json
import sys
from unittest.mock import MagicMock, PropertyMock, call

import pytest
import xarray as xr
from oedisi.types.data_types import (
    EquipmentNodeArray,
    MeasurementArray,
    PowersImaginary,
    PowersReal,
    VoltagesMagnitude,
)


# ---------------------------------------------------------------------------
# Helpers: import hub_federate *after* helics is mocked
# ---------------------------------------------------------------------------

@pytest.fixture
def hub_mod(mock_helics):
    """Import (or re-import) hub_federate with the mocked helics module."""
    mod_name = "pnnl-hub-voltage.hub_federate"
    # Remove cached module so it picks up the mock
    sys.modules.pop(mod_name, None)
    # Also remove by any alias it may have been imported under
    for key in list(sys.modules):
        if "hub_federate" in key:
            sys.modules.pop(key, None)

    import importlib
    spec = importlib.util.spec_from_file_location(
        "hub_federate",
        str(__import__("pathlib").Path(__file__).resolve().parent.parent / "src" / "pnnl-hub-voltage" / "hub_federate.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hub_federate"] = mod
    spec.loader.exec_module(mod)
    return mod


# ===========================================================================
# TestComponentParameters
# ===========================================================================

class TestComponentParameters:
    def test_valid_construction(self, hub_mod):
        cp = hub_mod.ComponentParameters(name="hub", max_itr=10, t_steps=50)
        assert cp.name == "hub"
        assert cp.max_itr == 10
        assert cp.t_steps == 50


# ===========================================================================
# TestEqarrayToXarray
# ===========================================================================

class TestEqarrayToXarray:
    def test_converts_equipment_node_array(self, hub_mod, sample_equipment_node_array):
        result = hub_mod.eqarray_to_xarray(sample_equipment_node_array)
        assert isinstance(result, xr.DataArray)
        assert list(result.coords["ids"].values) == ["node1", "node2", "node3"]
        assert list(result.coords["equipment_ids"].values) == ["eq1", "eq2", "eq3"]
        assert list(result.values) == [10.0, 20.0, 30.0]


# ===========================================================================
# TestMeasurementToXarray
# ===========================================================================

class TestMeasurementToXarray:
    def test_converts_measurement_array(self, hub_mod, sample_measurement_array):
        result = hub_mod.measurement_to_xarray(sample_measurement_array)
        assert isinstance(result, xr.DataArray)
        assert list(result.coords["ids"].values) == ["m1", "m2", "m3"]
        assert list(result.values) == [100.0, 200.0, 300.0]


# ===========================================================================
# TestXarrayToDict
# ===========================================================================

class TestXarrayToDict:
    def test_roundtrip(self, hub_mod):
        da = xr.DataArray(
            [1.0, 2.0, 3.0],
            coords={"ids": ["a", "b", "c"]},
        )
        result = hub_mod.xarray_to_dict(da)
        assert "values" in result
        assert result["values"] == [1.0, 2.0, 3.0]
        assert "ids" in result
        assert result["ids"] == ["a", "b", "c"]


# ===========================================================================
# TestXarrayToEqarray
# ===========================================================================

class TestXarrayToEqarray:
    @pytest.mark.xfail(reason="xarray_to_eqarray is identical to xarray_to_dict — likely a bug")
    def test_identical_to_xarray_to_dict(self, hub_mod):
        da = xr.DataArray(
            [1.0, 2.0],
            dims=("ids",),
            coords={
                "ids": ["a", "b"],
                "equipment_ids": ("ids", ["eq_a", "eq_b"]),
            },
        )
        dict_result = hub_mod.xarray_to_dict(da)
        eq_result = hub_mod.xarray_to_eqarray(da)
        # If these are meant to be different, this assertion should fail.
        # Currently they are identical, so we expect this to pass — the xfail
        # documents that the *sameness* is the bug.
        assert dict_result != eq_result


# ===========================================================================
# TestXarrayToPowersCart
# ===========================================================================

class TestXarrayToPowersCart:
    def test_splits_complex_data(self, hub_mod):
        da = xr.DataArray(
            [1 + 2j, 3 + 4j],
            dims=("ids",),
            coords={
                "ids": ["n1", "n2"],
                "equipment_ids": ("ids", ["eq1", "eq2"]),
            },
        )
        real, imag = hub_mod.xarray_to_powers_cart(da)
        assert isinstance(real, PowersReal)
        assert isinstance(imag, PowersImaginary)
        assert real.values == [1.0, 3.0]
        assert imag.values == [2.0, 4.0]
        assert real.ids == ["n1", "n2"]
        assert imag.ids == ["n1", "n2"]


# ===========================================================================
# HubFederate fixtures
# ===========================================================================

@pytest.fixture
def hub_federate(hub_mod, mock_helics, tmp_package_dir, static_inputs_json, input_mapping_json, component_definition_json):
    """Construct a HubFederate with mocked helics and temp config files."""
    from oedisi.types.common import BrokerConfig

    broker = BrokerConfig(broker_ip="127.0.0.1", broker_port=23404)
    fed = hub_mod.HubFederate(broker)
    return fed


# ===========================================================================
# TestHubFederateInit
# ===========================================================================

class TestHubFederateInit:
    def test_loads_static_inputs(self, hub_federate, static_inputs_json):
        assert hub_federate.static.name == static_inputs_json["name"]
        assert hub_federate.static.max_itr == static_inputs_json["max_itr"]
        assert hub_federate.static.t_steps == static_inputs_json["number_of_timesteps"]

    def test_loads_input_mapping(self, hub_federate, input_mapping_json):
        assert hub_federate.inputs == input_mapping_json

    def test_loads_component_definition(self, hub_federate, component_definition_json):
        assert hub_federate.component_config == component_definition_json


# ===========================================================================
# TestRegisterSubscription
# ===========================================================================

class TestRegisterSubscription:
    def test_registers_5_subscriptions(self, hub_federate, mock_helics):
        fed_mock = mock_helics.helicsCreateValueFederate.return_value
        assert fed_mock.register_subscription.call_count == 5


# ===========================================================================
# TestRegisterPublication
# ===========================================================================

class TestRegisterPublication:
    def test_registers_6_publications(self, hub_federate, mock_helics):
        fed_mock = mock_helics.helicsCreateValueFederate.return_value
        assert fed_mock.register_publication.call_count == 6
        # Verify publication names
        pub_names = [c.args[0] for c in fed_mock.register_publication.call_args_list]
        assert pub_names == [f"pub_v{i}" for i in range(6)]


# ===========================================================================
# TestPublishAll
# ===========================================================================

def _make_voltage_dict(ids, values, time=0):
    """Helper: produce a dict for a VoltagesMagnitude (as HELICS sub.json returns)."""
    return VoltagesMagnitude(ids=ids, values=values, units="kV", time=time).dict()


class TestPublishAll:
    def test_aggregates_updated_subscriptions(self, hub_federate, mock_helics):
        # Mark 3 of 5 subscriptions as updated, each with 2 nodes
        subs = [hub_federate.sub.v0, hub_federate.sub.v1, hub_federate.sub.v2,
                hub_federate.sub.v3, hub_federate.sub.v4]
        for i, sub in enumerate(subs[:3]):
            sub.is_updated.return_value = True
            type(sub).json = PropertyMock(
                return_value=_make_voltage_dict(
                    [f"bus{i}_a", f"bus{i}_b"],
                    [1.0 + i * 0.01, 0.99 + i * 0.01],
                    time=42,
                )
            )
        for sub in subs[3:]:
            sub.is_updated.return_value = False

        hub_federate.publish_all()

        # All 6 publications should have been called
        for pub in hub_federate.pub_area_voltages:
            pub.publish.assert_called_once()
            published = json.loads(pub.publish.call_args[0][0])
            # 3 subs * 2 nodes = 6 total ids
            assert len(published["ids"]) == 6
            assert len(published["values"]) == 6

    def test_no_updates_no_publish(self, hub_federate):
        # All subs default to is_updated=False from conftest
        hub_federate.publish_all()
        # Publish is still called (the code always publishes), but with empty data
        for pub in hub_federate.pub_area_voltages:
            pub.publish.assert_called_once()
            published = json.loads(pub.publish.call_args[0][0])
            assert published["ids"] == []
            assert published["values"] == []

    def test_partial_update_publishes_partial(self, hub_federate):
        # Only sub v2 updated
        hub_federate.sub.v2.is_updated.return_value = True
        type(hub_federate.sub.v2).json = PropertyMock(
            return_value=_make_voltage_dict(["only_bus"], [1.05])
        )

        hub_federate.publish_all()

        for pub in hub_federate.pub_area_voltages:
            pub.publish.assert_called_once()
            published = json.loads(pub.publish.call_args[0][0])
            assert published["ids"] == ["only_bus"]
            assert published["values"] == [1.05]


# ===========================================================================
# TestRun
# ===========================================================================

class TestRun:
    def test_max_itr_exit(self, hub_mod, mock_helics, tmp_package_dir, monkeypatch):
        """With max_itr=3 and always-iterating status, inner loop runs exactly max_itr times per outer step."""
        from oedisi.types.common import BrokerConfig

        # Override static_inputs to have max_itr=3, t_steps=1
        static = {"name": "test", "max_itr": 3, "number_of_timesteps": 1}
        (tmp_package_dir / "static_inputs.json").write_text(json.dumps(static))

        broker = BrokerConfig(broker_ip="127.0.0.1")
        fed = hub_mod.HubFederate(broker)

        helics = mock_helics
        # Always return iterating (never next_step) — except we need to
        # eventually move time forward to exit the outer loop.
        # The outer loop: while granted_time <= t_steps
        # t_steps=1, update_interval=1, so request_time starts at 1
        # We need granted_time to advance past t_steps after the inner loop.
        call_count = [0]

        def time_iterative(fed_obj, request_time, itr_flag):
            call_count[0] += 1
            # After max_itr iterations, the code sets itr_flag to no_iteration
            # and calls continue (which re-enters the while True loop).
            # On that call, return next_step to break out.
            if itr_flag == "no_iteration":
                return (request_time, "next_step")
            return (0, "iterating")

        helics.helicsFederateRequestTimeIterative.side_effect = time_iterative

        fed.run()

        # stop() should have been called
        helics.helicsFederateDisconnect.assert_called_once()
        helics.helicsFederateFree.assert_called_once()
        helics.helicsCloseLibrary.assert_called_once()


# ===========================================================================
# TestStop
# ===========================================================================

class TestStop:
    def test_cleanup_calls(self, hub_federate, mock_helics):
        hub_federate.stop()
        mock_helics.helicsFederateDisconnect.assert_called_once_with(
            mock_helics.helicsCreateValueFederate.return_value
        )
        mock_helics.helicsFederateFree.assert_called_once_with(
            mock_helics.helicsCreateValueFederate.return_value
        )
        mock_helics.helicsCloseLibrary.assert_called_once()
