import json
import sys
from pathlib import Path

# Insert package source directory to python path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "pnnl-hub-voltage"))

from hub_federate import ComponentParameters  # noqa: E402


def test_schema_and_component_definition() -> None:
    """Regenerate schema.json from ComponentParameters and verify component_definition.json static_inputs alignment."""
    # 1. Regenerate schema.json from ComponentParameters
    schema_path = Path(__file__).resolve().parents[1] / "schema.json"
    model_schema = ComponentParameters.model_json_schema()
    schema_content = json.dumps(model_schema, indent=2) + "\n"
    with open(schema_path, "w", encoding="utf-8") as f:
        f.write(schema_content)

    # 2. Verify schema.json matches model_schema
    with open(schema_path, encoding="utf-8") as f:
        schema_json = json.load(f)
    assert model_schema == schema_json

    # 3. Load component_definition.json and verify static_inputs match schema properties
    comp_def_path = Path(__file__).resolve().parents[1] / "component_definition.json"
    with open(comp_def_path, encoding="utf-8") as f:
        comp_def = json.load(f)

    static_inputs = comp_def.get("static_inputs", [])
    static_input_names = {item["port_id"] for item in static_inputs}
    schema_properties = set(model_schema.get("properties", {}).keys()) - {"name"}

    missing = schema_properties - static_input_names
    extra = static_input_names - schema_properties
    assert static_input_names == schema_properties, (
        "Mismatch between component_definition.json static_inputs and ComponentParameters schema properties.\n"
        f"Missing in component_definition.json: {missing}\n"
        f"Extra in component_definition.json: {extra}"
    )
