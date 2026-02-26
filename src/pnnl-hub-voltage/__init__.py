"""
OEDISI pnnl-hub-voltage - voltage aggregator/distributor

This component provides:
- HELICS co-simulation wrapper for distribution feeder voltages
"""

__version__ = "0.1.0"

from .hub_federate import ComponentParameters, StaticConfig, HubFederate

__all__ = [
    "__version__",
    "HubFederate",
    "StaticConfig",
    "ComponentParameters",
]
