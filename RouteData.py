from dataclasses import dataclass


@dataclass
class RouteData:
    """Data class can act like a record in Python"""
    vehicle_id: int
    node_sequence: list
    distance: int
    fuel_cost: int
    load: int