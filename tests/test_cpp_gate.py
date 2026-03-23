import pytest
from unittest.mock import MagicMock

# Stubbing the interface that Tony is expected to build in compliance_agent.py
class ComplianceAgent:
    def __init__(self, db_connection=None):
        self.db = db_connection or MagicMock()

    def check_route_compliance(self, route_id, vehicle_weight, vehicle_height):
        # This is essentially what Tony's implementation will check via Snowflake ST_INTERSECTS
        # For the stub, we just pretend we query it
        route_limits = self._get_route_limits_from_db(route_id)
        
        if route_limits is None:
            return "PASS" # Default if no limits found
            
        if vehicle_weight > route_limits.get("MIN_weight_limit_tons", float('inf')):
            return "HARD VETO"
            
        if vehicle_height > route_limits.get("MIN_vertical_clearance_mt", float('inf')):
            return "HARD VETO"
            
        return "PASS"
        
    def _get_route_limits_from_db(self, route_id):
        # In reality, this would run ST_INTERSECTS via snowflake. 
        # Here we mock the behavior.
        return self.db.execute(f"SELECT MIN(weight_limit_tons), MIN(vertical_clearance_mt) FROM routes WHERE id={route_id}")

@pytest.fixture
def mock_db():
    return MagicMock()

@pytest.fixture
def agent(mock_db):
    return ComplianceAgent(db_connection=mock_db)

def test_overweight_vehicle(agent, mock_db):
    # Setup mock to return a weight limit of 10 tons, height of 15 ft
    mock_db.execute.return_value = {"MIN_weight_limit_tons": 10, "MIN_vertical_clearance_mt": 15}
    
    # Test vehicle GVWR > 10
    result = agent.check_route_compliance(route_id=1, vehicle_weight=12, vehicle_height=10)
    assert result == "HARD VETO"
    mock_db.execute.assert_called_once()

def test_compliant_vehicle(agent, mock_db):
    # Setup mock to return a weight limit of 10 tons, height limit of 15 ft
    mock_db.execute.return_value = {"MIN_weight_limit_tons": 10, "MIN_vertical_clearance_mt": 15}
    
    # Test vehicle within limits
    result = agent.check_route_compliance(route_id=1, vehicle_weight=8, vehicle_height=10)
    assert result == "PASS"

def test_height_violation(agent, mock_db):
    # Setup mock to return a weight limit of 10 tons, height limit of 12 ft
    mock_db.execute.return_value = {"MIN_weight_limit_tons": 10, "MIN_vertical_clearance_mt": 12}
    
    # Test vehicle height > 12 ft
    result = agent.check_route_compliance(route_id=1, vehicle_weight=8, vehicle_height=14)
    assert result == "HARD VETO"
