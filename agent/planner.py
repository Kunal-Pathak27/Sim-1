import math
from typing import Dict, List, Tuple


def pick_heading(
    heading_costs: List[Tuple[float, float]],
    goal_angle_deg: float,
    obs_w: float = 1.2,
    goal_w: float = 0.6,
    safe_thr: float = 0.35,
) -> Tuple[float, float]:
    best_angle = 0.0
    best_cost = float("inf")
    for ang, oc in heading_costs:
        ang_err = abs(ang - goal_angle_deg) / 60.0
        score = obs_w * oc + goal_w * ang_err
        if oc < safe_thr and score < best_cost:
            best_cost = score
            best_angle = ang
    if best_cost == float("inf"):
        # No safe option, pick the least bad
        for ang, oc in heading_costs:
            ang_err = abs(ang - goal_angle_deg) / 60.0
            score = obs_w * oc + goal_w * ang_err
            if score < best_cost:
                best_cost = score
                best_angle = ang
    return best_angle, best_cost


def step_from_heading(angle_deg: float, step_dist: float) -> Dict[str, float]:
    # Positive angle rotates to the left in image space; simulator uses yaw about +Y
    # We interpret forward as +z in simulator's relative move implementation
    return {"turn": float(angle_deg), "distance": float(step_dist)}


def goal_bearing_from_position(pos: Dict[str, float], goal: Dict[str, float]) -> float:
    # Simulator uses +z forward when yaw = 0; relative move adds (sin(yaw), 0, cos(yaw))
    # Bearing angle: desired rotation to face from pos -> goal
    dx = float(goal["x"]) - float(pos.get("x", 0.0))
    dz = float(goal["z"]) - float(pos.get("z", 0.0))
    # yaw 0 means facing +z; atan2 uses (dx,dz) so that 0 deg is straight ahead
    angle_rad = math.atan2(dx, dz)
    angle_deg = math.degrees(angle_rad)
    return float(angle_deg)


def close_to_goal(pos: Dict[str, float], goal: Dict[str, float], radius: float = 3.0) -> bool:
    dx = float(goal["x"]) - float(pos.get("x", 0.0))
    dz = float(goal["z"]) - float(pos.get("z", 0.0))
    return (dx * dx + dz * dz) ** 0.5 <= float(radius)


