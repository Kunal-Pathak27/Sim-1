import asyncio
import json
import websockets
from flask import Flask, request, jsonify
import threading
import os
import http
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)

# --- CORS: allow simple cross-origin calls from control page ---
@app.after_request
def add_cors_headers(resp):
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Methods'] = 'GET,POST,OPTIONS'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    return resp

# ---------------------------
# Globals
# ---------------------------
connected = set()
async_loop = None
collision_count = 0  # <-- new: server-tracked collisions
image_queue: "asyncio.Queue[dict]" = None  # set in main
autonomous_enabled = True
goal_reached_flag = False
goal_reached_position = None
goal_reached_event: "asyncio.Event" = None

FLOOR_HALF = 50  # index.html uses PlaneGeometry(100, 100) centered at origin

# NEW: Configurable WS host/port via env
WS_HOST = os.getenv("WS_HOST", "localhost")
WS_PORT = int(os.getenv("WS_PORT", "8080"))

def corner_to_coords(corner: str, margin=5):
    c = corner.upper()
    x = FLOOR_HALF - margin if "E" in c else -(FLOOR_HALF - margin)
    z = FLOOR_HALF - margin if ("S" in c or "B" in c) else -(FLOOR_HALF - margin)
    if c in ("NE", "EN", "TR"): x, z = (FLOOR_HALF - margin, -(FLOOR_HALF - margin))
    if c in ("NW", "WN", "TL"): x, z = (-(FLOOR_HALF - margin), -(FLOOR_HALF - margin))
    if c in ("SE", "ES", "BR"): x, z = (FLOOR_HALF - margin, (FLOOR_HALF - margin))
    if c in ("SW", "WS", "BL"): x, z = (-(FLOOR_HALF - margin), (FLOOR_HALF - margin))
    return {"x": x, "y": 0, "z": z}

# ---------------------------
# WebSocket Handler
# ---------------------------
async def ws_handler(websocket, path=None):
    global collision_count, goal_reached_flag, goal_reached_position
    print("Client connected via WebSocket")
    connected.add(websocket)
    try:
        async for message in websocket:
            # Record any simulator messages and increment collision_count on "collision"
            try:
                data = json.loads(message)
                if isinstance(data, dict):
                    if data.get("type") == "collision" and data.get("collision"):
                        collision_count += 1
                    elif data.get("type") == "capture_image_response" and image_queue is not None:
                        # forward to controller
                        try:
                            image_queue.put_nowait(data)
                        except Exception:
                            pass
                    elif data.get("type") == "goal_reached":
                        goal_reached_flag = True
                        goal_reached_position = data.get("position")
                        if goal_reached_event is not None:
                            try:
                                goal_reached_event.set()
                            except Exception:
                                pass
            except Exception:
                pass
            print("Received from simulator:", message)
    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected")
    finally:
        connected.remove(websocket)

# Gracefully respond to non-WebSocket HTTP requests on the WS port
# to avoid noisy upgrade errors when users visit ws:// URL directly.
def http_fallback(protocol, request):
    # websockets 15 passes a Request object with .headers (mapping-like)
    headers = getattr(request, 'headers', request)
    try:
        connection_header = (headers.get('Connection') or '').lower()
        upgrade_header = (headers.get('Upgrade') or '').lower()
    except Exception:
        connection_header = ''
        upgrade_header = ''
    if 'upgrade' not in connection_header or upgrade_header != 'websocket':
        body = b'This endpoint expects a WebSocket connection. Load index.html instead.\n'
        return (
            http.HTTPStatus.UPGRADE_REQUIRED,
            [("Content-Type", "text/plain"), ("Content-Length", str(len(body)))],
            body,
        )


def broadcast(msg: dict):
    if not connected:
        return False
    for ws in list(connected):
        asyncio.run_coroutine_threadsafe(ws.send(json.dumps(msg)), async_loop)
    return True


# ---------------------------
# Autonomous Vision Controller
# ---------------------------
def _decode_datauri_png_to_image(data_uri: str) -> Image.Image:
    if not isinstance(data_uri, str):
        raise ValueError("image payload missing")
    if "," in data_uri:
        b64 = data_uri.split(",", 1)[1]
    else:
        b64 = data_uri
    raw = base64.b64decode(b64)
    img = Image.open(BytesIO(raw)).convert("RGB")
    return img


def _analyze_image_for_goal_and_obstacles(img: Image.Image) -> dict:
    # Downscale for speed
    target_width = 320
    w, h = img.size
    if w > target_width:
        new_h = int(h * (target_width / w))
        img = img.resize((target_width, new_h))
        w, h = img.size
    pixels = img.load()
    # Scan a mid band and a near band (closer to robot at bottom of image)
    y_start = int(h * 0.45)
    y_end = int(h * 0.70)
    near_start = int(h * 0.70)
    near_end = int(h * 0.92)
    # Accumulators
    obstacle_hist = [0] * w
    goal_sum_x = 0
    goal_count = 0
    for y in range(y_start, y_end):
        for x in range(0, w, 2):  # stride 2 for speed
            r, g, b = pixels[x, y]
            # obstacle ~ green boxes (0x00ff00)
            if g > 150 and r < 100 and b < 100:
                obstacle_hist[x] += 1
            # goal flag ~ cyan (0x00ccff)
            if g > 120 and b > 180 and r < 100:
                goal_sum_x += x
                goal_count += 1

    near_hist = [0] * w
    for y in range(near_start, near_end):
        for x in range(0, w, 2):
            r, g, b = pixels[x, y]
            if g > 140 and r < 110 and b < 110:
                near_hist[x] += 1
    # Compute densities
    left_density = sum(obstacle_hist[: w // 3])
    center_density = sum(obstacle_hist[w // 3 : (2 * w) // 3])
    right_density = sum(obstacle_hist[(2 * w) // 3 :])

    near_left = sum(near_hist[: w // 3])
    near_center = sum(near_hist[w // 3 : (2 * w) // 3])
    near_right = sum(near_hist[(2 * w) // 3 :])
    goal_centroid = (goal_sum_x / goal_count) if goal_count > 0 else None
    return {
        "width": w,
        "left_density": left_density,
        "center_density": center_density,
        "right_density": right_density,
        "goal_centroid": goal_centroid,
        "goal_visible": goal_count > 0,
        "near_left": near_left,
        "near_center": near_center,
        "near_right": near_right,
    }


def _plan_move_from_vision(vision: dict, robot_heading_deg: float | None = None, robot_pos: dict | None = None, goal_pos: dict | None = None) -> dict:
    w = vision["width"]
    left_d = vision["left_density"]
    center_d = vision["center_density"]
    right_d = vision["right_density"]
    goal_cx = vision["goal_centroid"]
    near_left = vision["near_left"]
    near_center = vision["near_center"]
    near_right = vision["near_right"]

    # Parameters
    max_turn_deg = 30.0
    forward_step = 0.5
    obstacle_threshold = 150  # earlier avoidance
    near_blocked_threshold = 1  # any near obstacle blocks forward

    # Hard safety: if near center has any obstacle, do turn-in-place away from denser side
    if near_center >= near_blocked_threshold:
        if near_left < near_right:
            return {"turn": -max_turn_deg, "distance": 0.0}
        else:
            return {"turn": max_turn_deg, "distance": 0.0}

    if vision["goal_visible"] and goal_cx is not None:
        # steer toward goal based on horizontal offset
        offset = (goal_cx - (w / 2)) / (w / 2)  # -1 .. +1
        turn = max(-max_turn_deg, min(max_turn_deg, offset * max_turn_deg))
        # if center blocked, bias turn away from denser side
        if center_d > obstacle_threshold:
            if left_d < right_d:
                turn -= 10
            else:
                turn += 10
            turn = max(-max_turn_deg, min(max_turn_deg, turn))
        return {"turn": float(turn), "distance": forward_step}

    # No goal seen: avoid obstacles; choose clearer side
    if center_d > obstacle_threshold:
        if left_d < right_d:
            return {"turn": -max_turn_deg, "distance": 0.0}
        else:
            return {"turn": max_turn_deg, "distance": 0.0}
    # path clear, go forward
    # If goal not visible, bias heading toward geometric goal direction if available
    if robot_heading_deg is not None and robot_pos is not None and goal_pos is not None:
        dx = float(goal_pos.get("x", 0.0)) - float(robot_pos.get("x", 0.0))
        dz = float(goal_pos.get("z", 0.0)) - float(robot_pos.get("z", 0.0))
        # camera forward is +z, robot yaw increases with left turn; use atan2 to compute desired yaw
        import math
        desired_yaw = math.degrees(math.atan2(dx, dz))  # degrees
        yaw_err = (desired_yaw - robot_heading_deg + 540.0) % 360.0 - 180.0  # wrap to [-180,180]
        yaw_correction = max(-max_turn_deg, min(max_turn_deg, yaw_err))
        if abs(yaw_correction) > 5.0:
            return {"turn": float(yaw_correction), "distance": 0.0}
        return {"turn": 0.0, "distance": forward_step}

    return {"turn": 0.0, "distance": forward_step}


async def autonomous_controller():
    # Wait for a simulator to connect
    while not connected:
        await asyncio.sleep(0.2)

    # Set goal near a corner (NE by default)
    goal_pos = corner_to_coords("NE")
    broadcast({"command": "set_goal", "position": goal_pos})
    print(f"[AUTO] Goal set: {goal_pos}")

    # Control loop
    while autonomous_enabled:
        # Request an image
        broadcast({"command": "capture_image"})
        # Wait for response
        try:
            msg = await asyncio.wait_for(image_queue.get(), timeout=2.0)
        except asyncio.TimeoutError:
            continue
        try:
            img = _decode_datauri_png_to_image(msg.get("image"))
        except Exception as e:
            print(f"[AUTO] Failed to decode image: {e}")
            continue
        vision = _analyze_image_for_goal_and_obstacles(img)
        heading = float(msg.get("headingDeg")) if isinstance(msg.get("headingDeg"), (int, float)) else None
        position = msg.get("position") if isinstance(msg.get("position"), dict) else None
        plan = _plan_move_from_vision(vision, robot_heading_deg=heading, robot_pos=position, goal_pos=goal_pos)
        # Send relative move
        cmd = {"command": "move_relative", "turn": plan["turn"], "distance": plan["distance"]}
        broadcast(cmd)
        await asyncio.sleep(0.1)

# ---------------------------
# Existing Endpoints
# ---------------------------
@app.route('/move', methods=['POST'])
def move():
    data = request.get_json()
    if not data or 'x' not in data or 'z' not in data:
        return jsonify({'error': 'Missing parameters. Please provide "x" and "z".'}), 400
    x, z = data['x'], data['z']
    msg = {"command": "move", "target": {"x": x, "y": 0, "z": z}}
    if not broadcast(msg):
        return jsonify({'error': 'No connected simulators.'}), 400
    return jsonify({'status': 'move command sent', 'command': msg})

@app.route('/move_rel', methods=['POST'])
def move_rel():
    data = request.get_json()
    if not data or 'turn' not in data or 'distance' not in data:
        return jsonify({'error': 'Missing parameters. Please provide "turn" and "distance".'}), 400
    msg = {"command": "move_relative", "turn": data['turn'], "distance": data['distance']}
    if not broadcast(msg):
        return jsonify({'error': 'No connected simulators.'}), 400
    return jsonify({'status': 'move relative command sent', 'command': msg})

@app.route('/stop', methods=['POST'])
def stop():
    msg = {"command": "stop"}
    if not broadcast(msg):
        return jsonify({'error': 'No connected simulators.'}), 400
    return jsonify({'status': 'stop command sent', 'command': msg})

@app.route('/capture', methods=['POST'])
def capture():
    msg = {"command": "capture_image"}
    if not broadcast(msg):
        return jsonify({'error': 'No connected simulators.'}), 400
    return jsonify({'status': 'capture command sent', 'command': msg})

# ---------------------------
# Goal + Obstacles (from your previous step)
# ---------------------------
@app.route('/goal', methods=['POST'])
def set_goal():
    data = request.get_json() or {}
    if 'corner' in data:
        pos = corner_to_coords(str(data['corner']))
    elif 'x' in data and 'z' in data:
        pos = {"x": float(data['x']), "y": float(data.get('y', 0)), "z": float(data['z'])}
    else:
        return jsonify({'error': 'Provide {"corner":"NE|NW|SE|SW"} OR {"x":..,"z":..}'}), 400

    msg = {"command": "set_goal", "position": pos}
    if not broadcast(msg):
        return jsonify({'error': 'No connected simulators.'}), 400
    return jsonify({'status': 'goal set', 'goal': pos})

@app.route('/obstacles/positions', methods=['POST'])
def set_obstacle_positions():
    data = request.get_json() or {}
    positions = data.get('positions')
    if not isinstance(positions, list) or not positions:
        return jsonify({'error': 'Provide "positions" as a non-empty list.'}), 400

    norm = []
    for p in positions:
        if not isinstance(p, dict) or 'x' not in p or 'z' not in p:
            return jsonify({'error': 'Each position needs "x" and "z".'}), 400
        norm.append({"x": float(p['x']), "y": float(p.get('y', 2)), "z": float(p['z'])})

    msg = {"command": "set_obstacles", "positions": norm}
    if not broadcast(msg):
        return jsonify({'error': 'No connected simulators.'}), 400
    return jsonify({'status': 'obstacles updated', 'count': len(norm)})

@app.route('/obstacles/motion', methods=['POST'])
def set_obstacle_motion():
    data = request.get_json() or {}
    if 'enabled' not in data:
        return jsonify({'error': 'Missing "enabled" boolean.'}), 400

    msg = {
        "command": "set_obstacle_motion",
        "enabled": bool(data['enabled']),
        "speed": float(data.get('speed', 0.05)),
        "velocities": data.get('velocities'),
        "bounds": data.get('bounds', {"minX": -45, "maxX": 45, "minZ": -45, "maxZ": 45}),
        "bounce": bool(data.get('bounce', True)),
    }
    if not broadcast(msg):
        return jsonify({'error': 'No connected simulators.'}), 400
    return jsonify({'status': 'obstacle motion updated', 'config': msg})

# ---------------------------
# NEW: Collisions & Reset
# ---------------------------
@app.route('/collisions', methods=['GET'])
def get_collisions():
    """Return the total number of collisions seen (from simulator messages)."""
    return jsonify({'count': collision_count})

@app.route('/reset', methods=['POST'])
def reset():
    """Reset collision count and broadcast a reset command to the simulator."""
    global collision_count, goal_reached_flag, goal_reached_position
    collision_count = 0
    goal_reached_flag = False
    goal_reached_position = None
    if goal_reached_event is not None:
        try:
            goal_reached_event.clear()
        except Exception:
            pass
    if not broadcast({"command": "reset"}):
        # Even if no simulator is connected, we consider the counter reset.
        return jsonify({'status': 'reset done (no simulators connected)', 'collisions': collision_count})
    return jsonify({'status': 'reset broadcast', 'collisions': collision_count})

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        'collisions': collision_count,
        'goal_reached': goal_reached_flag,
        'goal_position': goal_reached_position,
    })

# ---------------------------
# Flask Thread
# ---------------------------
def start_flask():
    app.run(port=5000)

# ---------------------------
# Main Async for WebSocket
# ---------------------------
async def main():
    global async_loop
    async_loop = asyncio.get_running_loop()
    ws_server = await websockets.serve(ws_handler, WS_HOST, WS_PORT, process_request=http_fallback)
    print(f"WebSocket server started on ws://{WS_HOST}:{WS_PORT}")

    # Init queues and start autonomous controller
    global image_queue, goal_reached_event
    image_queue = asyncio.Queue(maxsize=2)
    goal_reached_event = asyncio.Event()
    if autonomous_enabled:
        asyncio.create_task(autonomous_controller())

    await ws_server.wait_closed()

# ---------------------------
# Entry
# ---------------------------
if __name__ == "__main__":
    flask_thread = threading.Thread(target=start_flask, daemon=True)
    flask_thread.start()
    asyncio.run(main())
