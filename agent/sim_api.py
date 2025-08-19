import time
import requests
import numpy as np
import base64
import cv2
from typing import Optional, Tuple, Dict, Any


class SimAPI:
    def __init__(self, base: str = "http://localhost:5000"):
        self.base = base.rstrip("/")
        self.sess = requests.Session()
        self._last_capture_ts: Optional[int] = None

    def reset(self) -> Dict[str, Any]:
        r = self.sess.post(f"{self.base}/reset", timeout=5)
        r.raise_for_status()
        return r.json()

    def set_goal_corner(self, corner: str) -> Dict[str, Any]:
        r = self.sess.post(f"{self.base}/goal", json={"corner": corner}, timeout=5)
        r.raise_for_status()
        return r.json()

    def set_goal_coords(self, x: float, z: float, y: float = 0) -> Dict[str, Any]:
        r = self.sess.post(
            f"{self.base}/goal",
            json={"x": float(x), "y": float(y), "z": float(z)},
            timeout=5,
        )
        r.raise_for_status()
        return r.json()

    def move_rel(self, turn_deg: float, distance: float) -> Dict[str, Any]:
        r = self.sess.post(
            f"{self.base}/move_rel",
            json={"turn": float(turn_deg), "distance": float(distance)},
            timeout=5,
        )
        r.raise_for_status()
        return r.json()

    def move_abs(self, x: float, z: float, y: float = 0) -> Dict[str, Any]:
        r = self.sess.post(
            f"{self.base}/move",
            json={"x": float(x), "y": float(y), "z": float(z)},
            timeout=5,
        )
        r.raise_for_status()
        return r.json()

    def collisions(self) -> int:
        r = self.sess.get(f"{self.base}/collisions", timeout=5)
        r.raise_for_status()
        return int(r.json().get("count", 0))

    def set_obstacle_motion(
        self,
        enabled: bool = True,
        speed: float = 0.05,
        bounds: Optional[Dict[str, float]] = None,
        bounce: bool = True,
        velocities: Optional[list] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"enabled": enabled, "speed": speed, "bounce": bounce}
        if bounds:
            payload["bounds"] = bounds
        if velocities:
            payload["velocities"] = velocities
        r = self.sess.post(f"{self.base}/obstacles/motion", json=payload, timeout=5)
        r.raise_for_status()
        return r.json()

    def _get_last_capture(self) -> Optional[Dict[str, Any]]:
        try:
            r = self.sess.get(f"{self.base}/last_capture", timeout=3)
            if r.status_code != 200:
                return None
            return r.json()
        except requests.RequestException:
            return None

    def capture(self, wait_timeout_s: float = 2.0) -> Tuple[Optional[np.ndarray], Optional[Dict[str, float]], Optional[int]]:
        # Trigger a capture in the simulator
        self.sess.post(f"{self.base}/capture", timeout=5)

        # Poll for a fresh capture
        start = time.time()
        latest: Optional[Dict[str, Any]] = None
        while time.time() - start < wait_timeout_s:
            data = self._get_last_capture()
            if not data:
                time.sleep(0.05)
                continue
            ts = data.get("timestamp")
            if ts is None or (self._last_capture_ts is not None and ts == self._last_capture_ts):
                time.sleep(0.05)
                continue
            latest = data
            break

        if not latest:
            return None, None, None

        # Remember timestamp
        ts_val = latest.get("timestamp")
        if isinstance(ts_val, (int, float)):
            self._last_capture_ts = int(ts_val)

        # Image may be data URL or raw base64
        img_b64_or_url = latest.get("image", "") or ""
        if isinstance(img_b64_or_url, str) and img_b64_or_url.startswith("data:image"):
            # Strip header: data:image/png;base64,
            try:
                header, b64 = img_b64_or_url.split(",", 1)
            except ValueError:
                b64 = ""
        else:
            b64 = img_b64_or_url

        img = None
        if b64:
            img_bytes = base64.b64decode(b64)
            arr = np.frombuffer(img_bytes, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

        pos = latest.get("position")
        return img, pos, self._last_capture_ts


