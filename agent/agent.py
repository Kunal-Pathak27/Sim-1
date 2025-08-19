import time
import cv2
from typing import Optional, Tuple

from agent.sim_api import SimAPI
from agent.vision import obstacle_mask, sample_headings, annotate_debug
from agent.planner import pick_heading, step_from_heading, goal_bearing_from_position, close_to_goal


class NavigatorConfig:
    STEP_DIST = 4.0
    SAFE_THR = 0.35
    OBS_W = 1.2
    GOAL_W = 0.6
    FOV_DEG = 120
    NUM_HEADINGS = 31
    MAX_STEPS = 1200
    GOAL_RADIUS = 3.0
    SAVE_DEBUG = True


class Navigator:
    def __init__(self, api: SimAPI, corner: str = "NE", moving: bool = False, speed: float = 0.0):
        self.api = api
        self.corner = corner
        self.moving = moving
        self.speed = speed
        self.goal = None

    def set_goal(self) -> None:
        resp = self.api.set_goal_corner(self.corner)
        g = resp.get("goal") or resp.get("position")
        if g is None:
            mapping = {
                "NE": "TR",
                "NW": "TL",
                "SE": "BR",
                "SW": "BL",
                "TR": "TR",
                "TL": "TL",
                "BR": "BR",
                "BL": "BL",
            }
            alias = mapping.get(self.corner, "NE")
            corners = {
                "TL": {"x": -45, "y": 0, "z": 45},
                "TR": {"x": 45, "y": 0, "z": 45},
                "BL": {"x": -45, "y": 0, "z": -45},
                "BR": {"x": 45, "y": 0, "z": -45},
                "NE": {"x": 45, "y": 0, "z": 45},
                "NW": {"x": -45, "y": 0, "z": 45},
                "SE": {"x": 45, "y": 0, "z": -45},
                "SW": {"x": -45, "y": 0, "z": -45},
            }
            g = corners[alias]
        self.goal = {"x": float(g["x"]), "y": float(g.get("y", 0)), "z": float(g["z"])}

    def enable_obstacles_motion(self) -> None:
        if self.moving:
            self.api.set_obstacle_motion(
                True,
                speed=self.speed,
                bounds={"minX": -45, "maxX": 45, "minZ": -45, "maxZ": 45},
                bounce=True,
            )

    def run(self, video_path: Optional[str] = None) -> Tuple[int, int]:
        self.api.reset()
        self.set_goal()
        self.enable_obstacles_motion()

        collisions0 = self.api.collisions()
        coll_prev = collisions0
        total_collisions = 0

        writer = None
        try:
            if video_path and NavigatorConfig.SAVE_DEBUG:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(video_path, fourcc, 10.0, (640, 480))

            steps = 0
            while steps < NavigatorConfig.MAX_STEPS:
                img, pos, ts = self.api.capture()
                if img is None or pos is None:
                    time.sleep(0.05)
                    continue

                # Vision
                mask = obstacle_mask(img)
                headings = sample_headings(mask, NavigatorConfig.NUM_HEADINGS, NavigatorConfig.FOV_DEG)

                # Goal bearing
                goal_bearing = goal_bearing_from_position(pos, self.goal)

                # Pick
                ang, _score = pick_heading(
                    headings,
                    goal_bearing,
                    obs_w=NavigatorConfig.OBS_W,
                    goal_w=NavigatorConfig.GOAL_W,
                    safe_thr=NavigatorConfig.SAFE_THR,
                )
                step = step_from_heading(ang, NavigatorConfig.STEP_DIST)
                self.api.move_rel(step["turn"], step["distance"])

                # Collisions
                c = self.api.collisions()
                if c > coll_prev:
                    total_collisions += (c - coll_prev)
                    coll_prev = c
                    # Back off slightly
                    self.api.move_rel(0, -NavigatorConfig.STEP_DIST * 0.6)

                # Goal check
                if close_to_goal(pos, self.goal, radius=NavigatorConfig.GOAL_RADIUS):
                    self.api.move_rel(0, 0)
                    break

                # Debug overlay
                if writer:
                    overlay = annotate_debug(
                        img,
                        mask,
                        f"Goal:{self.corner} Steps:{steps} Collisions:{total_collisions}",
                    )
                    # Resize to standard size for writer
                    try:
                        out = cv2.resize(overlay, (640, 480))
                    except Exception:
                        out = overlay
                    writer.write(out)

                steps += 1
                time.sleep(0.05)

            if writer:
                writer.release()

            return total_collisions, steps
        finally:
            if writer:
                writer.release()


