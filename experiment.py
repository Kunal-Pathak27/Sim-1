import os
import time
import json
import random
import argparse
import requests


API = os.environ.get("SIM_API", "http://127.0.0.1:5000")


def post(path, payload=None):
    url = f"{API}{path}"
    r = requests.post(url, json=payload or {})
    r.raise_for_status()
    return r.json()


def get(path):
    url = f"{API}{path}"
    r = requests.get(url)
    r.raise_for_status()
    return r.json()


def wait_until_goal(max_seconds=120):
    start = time.time()
    while time.time() - start < max_seconds:
        s = get("/status")
        if s.get("goal_reached"):
            return s
        time.sleep(0.5)
    return None


def run_trial(corner: str, moving: bool = False, speed: float = 0.06):
    post("/reset")
    # set goal by corner
    post("/goal", {"corner": corner})
    if moving:
        # enable moving obstacles
        post("/obstacles/motion", {"enabled": True, "speed": speed})
    else:
        # disable motion
        post("/obstacles/motion", {"enabled": False})

    # let autonomous controller drive; poll status
    result = wait_until_goal()
    s = get("/status")
    collisions = s.get("collisions", 0)
    return {
        "corner": corner,
        "moving": moving,
        "speed": speed if moving else 0.0,
        "goal_reached": bool(result),
        "collisions": collisions,
        "status": s,
    }


def run_level1():
    corners = ["NE", "NW", "SE", "SW"]
    results = []
    for c in corners:
        print(f"[L1] Corner {c}")
        res = run_trial(corner=c, moving=False)
        print(res)
        results.append(res)
    avg = sum(r["collisions"] for r in results) / len(results)
    return {"results": results, "average_collisions": avg}


def run_level2():
    corners = ["NE", "NW", "SE", "SW"]
    results = []
    for c in corners:
        print(f"[L2] Corner {c}")
        res = run_trial(corner=c, moving=True, speed=0.06)
        print(res)
        results.append(res)
    avg = sum(r["collisions"] for r in results) / len(results)
    return {"results": results, "average_collisions": avg}


def run_level3(speeds=None):
    if speeds is None:
        speeds = [0.02, 0.04, 0.06, 0.08, 0.10]
    dataset = []
    for sp in speeds:
        corners = ["NE", "NW", "SE", "SW"]
        collisions = []
        for c in corners:
            print(f"[L3] speed={sp} corner={c}")
            res = run_trial(corner=c, moving=True, speed=sp)
            print(res)
            collisions.append(res["collisions"])
        avg = sum(collisions) / len(collisions)
        dataset.append({"speed": sp, "avg_collisions": avg})
    return dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("level", choices=["1", "2", "3"])
    args = ap.parse_args()

    if args.level == "1":
        out = run_level1()
        print(json.dumps(out, indent=2))
    elif args.level == "2":
        out = run_level2()
        print(json.dumps(out, indent=2))
    else:
        out = run_level3()
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()


