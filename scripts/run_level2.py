import os
import time
import statistics

from agent.sim_api import SimAPI
from agent.agent import Navigator


def ensure_dirs():
    os.makedirs("results/videos", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/graphs", exist_ok=True)


def main():
    ensure_dirs()
    corners = ["NE", "NW", "SE", "SW"]
    api = SimAPI()
    speed = 0.08

    results = []
    for c in corners:
        ts = int(time.time())
        video = f"results/videos/level2_{c}_s{str(speed).replace('.', '_')}_{ts}.mp4"
        nav = Navigator(api, corner=c, moving=True, speed=speed)
        col, steps = nav.run(video_path=video)
        print(f"{c} @speed {speed}: collisions={col}, steps={steps}")
        results.append(col)

    print("Per-run collisions:", results)
    print("Average collisions:", statistics.mean(results))


if __name__ == "__main__":
    main()


