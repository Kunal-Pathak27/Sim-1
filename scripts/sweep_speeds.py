import os
import time
import statistics
import matplotlib.pyplot as plt

from agent.sim_api import SimAPI
from agent.agent import Navigator


def ensure_dirs():
    os.makedirs("results/videos", exist_ok=True)
    os.makedirs("results/logs", exist_ok=True)
    os.makedirs("results/graphs", exist_ok=True)


def main():
    ensure_dirs()
    speeds = [0.04, 0.06, 0.08, 0.10]
    corners = ["NE", "NW", "SE", "SW"]

    api = SimAPI()
    avgs = []

    for sp in speeds:
        cols = []
        for c in corners:
            ts = int(time.time())
            video = None  # speed up runs; set a path to save visualizations
            nav = Navigator(api, corner=c, moving=True, speed=sp)
            col, steps = nav.run(video_path=video)
            print(f"Speed {sp} corner {c}: collisions={col}, steps={steps}")
            cols.append(col)
        avg = statistics.mean(cols)
        print(f"Speed {sp}: avg collisions={avg}")
        avgs.append(avg)

    plt.figure(figsize=(6, 4))
    plt.plot(speeds, avgs, marker='o')
    plt.xlabel("Obstacle speed")
    plt.ylabel("Average collisions (4 runs)")
    plt.title("Obstacle speed vs average collisions")
    plt.grid(True)
    plt.tight_layout()
    out_path = "results/graphs/speed_vs_collisions.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()


