import os
import time

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from dask.distributed import Client
from tqdm import tqdm, trange
from asyncio.exceptions import CancelledError

from ribs.archives import CVTArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler
from ribs.visualize import cvt_archive_3d_plot

from simulate import simulate

EPOCHS = 1000
WORKERS = 10
EMITTERS = 5
ENV_SEED = 321

CELLS = 500
QD_OFFSET = -600

NUM_VIDEOS = 20

GRAVITY = -10.0
WIND = True
WIND_POWER = 20.0
TURBULENCE = 1.5

video_dir = ".videos"

env = gym.make("LunarLander-v3", render_mode="rgb_array", gravity=GRAVITY, enable_wind=WIND, wind_power=WIND_POWER, turbulence_power=TURBULENCE)

def main():
    # error handling for worker count
    workers =min(WORKERS, os.cpu_count() - 1)
    # enable parallel processing
    client = Client(n_workers=workers, threads_per_worker=1)
    client.wait_for_workers(workers)
    # set up env
    reference_env = gym.make("LunarLander-v3")
    # get action space dimensions
    action_dim = reference_env.action_space.n
    # get observation space dimensions
    obs_dim = reference_env.observation_space.shape[0]
    # initialise reference solution
    solution = np.zeros((action_dim, obs_dim))

    # create archive
    archive = CVTArchive(
        solution_dim=solution.size,
        cells=CELLS,
        ranges=[(-3.0, 0.0), (-1.0, 1.0), (-1.0, 1.0),],
        qd_score_offset=QD_OFFSET
    )

    # ribs emitters
    emitters = []
    for _ in range(EMITTERS):
        emitters.append(
            EvolutionStrategyEmitter(
                archive=archive,
                x0=solution.flatten(),
                sigma0=1.0,
                ranker="2imp",
                batch_size=30,
            )
        )

    # ribs scheduler
    scheduler = Scheduler(archive, emitters)

    # begin training loop
    start_time = time.time()
    for e in trange(1, EPOCHS + 1, desc='Epochs'):
        # generate solutions
        solutions = scheduler.ask()
        # evaluate solutions
        futures = client.map(simulate_wrapper, solutions)
        results = client.gather(futures)
        objectives, measures = [], []
        for obj, impact_y_vel, impact_x_pos, impact_x_vel in results:
            objectives.append(obj)
            measures.append([impact_y_vel, impact_x_pos, impact_x_vel])
        # update/insert solutions in archive
        scheduler.tell(objectives, measures)
        if e % 25 == 0 or e == EPOCHS:
            tqdm.write(f"> {e} epochs completed after {time.time() - start_time:.2f}s")
            tqdm.write(f"  - Size: {archive.stats.num_elites}")
            tqdm.write(f"  - Coverage: {archive.stats.coverage}")
            tqdm.write(f"  - QD Score: {archive.stats.qd_score}")
            tqdm.write(f"  - Max Obj: {archive.stats.obj_max}")
            tqdm.write(f"  - Mean Obj: {archive.stats.obj_mean}")

    plt.figure(figsize=(8, 6))
    cvt_archive_3d_plot(archive, vmin=-300, vmax=300, cell_alpha=0.1)
    plt.ylabel("Impact x-speed")
    plt.xlabel("Impact y-velocity")

    occupied = list(archive.data("index"))
    if len(occupied) == 0:
        print("No elites in archive yet — can't save videos.")
        return
    
    # Sample solutions to be recorded
    elites = archive.sample_elites(NUM_VIDEOS)
    solutions = elites["solution"] 
    best_elite = archive.best_elite
    best_solution = best_elite["solution"]
    
    os.makedirs(video_dir, exist_ok=True)

    num_videos = NUM_VIDEOS

    # Record random samples
    print(f"\nSaving {num_videos} elite videos to {video_dir}/")
    for i in range(1, num_videos + 1):
        print(f"  - Recording elite {i}")
        simulate(solutions[i-1], seed=ENV_SEED, env=env, video_dir=video_dir, episode_id=f"random_{i}")
    # Record best performing elite
    print(f"  - Recording best elite")
    simulate(best_solution, seed=ENV_SEED, env=env, video_dir=video_dir, episode_id="best")

    plt.title("Grid Archive")
    plt.savefig("archive")


def simulate_wrapper(model):
    return simulate(model, seed=ENV_SEED, env=env)


if __name__ == '__main__':
    main()
