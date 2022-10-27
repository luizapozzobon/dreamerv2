import argparse
import os

import numpy as np

np.random.seed(42)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "host_name",
        type=str,
    )
    parser.add_argument(
        "num_levels",
        type=int,
        help="Number of levels to train on",
    )

    parser.add_argument(
        "--env",
        type=str,
        default="atari_riverraid",
        help="Name of the gym environment",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=32,
        help="Number of environments",
    )
    parser.add_argument(
        "--n-timesteps",
        type=int,
        default=50_000_000,
        help="Timesteps to train for",
    )
    return parser.parse_args()

def main(
    host_name:str,
    num_levels: int,
    env: str,
    n_envs: int,
    n_timesteps: int,
) -> None:
    run_cmd = """python dreamerv2/train.py \
        --logdir {path} \
        --configs atari \
        --task {env} \
        --steps {n_timesteps} \
        --envs {n_envs} \
        --env-kwargs enable_rom_patches:True,rom_patches_args:'"s"',num_levels:{num_levels},levels_set:0,game_seed:{game_seed},seed:{game_seed} \
        --eval-env-kwargs enable_rom_patches:True,rom_patches_args:'"s"',num_levels:100000,levels_set:1,game_seed:{game_seed_eval},seed:{game_seed_eval}"""

    os.environ["WANDB_HOST"] = host_name

    # 1 seed for each `num_levels`
    game_seeds = np.random.randint(0, 100_000, size=1)
    for game_seed in game_seeds:
        log_path = f"logdir/dreamerv2/{env}/{game_seed}"
        cmd = run_cmd.format(
           path=log_path,
           env=env,
           n_timesteps=n_timesteps,
           n_envs=n_envs,
           num_levels=num_levels,
           game_seed=game_seed,
           game_seed_eval=game_seed,
        )
        print(cmd)
        os.system(cmd)

if __name__ == "__main__":
    args = parse_args()
    main(**vars(args))


