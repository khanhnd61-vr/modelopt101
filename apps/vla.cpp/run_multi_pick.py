"""Pick several libero_object objects into the basket in ONE scene, one instruction at a time.

The stock client resets the scene after every success (LiberoEnv.step calls reset()), so each run
picks exactly one object. This script keeps one scene and chains instructions:

- no reset on success: it steps the inner LIBERO env itself, and turns off robosuite's 1000-step
  horizon (ignore_done), which would otherwise end the episode during object 3-4;
- one training instruction per object, and client.reset() on every switch so the old
  instruction's queued action chunk is dropped;
- per-object success from LIBERO's own In(<obj>, basket_1_contain_region) predicate;
- return to home after each object: SmolVLA only saw episodes starting from that pose;
- step budget + retry per object, so one failed grasp does not use up the rest of the run.

Needs a running vla-server; run through run_local.sh:  bash run_local.sh multi [--objects ...]
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(os.environ.get("VLA_CPP", Path.home() / "vla.cpp")) / "eval"))

import gymnasium as gym
import numpy as np

import sim.libero  # noqa: F401  registers the gymnasium envs
from libero.libero import benchmark
from robosuite.utils.control_utils import orientation_error
from client.vla_cpp_client import VlaCppClient
from client.adapters import LeRobotPipelineAdapter

p = argparse.ArgumentParser()
p.add_argument("--task-id", type=int, default=0, help="libero_object task whose scene is used")
p.add_argument("--objects", nargs="*", default=None,
               help="pick order, scene names like milk_1; default: the task's target, then the rest")
p.add_argument("--n-objects", type=int, default=4)
p.add_argument("--steps-per-object", type=int, default=400)
p.add_argument("--retries", type=int, default=1)
p.add_argument("--n-action-steps", type=int, default=10)
p.add_argument("--home-gain", type=float, default=0.5, help="P gain of the scripted return-to-home move")
p.add_argument("--output-dir", default=str(Path.home() / "libero_outputs" / "multi_pick"))
p.add_argument("--seed", type=int, default=42)
args = p.parse_args()

# the exact training instruction per object: "pick up the milk and place it in the basket" -> milk_1
instr = {}
for t in benchmark.get_benchmark_dict()["libero_object"]().tasks:
    name = t.language.removeprefix("pick up the ").removesuffix(" and place it in the basket")
    instr[name.replace(" ", "_") + "_1"] = t.language

env = gym.make(f"libero_object/task_{args.task_id}", seed=args.seed, video_fps=30,
               output_video_dir=Path(args.output_dir) / f"task_{args.task_id}").unwrapped
client = LeRobotPipelineAdapter(client=VlaCppClient(
    arch="smolvla", n_action_steps=args.n_action_steps, recv_timeout_ms=120_000))

obs, _ = env.reset()
sim = env._env.env
sim.ignore_done = True  # robosuite's 1000-step horizon would kill object 3-4 mid-episode
home = obs["robot_state"]["eef"]["pos"].copy()
home_mat = obs["robot_state"]["eef"]["mat"].copy()
goal = sim.parsed_problem["goal_state"][0]  # e.g. ['in', 'alphabet_soup_1', 'basket_1_contain_region']
scene = [o for o in sim.objects_dict if o in instr]
assert goal[1] in scene, f"instruction map broken: {goal[1]} not in {scene}"
order = args.objects or [goal[1]] + [o for o in scene if o != goal[1]]
bad = [o for o in order if o not in scene]
assert not bad, f"{bad} not in this scene, pick from {scene}"
order = order[:args.n_objects]
print("scene objects:", scene, "\npick order:", order, flush=True)


def in_basket(o):
    return sim._eval_predicate([goal[0], o, goal[2]])


def up_axis(o):
    # the object's own z axis in world coordinates (third column of its rotation matrix)
    w, x, y, z = sim.sim.data.body_xquat[sim.obj_body_id[o]]
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


start_up = {o: up_axis(o) for o in scene}


def knocked_over(o):
    # tilted more than ~45 degrees from how it started: SmolVLA never saw it lying like that
    return start_up[o] @ up_axis(o) < 0.7


def step(action):
    # LiberoEnv.step resets the whole scene on success, so drive the inner env directly
    raw, _, _, _ = env._env.step(action)
    o = env._format_raw_obs(raw)
    env._append_video_frame(o)
    return o


def go_home(obs):
    # gripper open, lift to home height first (do not sweep the basket), then move back over home.
    # Also turn the wrist back to its start orientation: every training episode starts from it, and
    # the policy may have rotated the wrist for the previous object.
    for lift_only in (True, False):
        for _ in range(80):
            pos = obs["robot_state"]["eef"]["pos"]
            err = (np.r_[pos[:2], home[2]] if lift_only else home) - pos
            rot = orientation_error(home_mat, obs["robot_state"]["eef"]["mat"])
            if np.linalg.norm(err) < 0.01 and np.linalg.norm(rot) < 0.05:
                break
            # OSC maps action 1.0 to 5 cm / 0.5 rad, so both terms close home_gain of the gap per step
            obs = step(np.r_[np.clip(err / 0.05 * args.home_gain, -1, 1),
                             np.clip(rot / 0.5 * args.home_gain, -1, 1), -1])
    return obs


results = []
for o in order:
    env.task_description = obs["task_description"] = instr[o]
    ok, used = False, 0
    for attempt in range(1 + args.retries):
        if knocked_over(o):
            print(f"{o} is knocked over, skipping", flush=True)
            break
        client.reset()  # drop the queued chunk from the previous instruction
        for _ in range(args.steps_per_object):
            obs = step(client.get_action(obs))
            used += 1
            if in_basket(o):
                ok = True
                break
        obs = go_home(obs)
        if ok:
            break
    results.append((o, ok, used, attempt))
    print(f"{o:<20} {'OK' if ok else 'FAIL':<4} steps={used} retries={attempt}", flush=True)

kept = sum(in_basket(o) for o in order)
print(f"*** placed {sum(r[1] for r in results)}/{len(order)}, still in basket at the end: {kept}/{len(order)}")
env.close()
print("video:", Path(args.output_dir) / f"task_{args.task_id}" / "episode_000000.mp4")
