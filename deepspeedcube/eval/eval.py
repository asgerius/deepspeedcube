from __future__ import annotations

from dataclasses import dataclass

import torch
from pelutils import DataStorage, log, TT
from pelutils.parser import JobDescription

from deepspeedcube import device
from deepspeedcube.envs import get_env
from deepspeedcube.envs.gen_states import gen_eval_states
from deepspeedcube.eval.solver import AStar, GreedyValueSolver
from deepspeedcube.model import ModelConfig, Model
from deepspeedcube.train.train import TrainConfig


@dataclass
class EvalConfig(DataStorage, json_name="eval_cfg.json", indent=4):
    solver:           str
    depths:           list[int]
    states_per_depth: int
    max_time:         float
    astar_lambda:     float
    astar_N:          int
    astar_d:          int

@dataclass
class EvalResults(DataStorage, json_name="eval_results.json", indent=4):
    solved:           list[int]
    solve_times:      list[list[float]]

@torch.no_grad()
def eval(job: JobDescription):
    log.section("Loading configurations")
    eval_cfg = EvalConfig(
        solver           = job.solver,
        depths           = list(range(job.max_depth+1)),
        states_per_depth = job.states_per_depth,
        max_time         = job.max_time,
        astar_lambda     = job.astar_lambda,
        astar_N          = job.astar_N,
        astar_d          = job.astar_d,
    )
    log("Got eval config", eval_cfg)

    train_cfg = TrainConfig.load(f"{job.location}/..")
    log("Got train config", train_cfg)

    model_cfg = ModelConfig.load(f"{job.location}/..")
    log("Got model config", model_cfg)

    log("Loading environment %s" % train_cfg.env)
    env = get_env(train_cfg.env)

    log.section("Loading %i models" % train_cfg.num_models)
    models = list()
    for i in range(train_cfg.num_models):
        TT.profile("Load model")
        log(f"Loading model {i+1} / {train_cfg.num_models}")
        model = Model(model_cfg).to(device)
        model.load_state_dict(torch.load(
            f"{job.location}/../model-{i}.pt",
            map_location=device,
        ))
        model.eval()
        models.append(model)
        TT.end_profile()

        if i == 0:
            log(model)

    log.section("Preparing solver")
    if eval_cfg.solver == "GreedyValueSolver":
        solver = GreedyValueSolver(env, eval_cfg.max_time, models)
    elif eval_cfg.solver == "AStar":
        solver = AStar(env, eval_cfg.max_time, models,
            eval_cfg.astar_lambda, eval_cfg.astar_N, eval_cfg.astar_d,
        )
    else:
        raise ValueError("Unknown solver '%s'" % eval_cfg.solver)

    results = EvalResults(
        solved      = list(),
        solve_times = list(),
    )

    log.section("Evaluating")
    states = gen_eval_states(env, eval_cfg.states_per_depth, eval_cfg.depths)
    for i, depth in enumerate(eval_cfg.depths):
        log("Evaluating at depth %i" % depth)
        results.solved.append(0)
        results.solve_times.append(list())
        TT.profile("Evaluate at depth %i" % depth)
        for state in states[i]:
            actions, time = solver.solve(state)
            if actions is not None:
                results.solved[-1] += 1
                results.solve_times[-1].append(time)
        TT.end_profile()

    log.section("Saving")
    eval_cfg.save(job.location)
    results.save(job.location)
