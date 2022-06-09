from __future__ import annotations

from math import ceil

import psutil
import torch
from pelutils import TT, log, thousands_seperators

from deepspeedcube import device, tensor_size
from deepspeedcube.envs import BaseEnvironment


def gen_new_states(env: BaseEnvironment, num_states: int, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    states_per_depth = ceil(num_states / K)

    with TT.profile("Create states"):
        states = env.get_multiple_solved(states_per_depth*K)
        scramble_depths = torch.zeros(len(states), dtype=torch.int16, device=device)

    with TT.profile("Scramble states"):
        for i in range(K):
            start = i * states_per_depth
            n = len(states) - start
            actions = torch.randint(
                0, len(env.action_space), (n,),
                dtype=torch.uint8,
                device=device,
            )
            env.multiple_moves(actions, states[start:], inplace=True)
            scramble_depths[start:] += 1

    with TT.profile("Shuffle states"):
        shuffle_index = torch.randperm(len(states), device=device)
        states[:] = states[shuffle_index]
        scramble_depths[:] = scramble_depths[shuffle_index]

    return states, scramble_depths

def get_batches_per_gen(env: BaseEnvironment, batch_size: int) -> int:
    max_gen_states = 250 * 10 ** 6
    max_memory_frac = 0.8

    # Calculate memory requirements for scrambling
    state_memory           = tensor_size(env.get_solved())
    scramble_depths_memory = 2  # int16
    actions_memory         = 1  # uint8
    shuffle_index_memory   = 8  # int64
    scramble_memory        = state_memory + scramble_depths_memory\
        + actions_memory + shuffle_index_memory

    # Calculate memory requirements for getting neighbour states
    state_memory     = tensor_size(env.get_solved()) * len(env.action_space)
    actions_memory   = tensor_size(env.action_space)
    neighbour_memory = state_memory + actions_memory

    total_batch_memory = batch_size * (scramble_memory + neighbour_memory)
    log.debug(
        "Memory requirements for generating states for a batch of size %i:" % batch_size,
        thousands_seperators(total_batch_memory),
    )

    if torch.cuda.is_available():
        avail_mem = torch.cuda.get_device_properties(device).total_memory
    else:
        avail_mem = psutil.virtual_memory().total
    avail_mem *= max_memory_frac

    num_batches = avail_mem // total_batch_memory
    if num_batches * batch_size * (1 + len(env.action_space)) > max_gen_states:
        num_batches = max_gen_states // (batch_size * (1 + len(env.action_space)))

    return num_batches
