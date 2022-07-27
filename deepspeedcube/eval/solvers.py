from __future__ import annotations

import abc
import ctypes
import math

import numpy as np
import torch
from pelutils import TickTock, TT

from deepspeedcube import device, LIBDSC, ptr
from deepspeedcube.envs import Environment
from deepspeedcube.min_heap import MinHeap
from deepspeedcube.model import Model

from pelutils import log


class Solver(abc.ABC):

    def __init__(self, env: Environment, max_time: float | None):
        self.env = env
        self.max_time = max_time or 1e6
        self.tt = TickTock()

    @abc.abstractmethod
    def solve(self, state: torch.Tensor) -> tuple[torch.Tensor | None, float | None]:
        """ Returns tensor of actions to solve the given state and the time taken. """
        pass

class GreedyValueSolver(Solver):

    def __init__(self, env: Environment, max_time: float | None, models: list[Model]):
        super().__init__(env, max_time)
        self.models = models
        for model in self.models:
            model.eval()

    @torch.no_grad()
    def solve(self, state: torch.Tensor) -> tuple[torch.Tensor | None, float | None]:
        self.tt.tick()

        actions = list()
        while self.tt.tock() < self.max_time:
            TT.profile("Iteration")

            neighbours = self.env.neighbours(state.unsqueeze(dim=0))
            solved = self.env.multiple_is_solved(neighbours)
            if torch.any(solved):
                actions.append(torch.where(solved)[0].item())
                return torch.tensor(actions), self.tt.tock()

            neighbours_oh = self.env.multiple_oh(neighbours)
            preds = torch.zeros(len(neighbours_oh), device=device)
            for model in self.models:
                preds += model(neighbours_oh).squeeze()

            action = preds.argmin().item()
            actions.append(action)

            state = self.env.move(action, state)

            TT.end_profile()

        return None, None

class AStar(Solver):

    def __init__(self, env: Environment, max_time: float | None, models: list[Model], l: float, N: int, d: int):
        super().__init__(env, max_time)
        self.models = models
        self.l = l  # lambda used to weigh moves spent and estimated cost-to-go
        self.N = N  # Number of states to expand each iteration
        self.d = d  # Depth expansion

        self.solved_state = self.env.get_solved().cpu()

        for model in self.models:
            model.eval()

    @torch.no_grad()
    def solve(self, state: torch.Tensor) -> tuple[torch.Tensor | None, float | None]:
        self.tt.tick()

        state = state.cpu()

        if self.env.is_solved(state):
            return torch.tensor([], dtype=torch.long), self.tt.tock()

        state_map_p = ctypes.c_void_p(LIBDSC.astar_init_state_map(
            ctypes.c_float(self.l),
            self.env.state_size,
        ))

        # Perform BFS as long as the frontier size is less than or equal to N
        with TT.profile("BFS"):

            bfs_iters = math.floor(math.log(self.N, len(self.env.action_space)))
            num_bfs_states = [0, *(len(self.env.action_space) ** np.arange(bfs_iters+1)).cumsum()]
            log("Doing %i BFS iters to explore %i states" % (bfs_iters, num_bfs_states[-1]))

            bfs_states = torch.empty((num_bfs_states[-1], *self.env.state_shape), dtype=self.env.dtype)
            bfs_g = torch.empty(num_bfs_states[-1], dtype=torch.int)
            bfs_back_actions = torch.empty(num_bfs_states[-1], dtype=torch.uint8)

            bfs_states[0] = state
            bfs_g[0] = 0
            bfs_back_actions[0] = 255

            longest_path_length = torch.tensor([bfs_iters+1])

            for i in range(bfs_iters):
                TT.profile("Iteration")
                start, mid, end = num_bfs_states[i:i+3]

                bfs_states[mid:end] = self.env.neighbours(bfs_states[start:mid])
                bfs_g[mid:end] = i + 1
                bfs_back_actions[mid:end] = self.env.reverse_moves(
                    self.env.action_space.repeat(mid-start)
                )

                TT.end_profile()

            with TT.profile("Insert BFS nodes"):
                LIBDSC.astar_insert_bfs_states(
                    state_map_p,
                    len(bfs_states),
                    ptr(bfs_states),
                    ptr(bfs_g),
                    ptr(bfs_back_actions),
                )

            with TT.profile("Solution check"):
                sol = self.env.multiple_is_solved(bfs_states)
                sol = torch.where(sol)[0]
                if len(sol):
                    back_actions = torch.empty(bfs_iters, dtype=torch.uint8)
                    log("BFS found solution")
                    LIBDSC.astar_get_back_actions(
                        self.env.state_size,
                        ptr(self.solved_state),
                        ptr(back_actions),
                        state_map_p,
                        LIBDSC.cube_multi_act,
                    )

                    return self.env.reverse_moves(back_actions), self.tt.tock()

        with TT.profile("A*"):

            frontier = MinHeap(self.env.state_shape, self.env.dtype)
            back_actions = self.env.reverse_moves(
                self.env.action_space.repeat(self.N)
            )
            from_index = torch.arange(self.N).repeat_interleave(len(self.env.action_space))
            LIBDSC.astar_insert_bfs_states(
                state_map_p,
                len(bfs_states),
                ptr(bfs_states),
                ptr(bfs_g),
                ptr(bfs_back_actions)
            )

            first_iter = True
            found_solution = False

            while self.tt.tock() < self.max_time:
                TT.profile("Iteration")
                log("ITERATION %.2f ms" % (self.tt.tock() * 1000))

                if first_iter:
                    states_to_expand = bfs_states
                else:
                    _, states_to_expand = frontier.extract_min_multiple(self.N)

                neighbour_states = self.env.neighbours(states_to_expand)

                if torch.any(self.env.is_solved(neighbour_states)):
                    found_solution = True
                    break

                h = self.cost_to_go(neighbour_states)

                # Expand heap to make sure there is enough room to accomodate new states
                max_frontier_size = len(frontier) + len(self.env.action_space) * self.N
                while len(frontier._keys) < max_frontier_size:
                    frontier._expand_heap()

                with TT.profile("Update search state"):
                    frontier._num_elems = LIBDSC.astar_update_search_state(
                        len(states_to_expand),
                        ptr(states_to_expand),
                        len(neighbour_states),
                        ptr(neighbour_states),
                        ptr(h),
                        ptr(back_actions),
                        ptr(longest_path_length),
                        ptr(from_index),

                        state_map_p,
                        frontier._heap_ptr,
                    )

                first_iter = False

                TT.end_profile()

                print(self.tt.tock())

        if found_solution:
            TT.profile("Get action path")
            back_actions = torch.empty(longest_path_length[0], dtype=torch.uint8)
            LIBDSC.astar_get_back_actions(
                self.env.state_size,
                ptr(self.solved_state),
                ptr(back_actions),
                state_map_p,
                LIBDSC.cube_multi_act,
            )
            TT.end_profile()
        else:
            back_actions = None

        LIBDSC.astar_free_state_map(state_map_p)
        return None, self.tt.tock()

    @torch.no_grad()
    def cost_to_go(self, states: torch.Tensor) -> torch.Tensor:
        with TT.profile("To device"):
            states_d = states.to(device)
        with TT.profile("One-hot"):
            states_oh = self.env.multiple_oh(states_d)

        preds = torch.zeros(len(states), dtype=torch.float, device=device)
        with TT.profile("Estimate cost-to-go"):
            for model in self.models:
                preds += model(states_oh).squeeze()

        return (preds / len(self.models)).cpu()
