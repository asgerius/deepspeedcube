from __future__ import annotations

import abc
import ctypes
from typing import Type

import numpy as np
import torch
import torch.nn.functional as F

from deepspeedcube import device, ptr, tensor_size, LIBDSC


if torch.cuda.is_available():
    _CUBELIB_CUDA = ctypes.cdll.LoadLibrary("lib/cube_cuda.so")

class Environment(abc.ABC):

    dtype: type
    action_space: torch.Tensor
    state_shape: torch.Size
    state_oh_size: int
    _solved_state: torch.Tensor
    state_size: int

    @classmethod
    def get_solved(cls) -> torch.Tensor:
        return cls._solved_state.clone()

    @classmethod
    def get_multiple_solved(cls, n: int) -> torch.Tensor:
        return cls._solved_state.repeat((n, *[1]*len(cls._solved_state.shape)))

    @classmethod
    def is_solved(cls, state: torch.Tensor) -> bool:
        return torch.all(state == cls._solved_state)

    @classmethod
    def multiple_is_solved(cls, states: torch.Tensor) -> torch.BoolTensor:
        return (states == cls._solved_state).all(dim=1)

    @abc.abstractclassmethod
    def move(cls, action: int, state: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractclassmethod
    def multiple_moves(cls, actions: torch.Tensor, states: torch.Tensor, inplace=False) -> torch.Tensor:
        pass

    @classmethod
    def neighbours(cls, states: torch.Tensor) -> torch.Tensor:
        neighbour_states = states.repeat_interleave(len(cls.action_space), dim=0)
        actions = cls.action_space.repeat(len(states)).to(states.device)
        cls.multiple_moves(actions, neighbour_states, inplace=True)
        return neighbour_states

    @classmethod
    def oh(cls, state: torch.Tensor) -> torch.Tensor:
        return F.one_hot(
            state.long(),
            num_classes=cls.state_oh_size // len(cls._solved_state),
        ).astype(torch.float32).view(1, -1)

    @classmethod
    def multiple_oh(cls, states: torch.Tensor) -> torch.Tensor:
        return F.one_hot(
            states.long(),
            num_classes=cls.state_oh_size // len(cls._solved_state),
        ).to(torch.float32).view(len(states), -1)

    @abc.abstractclassmethod
    def reverse_move(cls, action: int) -> int:
        pass

    @abc.abstractclassmethod
    def reverse_moves(cls, actions: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractclassmethod
    def string(cls, state: torch.Tensor) -> str:
        pass

class _CubeEnvironment(Environment):

    dtype = torch.uint8
    action_space = torch.arange(12, dtype=torch.uint8)
    state_shape = torch.Size([20])
    state_oh_size = 480
    _solved_state = torch.tensor(
        [0, 3, 6, 9, 12, 15, 18, 21,
         0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22],
        dtype=dtype,
        device=device,
    )
    state_size = tensor_size(_solved_state)

    # If the six sides are represented by an array, the order should be F, B, T, D, L, R
    F, B, T, D, L, R = 0, 1, 2, 3, 4, 5
    corner_633map = (
        ((F, 0, 0), (L, 0, 2), (T, 2, 0)),
        ((F, 2, 0), (D, 0, 0), (L, 2, 2)),
        ((F, 2, 2), (R, 2, 0), (D, 0, 2)),
        ((F, 0, 2), (T, 2, 2), (R, 0, 0)),
        ((B, 0, 2), (T, 0, 0), (L, 0, 0)),
        ((B, 2, 2), (L, 2, 0), (D, 2, 0)),
        ((B, 2, 0), (D, 2, 2), (R, 2, 2)),
        ((B, 0, 0), (R, 0, 2), (T, 0, 2)),
    )
    side_633map = (
        ((F, 0, 1), (T, 2, 1)),
        ((F, 1, 0), (L, 1, 2)),
        ((F, 2, 1), (D, 0, 1)),
        ((F, 1, 2), (R, 1, 0)),
        ((T, 1, 0), (L, 0, 1)),
        ((D, 1, 0), (L, 2, 1)),
        ((D, 1, 2), (R, 2, 1)),
        ((T, 1, 2), (R, 0, 1)),
        ((B, 0, 1), (T, 0, 1)),
        ((B, 1, 2), (L, 1, 0)),
        ((B, 2, 1), (D, 2, 1)),
        ((B, 1, 0), (R, 1, 2)),
    )

    full_action_maps = torch.tensor([
        [  # Corners
            [9, 11, 10, 0, 2, 1, 3, 5, 4, 6, 8, 7, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 15, 17, 16, 18, 20, 19, 21, 23, 22, 12, 14, 13],
            [14, 13, 12, 3, 4, 5, 6, 7, 8, 2, 1, 0, 23, 22, 21, 15, 16, 17, 18, 19, 20, 11, 10, 9],
            [0, 1, 2, 8, 7, 6, 20, 19, 18, 9, 10, 11, 12, 13, 14, 5, 4, 3, 17, 16, 15, 21, 22, 23],
            [4, 3, 5, 16, 15, 17, 6, 7, 8, 9, 10, 11, 1, 0, 2, 13, 12, 14, 18, 19, 20, 21, 22, 23],
            [0, 1, 2, 3, 4, 5, 10, 9, 11, 22, 21, 23, 12, 13, 14, 15, 16, 17, 7, 6, 8, 19, 18, 20],
            [3, 5, 4, 6, 8, 7, 9, 11, 10, 0, 2, 1, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 23, 22, 12, 14, 13, 15, 17, 16, 18, 20, 19],
            [11, 10, 9, 3, 4, 5, 6, 7, 8, 23, 22, 21, 2, 1, 0, 15, 16, 17, 18, 19, 20, 14, 13, 12],
            [0, 1, 2, 17, 16, 15, 5, 4, 3, 9, 10, 11, 12, 13, 14, 20, 19, 18, 8, 7, 6, 21, 22, 23],
            [13, 12, 14, 1, 0, 2, 6, 7, 8, 9, 10, 11, 16, 15, 17, 4, 3, 5, 18, 19, 20, 21, 22, 23],
            [0, 1, 2, 3, 4, 5, 19, 18, 20, 7, 6, 8, 12, 13, 14, 15, 16, 17, 22, 21, 23, 10, 9, 11]
        ],
        [  # Sides
            [6, 7, 0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 18, 19, 20, 21, 22, 23, 16, 17],
            [9, 8, 2, 3, 4, 5, 6, 7, 17, 16, 10, 11, 12, 13, 1, 0, 15, 14, 18, 19, 20, 21, 22, 23],
            [0, 1, 2, 3, 13, 12, 6, 7, 8, 9, 5, 4, 21, 20, 14, 15, 16, 17, 18, 19, 11, 10, 22, 23],
            [0, 1, 10, 11, 4, 5, 6, 7, 2, 3, 18, 19, 12, 13, 14, 15, 16, 17, 8, 9, 20, 21, 22, 23],
            [0, 1, 2, 3, 4, 5, 14, 15, 8, 9, 10, 11, 6, 7, 22, 23, 16, 17, 18, 19, 20, 21, 12, 13],
            [2, 3, 4, 5, 6, 7, 0, 1, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 22, 23, 16, 17, 18, 19, 20, 21],
            [15, 14, 2, 3, 4, 5, 6, 7, 1, 0, 10, 11, 12, 13, 17, 16, 9, 8, 18, 19, 20, 21, 22, 23],
            [0, 1, 2, 3, 11, 10, 6, 7, 8, 9, 21, 20, 5, 4, 14, 15, 16, 17, 18, 19, 13, 12, 22, 23],
            [0, 1, 8, 9, 4, 5, 6, 7, 18, 19, 2, 3, 12, 13, 14, 15, 16, 17, 10, 11, 20, 21, 22, 23],
            [0, 1, 2, 3, 4, 5, 12, 13, 8, 9, 10, 11, 22, 23, 6, 7, 16, 17, 18, 19, 20, 21, 14, 15]
        ],
    ], dtype=torch.uint8, device=device)

    @classmethod
    def move(cls, action: int, state: torch.Tensor) -> torch.Tensor:
        new_state = state.clone()
        action = torch.Tensor([action]).to(cls.dtype)
        if state.is_cuda:
            _CUBELIB_CUDA.multi_act(
                ptr(cls.full_action_maps),
                ptr(new_state),
                ptr(action.cuda()),
                1,
            )
            torch.cuda.synchronize()
        else:
            LIBDSC.cube_multi_act(
                ptr(new_state),
                ptr(action),
                1,
            )
        return new_state

    @classmethod
    def multiple_moves(cls, actions: torch.Tensor, states: torch.Tensor, inplace=False) -> torch.Tensor:
        assert actions.device == states.device
        new_states = states if inplace else states.clone()
        if new_states.is_cuda:
            _CUBELIB_CUDA.multi_act(
                ptr(cls.full_action_maps),
                ptr(new_states),
                ptr(actions),
                len(actions),
            )
            torch.cuda.synchronize()
        else:
            LIBDSC.cube_multi_act(
                ptr(new_states),
                ptr(actions),
                len(actions),
            )

        return new_states

    @classmethod
    def reverse_move(cls, action: int) -> int:
        return action + 6 if action < 6 else action - 6

    @classmethod
    def reverse_moves(cls, actions: torch.Tensor) -> torch.Tensor:
        pos_dir = actions >= 6
        actions = actions + 6
        actions[pos_dir] -= 12
        return actions

    @classmethod
    def _as633(cls, state: torch.Tensor) -> np.ndarray:

        state = state.numpy()
        state633 = (np.ones((3, 3, 6)) * np.arange(6)).transpose(2, 1, 0).astype(int)
        for i in range(8):
            # Inserts values for corner i in position pos
            pos = state[i] // 3
            orientation = state[i] % 3
            # For these corners, "right turn" order is 0 2 1 instead of 0 1 2, so orientation is messed up without this fix
            if pos in [0, 2, 5, 7]:
                orientation *= -1
            values = np.roll([x[0] for x in cls.corner_633map[i]], orientation)
            state633[cls.corner_633map[pos][0]] = values[0]
            state633[cls.corner_633map[pos][1]] = values[1]
            state633[cls.corner_633map[pos][2]] = values[2]

        for i in range(12):
            # Inserts values for side i in position pos
            pos = state[i + 8] // 2
            orientation = state[i + 8] % 2
            values = np.roll([x[0] for x in cls.side_633map[i]], orientation)
            state633[cls.side_633map[pos][0]] = values[0]
            state633[cls.side_633map[pos][1]] = values[1]

        return state633

    @classmethod
    def _stringify_cube(cls, state633: np.ndarray) -> str:
        stringarr = np.empty((9, 12), dtype=str)
        stringarr[...] = " "
        simple = np.array([
            [-1, cls.T, -1, -1],
            [cls.L, cls.F, cls.R, cls.B],
            [-1, cls.D, -1, -1],
        ], dtype=int)
        for i in range(6):
            pos = tuple(int(x) for x in np.where(simple == i))
            stringarr[pos[0] * 3: pos[0] * 3 + 3, pos[1] * 3: pos[1] * 3 + 3] = state633[i].astype(str)
        string = "\n".join([" ".join(list(y)) for y in stringarr])
        return string

    @classmethod
    def string(cls, state: torch.Tensor) -> str:
        return cls._stringify_cube(cls._as633(state.cpu()))

_ENVS = {
    "cube": _CubeEnvironment,
}

def get_env(env: str) -> Type[Environment]:
    return _ENVS[env]
