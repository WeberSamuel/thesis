from abc import ABC, abstractmethod
import numpy as np
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvIndices
from gym.core import Wrapper
from gym import spaces
import cv2
from typing import (
    Any,
    List,
    NamedTuple,
    Optional,
    Type,
    Union,
)

from src.envs.samplers.base_sampler import BaseSampler


class MetaVecEnv(VecEnv):
    def __init__(
        self,
        num_envs: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        goal_sampler: BaseSampler,
    ):
        super().__init__(num_envs, observation_space, action_space)
        self.goal_sampler = goal_sampler
        self.goal_sampler._init_sampler(self)

    def reset_current_goals(self):
        self.goals_idx = np.random.randint(0, len(self.goal_sampler.goals), self.num_envs)
        self.goals = self.goal_sampler.goals[self.goals_idx]
        self.tasks = self.goal_sampler.tasks[self.goals_idx]

    def close(self) -> None:
        """Cleanup env."""
        cv2.destroyAllWindows()

    def env_is_wrapped(self, wrapper_class: Type[Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Test if any of the given environment indices are wrapped in a wrapper of class ``wrapper_class``.

        Args:
            wrapper_class (Type[gym.Wrapper]): The type of wrapper to check for.
            indices (VecEnvIndices, optional): The indices of the environment to check for. Defaults to None.

        Returns:
            List[bool]: List of bools indicating if the individual environments are wrapped.
        """
        return [False for i in indices] if isinstance(indices, List) else [False]

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return a list of values of the individual environments.

        Args:
            attr_name (str): The attribute to query.
            indices (VecEnvIndices, optional): The indicies of the environment from which the value is queried. Defaults to None.

        Returns:
            List[Any]: The values of the queried attribute from the environments
        """
        return [getattr(self, attr_name) for i in indices] if isinstance(indices, List) else [getattr(self, attr_name)]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set the value of an attribute in the individual environments specified by indicies.

        Args:
            attr_name (str): Name of the attribute to set.
            value (Any): Value to set in the individual environment.
            indices (VecEnvIndices, optional): The indicies of the individual environment for which the value shall be set. Defaults to None.
        """
        setattr(self, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call a method on the individual environment.

        Args:
            method_name (str): Name of the method to call.
            indices (VecEnvIndices, optional): Indices of the environment on which the method shall be called. Defaults to None.

        Returns:
            List[Any]: The returned values of the method for each environment.
        """
        result = getattr(self, method_name)(method_args, method_kwargs)
        return [result for i in indices] if isinstance(indices, List) else [result]

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        """Initialize the seed for this enivornment.

        Args:
            seed (Optional[int], optional): Seed to set. Defaults to None.

        Returns:
            List[Union[None, int]]: Seed of the individual environments.
        """
        self._seed = seed
        return [seed for i in range(self.num_envs)]
