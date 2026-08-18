from abc import ABC, abstractmethod

from nd_math.probability.bayesian.observation import Observation


class Publisher(ABC):
    @abstractmethod
    def notify_action_potential_subscribers(self, observation:Observation) -> None:
        ...
