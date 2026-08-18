from abc import ABC, abstractmethod


class Subscriber(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def handle_new_action_potential(self, new_observation: Observation) -> None:
        ...
