from abc import ABC, abstractmethod
from typing import List

from nd_robotic_ai.robot.parts.body.parts.nervous_system.neural_coding.action_potential.action_potential import \
    ActionPotential
from nd_robotic_ai.robot.parts.body.parts.nervous_system.neural_coding.action_potential.observer.publisher import \
    Publisher as ObservationPublisher
from nd_robotic_ai.robot.parts.body.parts.nervous_system.parts.neuron.neuron import Neuron
from nd_robotic_ai.robot.parts.stimulus.stimulus import Stimulus


class Receptor(Neuron, ObservationPublisher, ABC):
    """
    #Receptor
    """
    def __init__(self):
     self._fired_action_potential_subscribers = []

    def attach_new_abservation_subscriber(self, subscriber)->None:
        self._fired_action_potential_subscribers.append(subscriber)

    def notify_action_potential_subscribers(self, sitimulus:Stimulus)->None:
        for new_observation_subscriber  in self._fired_action_potential_subscribers:
            new_observation_subscriber.handle_action_potential(sitimulus)

    def receive_stimulus(self, sitimulus:Stimulus)->None:
        self._convert_to_action_potentials(sitimulus)

    @abstractmethod
    def _convert_to_action_potentials(self, stimulus:Stimulus)->None:
        pass

    def _sort_action_potential(self, action_potential: ActionPotential):
        """
        - see sorting action potentials
        Find out to which branch in self._action_potential_composite   the new action potential belongs acoording to its shape
        Args:
            action_potential:

        Returns:

        """
        generating_neural_coding_is_necessary = False
        if generating_neural_coding_is_necessary:
            self._generate_neural_coding()

    def _generate_neural_coding(self):
        """
        Use a autu encoder for generating neural coding
        Returns:

        """
        pass
