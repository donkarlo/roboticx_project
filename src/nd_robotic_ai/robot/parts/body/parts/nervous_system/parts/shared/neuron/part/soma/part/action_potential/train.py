from nd_robotic_ai.robot.composition.kind.body.nervous_system.neuron.neural_coding.action_potential.action_potential import ActionPotential
from nd_utility.data.kind.group.group import Group as BaseGroup
from typing import List

class Train(BaseGroup):
    """
    Can be used for each cluster
    But in general_events it is related to
    population coding:
        https://en.wikipedia.org/wiki/Neural_coding#Population_coding

    """
    def __init__(self, action_potentials: List[ActionPotential]):
        BaseGroup.__init__(self, action_potentials)
