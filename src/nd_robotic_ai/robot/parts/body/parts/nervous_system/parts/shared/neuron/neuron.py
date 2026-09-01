from nd_robotic_ai.robot.composition.composite import Composite as RobotComposite
from nd_robotic_ai.robot.parts.body.parts.nervous_system.neural_coding.action_potential.action_potential import \
    ActionPotential
from nd_robotic_ai.robot.parts.body.parts.nervous_system.neural_coding.action_potential.observer.publisher import \
    Publisher as ActionPotentialPublisher
from nd_utility.data.kind.group.decorator.unikinded import Unikinded

ac
class Neuron(RobotComposite, ActionPotentialPublisher):
    def __init__(self):
        RobotComposite.__init__(self)



    def notify_action_potential_subscribers(self, action_potential: ActionPotential):
        pass

    def fire_action_potential_train(self) -> Unikinded[ActionPotential]:
        pass
