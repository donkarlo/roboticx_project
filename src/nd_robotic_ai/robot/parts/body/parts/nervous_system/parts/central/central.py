from nd_robotic_ai.robot.parts.body.parts.nervous_system.parts.central.part.brain.brain import Brain
from nd_robotic_ai.robot.parts.body.parts.nervous_system.parts.central.part.retina.retina import Retina
from nd_robotic_ai.robot.parts.body.parts.nervous_system.parts.central.part.spinal_cord.spinal_cord import \
    SpinalCord
from nd_robotic_ai.robot.composition.composite import Composite as RobioticCompositeUnit


class Central(RobioticCompositeUnit):
    def __init__(self):
        RobioticCompositeUnit.__init__(self)
        self.add_child(Brain())
        self.add_child(SpinalCord())
        self.add_child(Retina())
