from nd_robotic_ai.robot.composition.composite import Composite as RobioticCompositeUnit

class Retina(RobioticCompositeUnit):
    """
    Any self-processing robot component can be a Retina such as sensors that first process data a little with their internal chip and then they send it to brain
    """
    def __init__(self):
        RobioticCompositeUnit.__init__(self)