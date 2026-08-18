from nd_math.linear_algebra.tensor.vector.vector import Vector
from nd_robotic_ai.robot.robot import Observation as SensorObservation

class Observation(SensorObservation):
    def __init__(self, time:float, val:Vector):
        #@todo check 720 dim
        pass