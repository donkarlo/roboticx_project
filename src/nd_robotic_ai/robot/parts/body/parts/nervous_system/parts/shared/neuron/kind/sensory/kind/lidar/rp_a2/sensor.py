from nd_math.linear_algebra.tensor.vector.vector import Vector
from nd_physics.dimension.unit.scalar import Scalar
from nd_physics.dimension.unit.unit import Unit
from nd_robotic_ai.robot.robot import Sensor as BasicSensor

class Sensor(BasicSensor):
    # vector dim is 720
    def __init__(self):
        freq = Scalar(Unit("hz"), 98.95)
        super().__init__("lidar", freq, "rp_a2")

    def validate_observation(self, ranges:Vector)->bool:
        return True