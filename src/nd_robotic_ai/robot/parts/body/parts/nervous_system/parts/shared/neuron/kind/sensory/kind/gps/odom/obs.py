from nd_robotic_ai.robot.robot import Sensor
from nd_robotic_ai.robot.robot import Observation as SensorObs
from nd_math.linear_algebra.vec.opr.two_opranded import TwoOpranded
from nd_physics.dimension.unit.unit import Unit
from nd_physics.kinematics.pose import Pose
from nd_physics.kinematics.twist import Twist


class Obs(SensorObs):
    def __init__(self, sensor:Sensor, pose:Pose, pose_unit:Unit, twist:Twist, twist_unit:Unit):
        self._pose = pose
        self._pose_unit = pose_unit
        self._twist = twist
        self._twist_unit = twist_unit

        val = TwoOpranded(pose.get_components(), twist).concat()
        super().__init__(sensor, val)

