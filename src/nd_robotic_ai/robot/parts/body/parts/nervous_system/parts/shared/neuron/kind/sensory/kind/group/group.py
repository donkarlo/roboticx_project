from nd_robotic_ai.robot.robot import Sensor
from typing import Optional, Sequence, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from nd_robotic_ai.robot.robot import Sensor

class Group:
    '''
    Sensors of the same parent for example members of a robot
    - We do not use composit because a sensory must be
    '''
    def __init__(self, members: Sequence[Union["Sensory", "SingleType"]], name:Optional[str] = None, parent_name:
    Optional[str] = None):
        """

        Args:
            members:
            name: assigned from a bigger action_potential_group for example uav1 or uav2
            parent_name: None if it is the parent or the sensory network support no hierarchy
        """

        #@todo either it should be a set of sets or set of members
        # if the members do not have ids automatically generate unique ids
        self._memebers = members
        self._name = name
        self._parent_name = parent_name

    def get_sensor_set_id(self)->Optional[str]:
        return self._name

    def get_parent_id(self)->Optional[str]:
        return self._parent_name