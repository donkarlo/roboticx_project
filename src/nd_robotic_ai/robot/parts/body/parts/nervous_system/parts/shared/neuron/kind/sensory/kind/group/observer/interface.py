from typing import Protocol

from torch.testing._internal.distributed.rpc.examples.reinforcement_learning_rpc_test import Observer


class Interface(Protocol):
    def attach_new_sensor_observer(self, observer:Observer)->None:
        """
        When a new sensory is added probably things like this might happen
        - individual segragted predictive models such as when an IMU is added, then GPS Predicting models nad LIDAR Predicting models can be probably improved?
        """
        ...