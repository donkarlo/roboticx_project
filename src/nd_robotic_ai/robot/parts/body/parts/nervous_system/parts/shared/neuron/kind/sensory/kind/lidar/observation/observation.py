from nd_physics.quantity.vector_quantifiable import VectorQuantifiable
from typing import override, Sequence
from nd_math.linear_algebra.tensor.vector.vector import Vector

class Observation(VectorQuantifiable):
    def __init__(self, ranges:Sequence[float]):
        self._ranges = ranges
        self._vector_representation = Vector(self._ranges)

    @override
    def get_vector_representation(self)->Vector:
        return self._vector_representation