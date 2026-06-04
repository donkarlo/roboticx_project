import numpy
import matplotlib.pyplot as pyplot


class FmmBump:
    def __init__(self, center: float, omega: float, amplitude: float, time_values: numpy.ndarray) -> None:
        self.center = center
        self.omega = omega
        self.amplitude = amplitude
        self.time_values = time_values

    def compute_values(self) -> numpy.ndarray:
        half_angle_difference = 0.5 * (self.time_values - self.center)

        phase_values = 2.0 * numpy.arctan2(
            self.omega * numpy.sin(half_angle_difference),
            numpy.cos(half_angle_difference),
        )

        bump_values = 0.5 * (1.0 + numpy.cos(phase_values))
        signal_values = self.amplitude * bump_values

        return signal_values


class ActionPotentialLikeFmmSignal:
    def __init__(self, time_values: numpy.ndarray) -> None:
        self.time_values = time_values

        self.depolarization_component = FmmBump(
            center=1.70,
            omega=8.00,
            amplitude=1.25,
            time_values=self.time_values,
        )

        self.undershoot_component = FmmBump(
            center=3.80,
            omega=2.00,
            amplitude=0.32,
            time_values=self.time_values,
        )

    def compute_values(self) -> tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]:
        depolarization_values = self.depolarization_component.compute_values()
        undershoot_values = self.undershoot_component.compute_values()

        signal_values = depolarization_values - undershoot_values
        corrected_signal_values = self.remove_boundary_offset(signal_values)

        return corrected_signal_values, depolarization_values, -undershoot_values

    def remove_boundary_offset(self, signal_values: numpy.ndarray) -> numpy.ndarray:
        start_value = signal_values[0]
        end_value = signal_values[-1]

        boundary_line_values = start_value + (end_value - start_value) * (
                (self.time_values - self.time_values[0]) / (self.time_values[-1] - self.time_values[0])
        )

        corrected_signal_values = signal_values - boundary_line_values

        return corrected_signal_values


class ActionPotentialPlotter:
    def __init__(self) -> None:
        self.time_values = numpy.linspace(0.0, 2.0 * numpy.pi, 1000, endpoint=True)
        self.signal = ActionPotentialLikeFmmSignal(time_values=self.time_values)

    def plot(self) -> None:
        signal_values, depolarization_values, undershoot_values = self.signal.compute_values()

        figure, axis = pyplot.subplots(figsize=(12, 7))

        axis.plot(self.time_values, signal_values, linewidth=2.5, label="action-potential-like FMM")
        axis.plot(self.time_values, depolarization_values, linestyle="--", label="depolarization component")
        axis.plot(self.time_values, undershoot_values, linestyle="--", label="undershoot component")

        axis.axhline(0.0, linewidth=0.8)
        axis.set_xlabel("t")
        axis.set_ylabel("v(t)")
        axis.set_title("Action-potential-like signal from two FMM components")
        axis.legend()
        axis.grid(True)

        pyplot.show()


if __name__ == "__main__":
    plotter = ActionPotentialPlotter()
    plotter.plot()