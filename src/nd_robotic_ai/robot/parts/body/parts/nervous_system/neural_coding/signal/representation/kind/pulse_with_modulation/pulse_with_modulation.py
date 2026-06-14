from nd_robotic_ai.robot.state.state import State


class PulseWithModulation(State):
    """
    TODO: Can be associated with neural coding.

    Application: By changing PWC shape we can increase or decrease a rotor_composite speed
    Application 2:  changing the shape is changing the frequencey of the voltage in brushless rotors

    Represents an ideal PWM waveform.

    - frequency_hz: carrier frequency in Hz
    - duty_cycle: fraction in [0.0, 1.0]
    - high_level: output level when "on" (e.g., volts)
    - low_level: output level when "off" (e.g., volts)
    - phase_seconds: time_based shift applied to the waveform (seconds)
    """

    def __init__(self, frequency_hz: float, duty_cycle: float, high_level: float = 1.0, low_level: float = 0.0,
                 phase_seconds: float = 0.0) -> None:
        if frequency_hz <= 0.0:
            raise ValueError(f"frequency_hz must be > 0, got {frequency_hz}")

        if duty_cycle < 0.0:
            raise ValueError(f"duty_cycle must be >= 0, got {duty_cycle}")
        else:
            if duty_cycle > 1.0:
                raise ValueError(f"duty_cycle must be <= 1, got {duty_cycle}")

        self._frequency_hz: float = frequency_hz
        self._duty_cycle: float = duty_cycle
        self._high_level: float = high_level
        self._low_level: float = low_level
        self._phase_seconds: float = phase_seconds

    def get_frequency_hz(self) -> float:
        return self._frequency_hz

    def get_duty_cycle(self) -> float:
        return self._duty_cycle

    def get_high_level(self) -> float:
        return self._high_level

    def get_low_level(self) -> float:
        return self._low_level

    def get_phase_seconds(self) -> float:
        return self._phase_seconds

    def get_period_seconds(self) -> float:
        return 1.0 / self._frequency_hz

    def get_pulse_width_seconds(self) -> float:
        period_seconds: float = self.get_period_seconds()
        return period_seconds * self._duty_cycle

    def get_average_level(self) -> float:
        return (self._duty_cycle * self._high_level) + ((1.0 - self._duty_cycle) * self._low_level)

    def sample_level(self, time_seconds: float) -> float:
        """
        Returns the instantaneous level at time_seconds.

        This is an ideal rectangular PWM. No rise/fall time_based, no jitter.
        """
        period_seconds: float = self.get_period_seconds()
        local_time_seconds: float = time_seconds - self._phase_seconds

        cycles_float: float = local_time_seconds / period_seconds
        cycles_int: int = int(cycles_float)

        phase_in_period_seconds: float = local_time_seconds - (cycles_int * period_seconds)

        if phase_in_period_seconds < 0.0:
            phase_in_period_seconds = phase_in_period_seconds + period_seconds

        pulse_width_seconds: float = self.get_pulse_width_seconds()

        if phase_in_period_seconds < pulse_width_seconds:
            return self._high_level
        else:
            return self._low_level

    def with_duty_cycle(self, duty_cycle: float) -> "PulseWithModulation":
        return PulseWithModulation(
            frequency_hz=self._frequency_hz,
            duty_cycle=duty_cycle,
            high_level=self._high_level,
            low_level=self._low_level,
            phase_seconds=self._phase_seconds,
        )

    def with_frequency(self, frequency_hz: float) -> "PulseWithModulation":
        return PulseWithModulation(
            frequency_hz=frequency_hz,
            duty_cycle=self._duty_cycle,
            high_level=self._high_level,
            low_level=self._low_level,
            phase_seconds=self._phase_seconds,
        )

    def __repr__(self) -> str:
        return (
            f"PulseWithModulation("
            f"frequency_hz={self._frequency_hz}, "
            f"duty_cycle={self._duty_cycle}, "
            f"high_level={self._high_level}, "
            f"low_level={self._low_level}, "
            f"phase_seconds={self._phase_seconds})"
        )