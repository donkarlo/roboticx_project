from __future__ import annotations

import math
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy
from scipy.optimize import differential_evolution
from scipy.optimize import least_squares


class FmmComponentParameters:
    def __init__(self, amplitude: float, alpha: float, beta: float, omega: float) -> None:
        self.amplitude = amplitude
        self.alpha = alpha
        self.beta = beta
        self.omega = omega

    def get_as_list(self) -> List[float]:
        return [
            self.amplitude,
            self.alpha,
            self.beta,
            self.omega
        ]

    def create_wrapped_angle(self, angle: float) -> float:
        wrapped_angle = angle

        while wrapped_angle >= 2.0 * math.pi:
            wrapped_angle = wrapped_angle - 2.0 * math.pi

        while wrapped_angle < 0.0:
            wrapped_angle = wrapped_angle + 2.0 * math.pi

        return wrapped_angle

    def create_modified(self, amplitude_scale: float, alpha_shift: float, beta_shift: float,
                        omega_scale: float) -> FmmComponentParameters:
        shifted_alpha = self.create_wrapped_angle(self.alpha + alpha_shift)
        shifted_beta = self.create_wrapped_angle(self.beta + beta_shift)

        modified_omega = self.omega * omega_scale

        if modified_omega < 0.001:
            modified_omega = 0.001

        if modified_omega > 1.0:
            modified_omega = 1.0

        parameters = FmmComponentParameters(
            self.amplitude * amplitude_scale,
            shifted_alpha,
            shifted_beta,
            modified_omega
        )

        return parameters


class SingleFmmSignalParameters:
    def __init__(self, baseline: float, component_parameters: FmmComponentParameters) -> None:
        self.baseline = baseline
        self.component_parameters = component_parameters

    def get_as_list(self) -> List[float]:
        return [
            self.baseline,
            self.component_parameters.amplitude,
            self.component_parameters.alpha,
            self.component_parameters.beta,
            self.component_parameters.omega
        ]


class TwoComponentFmmSignalParameters:
    def __init__(self, baseline: float, first_component_parameters: FmmComponentParameters,
                 second_component_parameters: FmmComponentParameters) -> None:
        self.baseline = baseline
        self.first_component_parameters = first_component_parameters
        self.second_component_parameters = second_component_parameters


class FmmComponent:
    def __init__(self, parameters: FmmComponentParameters) -> None:
        self.parameters = parameters

    def evaluate(self, time_points: numpy.ndarray) -> numpy.ndarray:
        half_angle_difference = 0.5 * (time_points - self.parameters.alpha)

        phase = self.parameters.beta + 2.0 * numpy.arctan2(
            self.parameters.omega * numpy.sin(half_angle_difference),
            numpy.cos(half_angle_difference)
        )

        values = self.parameters.amplitude * numpy.cos(phase)
        return values


class SingleFmmSignal:
    def __init__(self, parameters: SingleFmmSignalParameters) -> None:
        self.parameters = parameters
        self.component = FmmComponent(parameters.component_parameters)

    def evaluate(self, time_points: numpy.ndarray) -> numpy.ndarray:
        values = self.parameters.baseline + self.component.evaluate(time_points)
        return values


class ActionPotentialLikeFmmSignal:
    def __init__(self, parameters: TwoComponentFmmSignalParameters, resting_potential: float,
                 reference_time: float) -> None:
        self.parameters = parameters
        self.resting_potential = resting_potential
        self.reference_time = reference_time
        self.first_component = FmmComponent(parameters.first_component_parameters)
        self.second_component = FmmComponent(parameters.second_component_parameters)
        self.resting_offset = self.create_resting_offset()

    def create_resting_offset(self) -> float:
        reference_time_points = numpy.array([self.reference_time], dtype=float)
        reference_value = self.evaluate_without_resting_correction(reference_time_points)[0]
        resting_offset = self.resting_potential - float(reference_value)
        return resting_offset

    def evaluate_without_resting_correction(self, time_points: numpy.ndarray) -> numpy.ndarray:
        values = self.parameters.baseline
        values = values + self.first_component.evaluate(time_points)
        values = values + self.second_component.evaluate(time_points)
        return values

    def evaluate(self, time_points: numpy.ndarray) -> numpy.ndarray:
        values = self.evaluate_without_resting_correction(time_points) + self.resting_offset
        return values


class WeightedCombinedActionPotentialSignal:
    def __init__(self, resting_potential: float, first_signal: ActionPotentialLikeFmmSignal,
                 second_signal: ActionPotentialLikeFmmSignal, first_weight: float, second_weight: float) -> None:
        self.resting_potential = resting_potential
        self.first_signal = first_signal
        self.second_signal = second_signal
        self.first_weight = first_weight
        self.second_weight = second_weight

    def evaluate(self, time_points: numpy.ndarray) -> numpy.ndarray:
        first_wave_values = self.first_signal.evaluate(time_points)
        second_wave_values = self.second_signal.evaluate(time_points)

        first_deviation = first_wave_values - self.resting_potential
        second_deviation = second_wave_values - self.resting_potential

        values = self.resting_potential
        values = values + self.first_weight * first_deviation
        values = values + self.second_weight * second_deviation

        return values


class ActionPotentialTemplateFactory:
    def create_template_parameters(self) -> TwoComponentFmmSignalParameters:
        first_component_parameters = FmmComponentParameters(
            53.03187484,
            5.06442008,
            2.40657480,
            0.10937460
        )

        second_component_parameters = FmmComponentParameters(
            34.65401532,
            5.39860485,
            5.39098286,
            0.13734080
        )

        parameters = TwoComponentFmmSignalParameters(
            -50.51165188,
            first_component_parameters,
            second_component_parameters
        )

        return parameters


class RandomActionPotentialParameterFactory:
    def __init__(self) -> None:
        self.random_generator = numpy.random.default_rng()
        self.template_factory = ActionPotentialTemplateFactory()

    def create_random_parameters(self) -> TwoComponentFmmSignalParameters:
        template_parameters = self.template_factory.create_template_parameters()

        baseline_shift = float(self.random_generator.uniform(-4.0, 4.0))

        first_amplitude_scale = float(self.random_generator.uniform(0.85, 1.15))
        first_alpha_shift = float(self.random_generator.uniform(-0.20, 0.20))
        first_beta_shift = float(self.random_generator.uniform(-0.20, 0.20))
        first_omega_scale = float(self.random_generator.uniform(0.85, 1.15))

        second_amplitude_scale = float(self.random_generator.uniform(0.85, 1.15))
        second_alpha_shift = float(self.random_generator.uniform(-0.20, 0.20))
        second_beta_shift = float(self.random_generator.uniform(-0.20, 0.20))
        second_omega_scale = float(self.random_generator.uniform(0.85, 1.15))

        first_component_parameters = template_parameters.first_component_parameters.create_modified(
            first_amplitude_scale,
            first_alpha_shift,
            first_beta_shift,
            first_omega_scale
        )

        second_component_parameters = template_parameters.second_component_parameters.create_modified(
            second_amplitude_scale,
            second_alpha_shift,
            second_beta_shift,
            second_omega_scale
        )

        parameters = TwoComponentFmmSignalParameters(
            template_parameters.baseline + baseline_shift,
            first_component_parameters,
            second_component_parameters
        )

        return parameters


class FmmParameterVectorFactory:
    def create_single_fmm_parameters_from_vector(self, parameter_vector: numpy.ndarray) -> SingleFmmSignalParameters:
        baseline = float(parameter_vector[0])
        amplitude = float(parameter_vector[1])
        alpha = float(parameter_vector[2])
        beta = float(parameter_vector[3])
        omega = float(parameter_vector[4])

        component_parameters = FmmComponentParameters(
            amplitude,
            alpha,
            beta,
            omega
        )

        parameters = SingleFmmSignalParameters(
            baseline,
            component_parameters
        )

        return parameters


class SingleFmmFitObjective:
    def __init__(self, time_points: numpy.ndarray, target_values: numpy.ndarray) -> None:
        self.time_points = time_points
        self.target_values = target_values
        self.parameter_vector_factory = FmmParameterVectorFactory()

    def compute_residuals(self, parameter_vector: numpy.ndarray) -> numpy.ndarray:
        parameters = self.parameter_vector_factory.create_single_fmm_parameters_from_vector(parameter_vector)
        signal = SingleFmmSignal(parameters)
        predicted_values = signal.evaluate(self.time_points)
        residuals = self.target_values - predicted_values
        return residuals

    def compute_sum_of_squared_errors(self, parameter_vector: numpy.ndarray) -> float:
        residuals = self.compute_residuals(parameter_vector)
        sum_of_squared_errors = float(numpy.sum(residuals ** 2))
        return sum_of_squared_errors


class SingleFmmParameterBounds:
    def __init__(self, target_values: numpy.ndarray) -> None:
        target_minimum = float(numpy.min(target_values))
        target_maximum = float(numpy.max(target_values))
        signal_range = target_maximum - target_minimum

        if signal_range <= 0.0:
            signal_range = 1.0

        self.baseline_lower = target_minimum - signal_range
        self.baseline_upper = target_maximum + signal_range
        self.amplitude_lower = 0.0
        self.amplitude_upper = 2.0 * signal_range
        self.alpha_lower = 0.0
        self.alpha_upper = 2.0 * math.pi
        self.beta_lower = 0.0
        self.beta_upper = 2.0 * math.pi
        self.omega_lower = 0.001
        self.omega_upper = 1.0

    def get_differential_evolution_bounds(self) -> List[Tuple[float, float]]:
        bounds = [
            (self.baseline_lower, self.baseline_upper),
            (self.amplitude_lower, self.amplitude_upper),
            (self.alpha_lower, self.alpha_upper),
            (self.beta_lower, self.beta_upper),
            (self.omega_lower, self.omega_upper)
        ]

        return bounds

    def get_least_squares_bounds(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        lower_bounds = numpy.array(
            [
                self.baseline_lower,
                self.amplitude_lower,
                self.alpha_lower,
                self.beta_lower,
                self.omega_lower
            ],
            dtype=float
        )

        upper_bounds = numpy.array(
            [
                self.baseline_upper,
                self.amplitude_upper,
                self.alpha_upper,
                self.beta_upper,
                self.omega_upper
            ],
            dtype=float
        )

        return lower_bounds, upper_bounds


class FitMetrics:
    def __init__(self, target_values: numpy.ndarray, fitted_values: numpy.ndarray) -> None:
        self.target_values = target_values
        self.fitted_values = fitted_values

        self.residual_values = self.target_values - self.fitted_values
        self.absolute_error_values = numpy.abs(self.residual_values)

        self.dynamic_range = self.compute_dynamic_range()
        self.sum_of_squared_errors = self.compute_sum_of_squared_errors()
        self.mean_absolute_error = self.compute_mean_absolute_error()
        self.root_mean_squared_error = self.compute_root_mean_squared_error()
        self.maximum_absolute_error = self.compute_maximum_absolute_error()

        self.normalized_mean_absolute_error = self.mean_absolute_error / self.dynamic_range
        self.normalized_root_mean_squared_error = self.root_mean_squared_error / self.dynamic_range
        self.normalized_maximum_absolute_error = self.maximum_absolute_error / self.dynamic_range
        self.normalized_absolute_error_values = self.absolute_error_values / self.dynamic_range

        self.r_squared = self.compute_r_squared()

    def compute_dynamic_range(self) -> float:
        target_minimum = float(numpy.min(self.target_values))
        target_maximum = float(numpy.max(self.target_values))
        dynamic_range = target_maximum - target_minimum

        if dynamic_range <= 1e-12:
            dynamic_range = 1e-12

        return dynamic_range

    def compute_sum_of_squared_errors(self) -> float:
        sum_of_squared_errors = float(numpy.sum(self.residual_values ** 2))
        return sum_of_squared_errors

    def compute_mean_absolute_error(self) -> float:
        mean_absolute_error = float(numpy.mean(self.absolute_error_values))
        return mean_absolute_error

    def compute_root_mean_squared_error(self) -> float:
        root_mean_squared_error = float(numpy.sqrt(numpy.mean(self.residual_values ** 2)))
        return root_mean_squared_error

    def compute_maximum_absolute_error(self) -> float:
        maximum_absolute_error = float(numpy.max(self.absolute_error_values))
        return maximum_absolute_error

    def compute_r_squared(self) -> float:
        target_mean = float(numpy.mean(self.target_values))
        total_sum_of_squares = float(numpy.sum((self.target_values - target_mean) ** 2))

        if total_sum_of_squares <= 1e-12:
            r_squared = 0.0
        else:
            r_squared = 1.0 - self.sum_of_squared_errors / total_sum_of_squares

        return r_squared


class SingleFmmFitResult:
    def __init__(self, parameters: SingleFmmSignalParameters, fitted_values: numpy.ndarray,
                 metrics: FitMetrics) -> None:
        self.parameters = parameters
        self.fitted_values = fitted_values
        self.metrics = metrics

    def create_signal(self) -> SingleFmmSignal:
        signal = SingleFmmSignal(self.parameters)
        return signal


class SingleFmmWaveFitter:
    def __init__(self, maximum_iterations: int) -> None:
        self.maximum_iterations = maximum_iterations
        self.parameter_vector_factory = FmmParameterVectorFactory()
        self.random_generator = numpy.random.default_rng()

    def create_fit_random_seed(self) -> int:
        random_seed = int(self.random_generator.integers(0, 2147483647))
        return random_seed

    def fit(self, time_points: numpy.ndarray, target_values: numpy.ndarray) -> SingleFmmFitResult:
        objective = SingleFmmFitObjective(time_points, target_values)
        bounds = SingleFmmParameterBounds(target_values)
        fit_random_seed = self.create_fit_random_seed()

        global_result = differential_evolution(
            objective.compute_sum_of_squared_errors,
            bounds.get_differential_evolution_bounds(),
            maxiter=self.maximum_iterations,
            seed=fit_random_seed,
            polish=False,
            popsize=8,
            tol=0.01
        )

        local_result = least_squares(
            objective.compute_residuals,
            global_result.x,
            bounds=bounds.get_least_squares_bounds(),
            max_nfev=2500
        )

        best_parameter_vector = local_result.x
        best_parameters = self.parameter_vector_factory.create_single_fmm_parameters_from_vector(best_parameter_vector)

        fitted_signal = SingleFmmSignal(best_parameters)
        fitted_values = fitted_signal.evaluate(time_points)

        metrics = FitMetrics(
            target_values,
            fitted_values
        )

        result = SingleFmmFitResult(
            best_parameters,
            fitted_values,
            metrics
        )

        return result


class CombinedWaveFitResult:
    def __init__(self, pair_index: int, combined_values: numpy.ndarray, fit_result: SingleFmmFitResult) -> None:
        self.pair_index = pair_index
        self.combined_values = combined_values
        self.fit_result = fit_result


class MetricSummary:
    def __init__(self, metric_name: str, values: numpy.ndarray) -> None:
        self.metric_name = metric_name
        self.values = values

    def print_summary(self) -> None:
        print("")
        print(self.metric_name)
        print("mean     =", float(numpy.mean(self.values)))
        print("variance =", float(numpy.var(self.values)))
        print("std      =", float(numpy.std(self.values)))
        print("median   =", float(numpy.median(self.values)))
        print("p90      =", float(numpy.percentile(self.values, 90)))
        print("p95      =", float(numpy.percentile(self.values, 95)))
        print("minimum  =", float(numpy.min(self.values)))
        print("maximum  =", float(numpy.max(self.values)))


class BatchExperimentResult:
    def __init__(self, time_points: numpy.ndarray, pair_results: List[CombinedWaveFitResult]) -> None:
        self.time_points = time_points
        self.pair_results = pair_results

    def create_metric_array(self, metric_name: str) -> numpy.ndarray:
        value_list = []

        for pair_result in self.pair_results:
            metrics = pair_result.fit_result.metrics

            if metric_name == "normalized_mean_absolute_error":
                value_list.append(metrics.normalized_mean_absolute_error)
            elif metric_name == "normalized_root_mean_squared_error":
                value_list.append(metrics.normalized_root_mean_squared_error)
            elif metric_name == "normalized_maximum_absolute_error":
                value_list.append(metrics.normalized_maximum_absolute_error)
            elif metric_name == "r_squared":
                value_list.append(metrics.r_squared)
            elif metric_name == "dynamic_range":
                value_list.append(metrics.dynamic_range)
            elif metric_name == "mean_absolute_error":
                value_list.append(metrics.mean_absolute_error)
            elif metric_name == "root_mean_squared_error":
                value_list.append(metrics.root_mean_squared_error)
            elif metric_name == "maximum_absolute_error":
                value_list.append(metrics.maximum_absolute_error)
            elif metric_name == "sum_of_squared_errors":
                value_list.append(metrics.sum_of_squared_errors)
            else:
                raise ValueError("Unknown metric name: " + metric_name)

        values = numpy.array(value_list, dtype=float)
        return values

    def get_sorted_results_by_normalized_root_mean_squared_error(self) -> List[CombinedWaveFitResult]:
        sorted_results = sorted(
            self.pair_results,
            key=lambda pair_result: pair_result.fit_result.metrics.normalized_root_mean_squared_error
        )

        return sorted_results

    def get_best_result(self) -> CombinedWaveFitResult:
        sorted_results = self.get_sorted_results_by_normalized_root_mean_squared_error()
        result = sorted_results[0]
        return result

    def get_median_result(self) -> CombinedWaveFitResult:
        sorted_results = self.get_sorted_results_by_normalized_root_mean_squared_error()
        median_index = int(len(sorted_results) / 2)
        result = sorted_results[median_index]
        return result

    def get_worst_result(self) -> CombinedWaveFitResult:
        sorted_results = self.get_sorted_results_by_normalized_root_mean_squared_error()
        result = sorted_results[-1]
        return result

    def print_summary(self) -> None:
        print("")
        print("Batch experiment summary")
        print("pair_count =", len(self.pair_results))

        MetricSummary(
            "Normalized MAE",
            self.create_metric_array("normalized_mean_absolute_error")
        ).print_summary()

        MetricSummary(
            "Normalized RMSE",
            self.create_metric_array("normalized_root_mean_squared_error")
        ).print_summary()

        MetricSummary(
            "Normalized MAX AE",
            self.create_metric_array("normalized_maximum_absolute_error")
        ).print_summary()

        MetricSummary(
            "R squared",
            self.create_metric_array("r_squared")
        ).print_summary()

        MetricSummary(
            "Dynamic range",
            self.create_metric_array("dynamic_range")
        ).print_summary()

        MetricSummary(
            "Raw RMSE",
            self.create_metric_array("root_mean_squared_error")
        ).print_summary()

        best_result = self.get_best_result()
        median_result = self.get_median_result()
        worst_result = self.get_worst_result()

        print("")
        print("Representative cases by normalized RMSE")
        print("best_pair_index   =", best_result.pair_index)
        print("best_nrmse        =", best_result.fit_result.metrics.normalized_root_mean_squared_error)
        print("median_pair_index =", median_result.pair_index)
        print("median_nrmse      =", median_result.fit_result.metrics.normalized_root_mean_squared_error)
        print("worst_pair_index  =", worst_result.pair_index)
        print("worst_nrmse       =", worst_result.fit_result.metrics.normalized_root_mean_squared_error)


class BatchFmmExperiment:
    def __init__(self, pair_count: int, sample_count: int, maximum_iterations: int) -> None:
        self.pair_count = pair_count
        self.sample_count = sample_count
        self.maximum_iterations = maximum_iterations

        self.time_start = 0.0
        self.time_end = 2.0 * math.pi
        self.resting_potential = -70.0
        self.reference_time = 0.0
        self.first_weight = 0.55
        self.second_weight = 0.45

        self.random_parameter_factory = RandomActionPotentialParameterFactory()
        self.fitter = SingleFmmWaveFitter(maximum_iterations)

    def create_time_points(self) -> numpy.ndarray:
        time_points = numpy.linspace(
            self.time_start,
            self.time_end,
            self.sample_count,
            endpoint=False
        )

        return time_points

    def create_random_action_potential_signal(self) -> ActionPotentialLikeFmmSignal:
        parameters = self.random_parameter_factory.create_random_parameters()

        signal = ActionPotentialLikeFmmSignal(
            parameters,
            self.resting_potential,
            self.reference_time
        )

        return signal

    def create_combined_signal(self) -> WeightedCombinedActionPotentialSignal:
        first_signal = self.create_random_action_potential_signal()
        second_signal = self.create_random_action_potential_signal()

        combined_signal = WeightedCombinedActionPotentialSignal(
            self.resting_potential,
            first_signal,
            second_signal,
            self.first_weight,
            self.second_weight
        )

        return combined_signal

    def run_single_pair(self, time_points: numpy.ndarray, pair_index: int) -> CombinedWaveFitResult:
        combined_signal = self.create_combined_signal()
        combined_values = combined_signal.evaluate(time_points)
        fit_result = self.fitter.fit(time_points, combined_values)

        result = CombinedWaveFitResult(
            pair_index,
            combined_values,
            fit_result
        )

        return result

    def run(self) -> BatchExperimentResult:
        time_points = self.create_time_points()
        pair_results = []

        for pair_index in range(self.pair_count):
            pair_result = self.run_single_pair(time_points, pair_index)
            pair_results.append(pair_result)

            finished_pair_count = pair_index + 1

            if finished_pair_count % 10 == 0:
                print("finished_pairs =", finished_pair_count)

        experiment_result = BatchExperimentResult(
            time_points,
            pair_results
        )

        return experiment_result


class PerformancePlotter:
    def plot(self, experiment_result: BatchExperimentResult) -> None:
        median_result = experiment_result.get_median_result()

        time_points = experiment_result.time_points
        combined_values = median_result.combined_values
        fitted_values = median_result.fit_result.fitted_values
        normalized_absolute_error_values = median_result.fit_result.metrics.normalized_absolute_error_values

        normalized_root_mean_squared_error_values = experiment_result.create_metric_array(
            "normalized_root_mean_squared_error"
        )

        normalized_mean_absolute_error_values = experiment_result.create_metric_array(
            "normalized_mean_absolute_error"
        )

        dynamic_range_values = experiment_result.create_metric_array(
            "dynamic_range"
        )

        figure, axes = plt.subplots(
            2,
            2,
            figsize=(14, 10)
        )

        axes[0, 0].plot(
            time_points,
            combined_values,
            label="Combined signal"
        )

        axes[0, 0].plot(
            time_points,
            fitted_values,
            linestyle="--",
            label="Fitted single FMM"
        )

        axes[0, 0].set_title(
            "Median case fit\n"
            + "normalized RMSE="
            + str(round(median_result.fit_result.metrics.normalized_root_mean_squared_error, 4))
            + ", R squared="
            + str(round(median_result.fit_result.metrics.r_squared, 4))
        )

        axes[0, 0].set_xlabel("Time")
        axes[0, 0].set_ylabel("Membrane potential")
        axes[0, 0].legend()
        axes[0, 0].grid(True)

        axes[0, 1].plot(
            time_points,
            normalized_absolute_error_values
        )

        axes[0, 1].set_title("Normalized absolute error for median case")
        axes[0, 1].set_xlabel("Time")
        axes[0, 1].set_ylabel("Absolute error / dynamic range")
        axes[0, 1].grid(True)

        axes[1, 0].hist(
            normalized_root_mean_squared_error_values,
            bins=20
        )

        axes[1, 0].set_title("Distribution of normalized RMSE over all pairs")
        axes[1, 0].set_xlabel("Normalized RMSE")
        axes[1, 0].set_ylabel("Count")
        axes[1, 0].grid(True)

        axes[1, 1].scatter(
            dynamic_range_values,
            normalized_root_mean_squared_error_values,
            label="Normalized RMSE"
        )

        axes[1, 1].scatter(
            dynamic_range_values,
            normalized_mean_absolute_error_values,
            label="Normalized MAE"
        )

        axes[1, 1].set_title("Error versus dynamic range")
        axes[1, 1].set_xlabel("Dynamic range")
        axes[1, 1].set_ylabel("Normalized error")
        axes[1, 1].legend()
        axes[1, 1].grid(True)

        figure.tight_layout()
        plt.show()


class ExampleApplication:
    def __init__(self) -> None:
        self.pair_count = 200
        self.sample_count = 600
        self.maximum_iterations = 80

    def run(self) -> None:
        experiment = BatchFmmExperiment(
            self.pair_count,
            self.sample_count,
            self.maximum_iterations
        )

        result = experiment.run()
        result.print_summary()

        plotter = PerformancePlotter()
        plotter.plot(result)


if __name__ == "__main__":
    application = ExampleApplication()
    application.run()