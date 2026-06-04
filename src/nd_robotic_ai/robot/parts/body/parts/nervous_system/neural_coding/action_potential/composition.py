from __future__ import annotations

import math
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from nd_sociomind.experiment.parts.oldest.uav1_300k_normal_time_position_modality import (
    Uav1300kNormalTimePositionModality,
)
from nd_sociomind.experiment.parts.oldest.uav1_normal_lidar_time_ranges_modality import (
    Uav1NormalLidarTimeRangesModality,
)


class SamplingLayer(tf.keras.layers.Layer):
    def call(self, inputs: Tuple[tf.Tensor, tf.Tensor]) -> tf.Tensor:
        latent_mean, latent_log_variance = inputs
        noise = tf.random.normal(shape=tf.shape(latent_mean))
        latent_vector = latent_mean + tf.exp(0.5 * latent_log_variance) * noise

        return latent_vector


class SensorStandardizer:
    def __init__(self) -> None:
        self.mean_values = None
        self.standard_deviation_values = None

    def fit(self, values: np.ndarray) -> None:
        self.mean_values = np.mean(values, axis=0)
        self.standard_deviation_values = np.std(values, axis=0)
        self.standard_deviation_values[self.standard_deviation_values < 1e-8] = 1.0

    def transform(self, values: np.ndarray) -> np.ndarray:
        if self.mean_values is None:
            raise ValueError("The standardizer must be fitted before transformation.")

        if self.standard_deviation_values is None:
            raise ValueError("The standardizer must be fitted before transformation.")

        normalized_values = (values - self.mean_values) / self.standard_deviation_values

        return normalized_values.astype(np.float32)

    def fit_transform(self, values: np.ndarray) -> np.ndarray:
        self.fit(values)
        normalized_values = self.transform(values)

        return normalized_values


class AlignedGpsLidarLoader:
    def __init__(self, data_slice: slice, maximum_lidar_range: float) -> None:
        self.data_slice = data_slice
        self.maximum_lidar_range = maximum_lidar_range

    def load(self) -> Tuple[np.ndarray, np.ndarray]:
        gps_memory = Uav1300kNormalTimePositionModality(self.data_slice)
        lidar_memory = Uav1NormalLidarTimeRangesModality(self.data_slice)

        gps_data = gps_memory.get_np_positions()
        lidar_data = lidar_memory.get_np_time_ranges()

        gps_points = gps_data[:, 0:3]

        if lidar_data.shape[1] == 720:
            lidar_ranges = lidar_data
        else:
            if lidar_data.shape[1] > 720:
                lidar_ranges = lidar_data[:, -720:]
            else:
                raise ValueError("LiDAR data must have at least 720 columns.")

        shared_row_count = min(gps_points.shape[0], lidar_ranges.shape[0])

        gps_points = gps_points[0:shared_row_count]
        lidar_ranges = lidar_ranges[0:shared_row_count]

        valid_gps_rows = np.isfinite(gps_points).all(axis=1)

        gps_points = gps_points[valid_gps_rows]
        lidar_ranges = lidar_ranges[valid_gps_rows]

        lidar_ranges = np.nan_to_num(
            lidar_ranges,
            nan=self.maximum_lidar_range,
            posinf=self.maximum_lidar_range,
            neginf=0.0,
        )

        lidar_ranges[lidar_ranges < 0.0] = 0.0
        lidar_ranges[lidar_ranges > self.maximum_lidar_range] = self.maximum_lidar_range

        if gps_points.shape[0] == 0:
            raise ValueError("No aligned GPS rows remained after preprocessing.")

        if lidar_ranges.shape[0] == 0:
            raise ValueError("No aligned LiDAR rows remained after preprocessing.")

        return gps_points.astype(np.float32), lidar_ranges.astype(np.float32)


class ConsecutiveWindowSelector:
    def __init__(self, start_index: int, window_size: int) -> None:
        self.start_index = start_index
        self.window_size = window_size

    def select(self, gps_points: np.ndarray, lidar_ranges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        end_index = self.start_index + self.window_size

        if self.start_index < 0:
            raise ValueError("start_index must not be negative.")

        if end_index > gps_points.shape[0]:
            raise ValueError("The selected GPS window is outside the data range.")

        if end_index > lidar_ranges.shape[0]:
            raise ValueError("The selected LiDAR window is outside the data range.")

        selected_gps_points = gps_points[self.start_index:end_index]
        selected_lidar_ranges = lidar_ranges[self.start_index:end_index]

        return selected_gps_points.astype(np.float32), selected_lidar_ranges.astype(np.float32)


class Fmm1ParameterBounder(tf.keras.layers.Layer):
    def __init__(self) -> None:
        super().__init__()

        self.baseline_scale = tf.constant(30.0, dtype=tf.float32)

        self.amplitude_minimum = tf.constant(40.0, dtype=tf.float32)
        self.amplitude_maximum = tf.constant(80.0, dtype=tf.float32)

        self.alpha_minimum = tf.constant(6.00, dtype=tf.float32)
        self.alpha_maximum = tf.constant(2.0 * math.pi, dtype=tf.float32)

        self.beta_minimum = tf.constant(3.75, dtype=tf.float32)
        self.beta_maximum = tf.constant(4.25, dtype=tf.float32)

        self.omega_minimum = tf.constant(0.09, dtype=tf.float32)
        self.omega_maximum = tf.constant(0.18, dtype=tf.float32)

    def call(self, raw_parameters: tf.Tensor) -> tf.Tensor:
        raw_baseline = raw_parameters[:, 0:1]
        raw_amplitude = raw_parameters[:, 1:2]
        raw_alpha = raw_parameters[:, 2:3]
        raw_beta = raw_parameters[:, 3:4]
        raw_omega = raw_parameters[:, 4:5]

        baseline = self.baseline_scale * tf.tanh(raw_baseline)

        amplitude = self.amplitude_minimum + (
                self.amplitude_maximum - self.amplitude_minimum
        ) * tf.sigmoid(raw_amplitude)

        alpha = self.alpha_minimum + (
                self.alpha_maximum - self.alpha_minimum
        ) * tf.sigmoid(raw_alpha)

        beta = self.beta_minimum + (
                self.beta_maximum - self.beta_minimum
        ) * tf.sigmoid(raw_beta)

        omega = self.omega_minimum + (
                self.omega_maximum - self.omega_minimum
        ) * tf.sigmoid(raw_omega)

        parameters = tf.concat(
            [
                baseline,
                amplitude,
                alpha,
                beta,
                omega,
            ],
            axis=1,
        )

        return parameters


class Fmm1WaveLayer(tf.keras.layers.Layer):
    def __init__(self, time_points: np.ndarray) -> None:
        super().__init__()
        self.time_points = tf.constant(time_points.astype(np.float32), dtype=tf.float32)

    def call(self, parameters: tf.Tensor) -> tf.Tensor:
        baseline = parameters[:, 0:1]
        amplitude = parameters[:, 1:2]
        alpha = parameters[:, 2:3]
        beta = parameters[:, 3:4]
        omega = parameters[:, 4:5]

        half_angle_difference = 0.5 * (self.time_points[tf.newaxis, :] - alpha)

        phase = beta + 2.0 * tf.atan2(
            omega * tf.sin(half_angle_difference),
            tf.cos(half_angle_difference),
        )

        values = baseline + amplitude * tf.cos(phase)
        values = values - values[:, 0:1]

        return values


class SensorToFmm1VariationalAutoencoder(tf.keras.Model):
    def __init__(self, input_dimension: int, output_dimension: int, time_points: np.ndarray,
                 hidden_layer_sizes: Tuple[int, ...]) -> None:
        super().__init__()

        encoder_layers = []
        encoder_layers.append(tf.keras.layers.InputLayer(shape=(input_dimension,)))

        for hidden_layer_size in hidden_layer_sizes:
            encoder_layers.append(tf.keras.layers.Dense(hidden_layer_size, activation="relu"))

        decoder_layers = []
        decoder_layers.append(tf.keras.layers.InputLayer(shape=(5,)))

        reversed_hidden_layer_sizes = list(hidden_layer_sizes)
        reversed_hidden_layer_sizes.reverse()

        for hidden_layer_size in reversed_hidden_layer_sizes:
            decoder_layers.append(tf.keras.layers.Dense(hidden_layer_size, activation="relu"))

        decoder_layers.append(tf.keras.layers.Dense(output_dimension))

        self.encoder_network = tf.keras.Sequential(encoder_layers)
        self.latent_mean_layer = tf.keras.layers.Dense(5)
        self.latent_log_variance_layer = tf.keras.layers.Dense(5)
        self.sampling_layer = SamplingLayer()
        self.decoder_network = tf.keras.Sequential(decoder_layers)
        self.parameter_bounder = Fmm1ParameterBounder()
        self.wave_layer = Fmm1WaveLayer(time_points)

    def encode(self, sensor_values: tf.Tensor, training: bool = False) -> Tuple[tf.Tensor, tf.Tensor]:
        hidden_values = self.encoder_network(sensor_values, training=training)

        latent_mean = self.latent_mean_layer(hidden_values)
        latent_log_variance = self.latent_log_variance_layer(hidden_values)

        latent_log_variance = tf.clip_by_value(
            latent_log_variance,
            -6.0,
            6.0,
        )

        return latent_mean, latent_log_variance

    def call(self, sensor_values: tf.Tensor, training: bool = False) -> Tuple[
        tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        latent_mean, latent_log_variance = self.encode(
            sensor_values,
            training=training,
        )

        if training:
            latent_vector = self.sampling_layer(
                (
                    latent_mean,
                    latent_log_variance,
                )
            )
        else:
            latent_vector = latent_mean

        reconstructed_sensor_values = self.decoder_network(
            latent_vector,
            training=training,
        )

        fmm_parameters = self.parameter_bounder(latent_vector)
        fmm_waves = self.wave_layer(fmm_parameters)

        return reconstructed_sensor_values, latent_mean, latent_log_variance, latent_vector, fmm_parameters, fmm_waves


class VariationalAutoencoderTrainer:
    def __init__(self, model: SensorToFmm1VariationalAutoencoder, learning_rate: float, kl_weight: float,
                 model_name: str) -> None:
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.kl_weight = kl_weight
        self.model_name = model_name

    def calculate_losses(self, sensor_values: tf.Tensor, training: bool) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        reconstructed_sensor_values, latent_mean, latent_log_variance, latent_vector, fmm_parameters, fmm_waves = self.model(
            sensor_values,
            training=training,
        )

        reconstruction_loss = tf.reduce_mean(
            tf.square(sensor_values - reconstructed_sensor_values)
        )

        kl_loss = -0.5 * tf.reduce_mean(
            1.0
            + latent_log_variance
            - tf.square(latent_mean)
            - tf.exp(latent_log_variance)
        )

        total_loss = reconstruction_loss + self.kl_weight * kl_loss

        return total_loss, reconstruction_loss, kl_loss

    def train_step(self, sensor_values: tf.Tensor) -> Tuple[float, float, float]:
        with tf.GradientTape() as tape:
            total_loss, reconstruction_loss, kl_loss = self.calculate_losses(
                sensor_values=sensor_values,
                training=True,
            )

        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        return float(total_loss), float(reconstruction_loss), float(kl_loss)

    def create_dataset(self, sensor_values: np.ndarray, batch_size: int) -> tf.data.Dataset:
        if len(sensor_values) == 0:
            raise ValueError("Dataset is empty.")

        dataset = tf.data.Dataset.from_tensor_slices(sensor_values.astype(np.float32))
        dataset = dataset.shuffle(buffer_size=len(sensor_values))
        dataset = dataset.batch(batch_size)

        return dataset

    def train(self, sensor_values: np.ndarray, epoch_count: int, batch_size: int) -> None:
        dataset = self.create_dataset(
            sensor_values=sensor_values,
            batch_size=batch_size,
        )

        for epoch_index in range(epoch_count):
            total_losses = []
            reconstruction_losses = []
            kl_losses = []

            for batch_values in dataset:
                total_loss, reconstruction_loss, kl_loss = self.train_step(batch_values)

                total_losses.append(total_loss)
                reconstruction_losses.append(reconstruction_loss)
                kl_losses.append(kl_loss)

            print(
                self.model_name,
                "epoch =",
                epoch_index + 1,
                "total_loss =",
                float(np.mean(total_losses)),
                "reconstruction_mse =",
                float(np.mean(reconstruction_losses)),
                "kl =",
                float(np.mean(kl_losses)),
            )


class FmmWaveExtractor:
    def extract(self, model: SensorToFmm1VariationalAutoencoder, normalized_values: np.ndarray) -> Tuple[
        np.ndarray, np.ndarray]:
        tensor_values = tf.convert_to_tensor(normalized_values.astype(np.float32))

        reconstructed_sensor_values, latent_mean, latent_log_variance, latent_vector, fmm_parameters, fmm_waves = model(
            tensor_values,
            training=False,
        )

        return fmm_parameters.numpy(), fmm_waves.numpy()


class ConsecutiveWaveAssembler:
    def __init__(self, internal_time_points: np.ndarray, period_length: float) -> None:
        self.internal_time_points = internal_time_points
        self.period_length = period_length

    def create_long_time_axis(self, wave_count: int) -> np.ndarray:
        time_parts = []

        for wave_index in range(wave_count):
            shifted_time_points = self.internal_time_points + wave_index * self.period_length
            time_parts.append(shifted_time_points)

        long_time_axis = np.concatenate(time_parts)

        return long_time_axis

    def create_long_wave(self, waves: np.ndarray) -> np.ndarray:
        wave_parts = []

        for wave_index in range(waves.shape[0]):
            wave_parts.append(waves[wave_index])

        long_wave = np.concatenate(wave_parts)

        return long_wave


class ParameterReporter:
    def report(self, name: str, parameters: np.ndarray) -> None:
        print("")
        print(name, "selected FMM parameters")
        print("M, A, alpha, beta, omega")
        print(parameters)


class GpsLidarFmm3DPlotter:
    def __init__(self, period_length: float, vector_count: int) -> None:
        self.period_length = period_length
        self.vector_count = vector_count
        self.display_height = 2.0

    def calculate_maximum_absolute_value(self, gps_wave: np.ndarray, lidar_wave: np.ndarray) -> float:
        gps_maximum = float(np.max(np.abs(gps_wave)))
        lidar_maximum = float(np.max(np.abs(lidar_wave)))

        maximum_absolute_value = max(gps_maximum, lidar_maximum)

        if maximum_absolute_value < 1e-8:
            maximum_absolute_value = 1.0

        return maximum_absolute_value

    def scale_wave(self, wave: np.ndarray, maximum_absolute_value: float) -> np.ndarray:
        scaled_wave = wave / maximum_absolute_value
        scaled_wave = scaled_wave * self.display_height

        return scaled_wave

    def plot(self, long_time_axis: np.ndarray, long_gps_wave: np.ndarray, long_lidar_wave: np.ndarray,
             wave_count: int) -> None:
        maximum_absolute_value = self.calculate_maximum_absolute_value(
            gps_wave=long_gps_wave,
            lidar_wave=long_lidar_wave,
        )

        scaled_gps_wave = self.scale_wave(
            wave=long_gps_wave,
            maximum_absolute_value=maximum_absolute_value,
        )

        scaled_lidar_wave = self.scale_wave(
            wave=long_lidar_wave,
            maximum_absolute_value=maximum_absolute_value,
        )

        display_gps_wave = -scaled_gps_wave
        display_lidar_wave = scaled_lidar_wave

        gps_x_values = long_time_axis
        gps_y_values = display_gps_wave
        gps_z_values = np.zeros_like(display_gps_wave)

        lidar_x_values = long_time_axis
        lidar_y_values = np.zeros_like(display_lidar_wave)
        lidar_z_values = display_lidar_wave

        resultant_x_values = long_time_axis
        resultant_y_values = display_gps_wave
        resultant_z_values = display_lidar_wave

        figure = plt.figure(figsize=(22, 10))
        axes = figure.add_subplot(1, 1, 1, projection="3d")

        surface_time_values = np.linspace(
            float(np.min(long_time_axis)),
            float(np.max(long_time_axis)),
            160,
        )

        surface_amplitude_values = np.linspace(
            -self.display_height,
            self.display_height,
            40,
        )

        gps_time_grid, gps_amplitude_grid = np.meshgrid(
            surface_time_values,
            surface_amplitude_values,
        )

        gps_zero_grid = np.zeros_like(gps_time_grid)

        lidar_time_grid, lidar_amplitude_grid = np.meshgrid(
            surface_time_values,
            surface_amplitude_values,
        )

        lidar_zero_grid = np.zeros_like(lidar_time_grid)

        axes.plot_surface(
            gps_time_grid,
            gps_amplitude_grid,
            gps_zero_grid,
            alpha=0.08,
        )

        axes.plot_surface(
            lidar_time_grid,
            lidar_zero_grid,
            lidar_amplitude_grid,
            alpha=0.08,
        )

        axes.plot(
            gps_x_values,
            gps_y_values,
            gps_z_values,
            linewidth=2.0,
            label="GPS wave on GPS plane",
        )

        axes.plot(
            lidar_x_values,
            lidar_y_values,
            lidar_z_values,
            linewidth=2.0,
            label="LiDAR wave on LiDAR plane",
        )

        axes.plot(
            resultant_x_values,
            resultant_y_values,
            resultant_z_values,
            linewidth=2.4,
            label="resultant FMM vector wave",
        )

        axes.plot(
            [
                float(np.min(long_time_axis)),
                float(np.max(long_time_axis)),
            ],
            [
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
            ],
            linewidth=2.0,
            label="shared intersection line",
        )

        vector_indices = np.linspace(
            0,
            len(long_time_axis) - 1,
            self.vector_count,
            dtype=int,
        )

        for index in vector_indices:
            current_time = float(long_time_axis[index])
            current_gps_value = float(display_gps_wave[index])
            current_lidar_value = float(display_lidar_wave[index])

            axes.quiver(
                current_time,
                0.0,
                0.0,
                0.0,
                current_gps_value,
                0.0,
                arrow_length_ratio=0.18,
                linewidth=0.9,
            )

            axes.quiver(
                current_time,
                0.0,
                0.0,
                0.0,
                0.0,
                current_lidar_value,
                arrow_length_ratio=0.18,
                linewidth=0.9,
            )

            axes.quiver(
                current_time,
                0.0,
                0.0,
                0.0,
                current_gps_value,
                current_lidar_value,
                arrow_length_ratio=0.12,
                linewidth=1.0,
            )

        for wave_index in range(wave_count + 1):
            boundary = wave_index * self.period_length

            axes.plot(
                [
                    boundary,
                    boundary,
                ],
                [
                    -self.display_height,
                    self.display_height,
                ],
                [
                    0.0,
                    0.0,
                ],
                linestyle="--",
                linewidth=0.7,
            )

            axes.plot(
                [
                    boundary,
                    boundary,
                ],
                [
                    0.0,
                    0.0,
                ],
                [
                    -self.display_height,
                    self.display_height,
                ],
                linestyle="--",
                linewidth=0.7,
            )

        axes.set_xlabel("concatenated internal time")
        axes.set_ylabel("GPS wave value")
        axes.set_zlabel("LiDAR wave value")
        axes.set_title("GPS and LiDAR waves as perpendicular components from one shared intersection line")
        axes.legend()
        axes.grid(True)

        axes.set_ylim(
            -self.display_height,
            self.display_height,
        )

        axes.set_zlim(
            -self.display_height,
            self.display_height,
        )

        axes.view_init(
            elev=24,
            azim=-58,
        )

        figure.tight_layout()
        plt.show()


class Application:
    def __init__(self, data_row_count: int, start_index: int, selected_wave_count: int, time_point_count: int,
                 maximum_lidar_range: float, epoch_count: int, gps_batch_size: int, lidar_batch_size: int,
                 learning_rate: float, kl_weight: float, vector_count: int) -> None:
        self.data_row_count = data_row_count
        self.start_index = start_index
        self.selected_wave_count = selected_wave_count
        self.time_point_count = time_point_count
        self.maximum_lidar_range = maximum_lidar_range
        self.epoch_count = epoch_count
        self.gps_batch_size = gps_batch_size
        self.lidar_batch_size = lidar_batch_size
        self.learning_rate = learning_rate
        self.kl_weight = kl_weight
        self.vector_count = vector_count
        self.period_length = 2.0 * math.pi

    def run(self) -> None:
        loader = AlignedGpsLidarLoader(
            data_slice=slice(0, self.data_row_count),
            maximum_lidar_range=self.maximum_lidar_range,
        )

        gps_points, lidar_ranges = loader.load()

        print("Aligned GPS shape =", gps_points.shape)
        print("Aligned LiDAR shape =", lidar_ranges.shape)

        gps_standardizer = SensorStandardizer()
        lidar_standardizer = SensorStandardizer()

        normalized_gps_points = gps_standardizer.fit_transform(gps_points)
        normalized_lidar_ranges = lidar_standardizer.fit_transform(lidar_ranges)

        internal_time_points = np.linspace(
            0.0,
            self.period_length,
            self.time_point_count,
            endpoint=False,
        ).astype(np.float32)

        gps_model = SensorToFmm1VariationalAutoencoder(
            input_dimension=3,
            output_dimension=3,
            time_points=internal_time_points,
            hidden_layer_sizes=(64, 64, 32),
        )

        lidar_model = SensorToFmm1VariationalAutoencoder(
            input_dimension=720,
            output_dimension=720,
            time_points=internal_time_points,
            hidden_layer_sizes=(512, 256, 128, 64),
        )

        gps_trainer = VariationalAutoencoderTrainer(
            model=gps_model,
            learning_rate=self.learning_rate,
            kl_weight=self.kl_weight,
            model_name="GPS VAE",
        )

        lidar_trainer = VariationalAutoencoderTrainer(
            model=lidar_model,
            learning_rate=self.learning_rate,
            kl_weight=self.kl_weight,
            model_name="LiDAR VAE",
        )

        gps_trainer.train(
            sensor_values=normalized_gps_points,
            epoch_count=self.epoch_count,
            batch_size=self.gps_batch_size,
        )

        lidar_trainer.train(
            sensor_values=normalized_lidar_ranges,
            epoch_count=self.epoch_count,
            batch_size=self.lidar_batch_size,
        )

        selector = ConsecutiveWindowSelector(
            start_index=self.start_index,
            window_size=self.selected_wave_count,
        )

        selected_normalized_gps_points, selected_normalized_lidar_ranges = selector.select(
            gps_points=normalized_gps_points,
            lidar_ranges=normalized_lidar_ranges,
        )

        extractor = FmmWaveExtractor()

        gps_parameters, gps_waves = extractor.extract(
            model=gps_model,
            normalized_values=selected_normalized_gps_points,
        )

        lidar_parameters, lidar_waves = extractor.extract(
            model=lidar_model,
            normalized_values=selected_normalized_lidar_ranges,
        )

        assembler = ConsecutiveWaveAssembler(
            internal_time_points=internal_time_points,
            period_length=self.period_length,
        )

        long_time_axis = assembler.create_long_time_axis(
            wave_count=self.selected_wave_count,
        )

        long_gps_wave = assembler.create_long_wave(
            waves=gps_waves,
        )

        long_lidar_wave = assembler.create_long_wave(
            waves=lidar_waves,
        )

        reporter = ParameterReporter()

        reporter.report(
            name="GPS",
            parameters=gps_parameters,
        )

        reporter.report(
            name="LiDAR",
            parameters=lidar_parameters,
        )

        plotter = GpsLidarFmm3DPlotter(
            period_length=self.period_length,
            vector_count=self.vector_count,
        )

        plotter.plot(
            long_time_axis=long_time_axis,
            long_gps_wave=long_gps_wave,
            long_lidar_wave=long_lidar_wave,
            wave_count=self.selected_wave_count,
        )


if __name__ == "__main__":
    application = Application(
        data_row_count=50000,
        start_index=0,
        selected_wave_count=10,
        time_point_count=1000,
        maximum_lidar_range=4.0,
        epoch_count=15,
        gps_batch_size=2048,
        lidar_batch_size=512,
        learning_rate=0.001,
        kl_weight=0.0001,
        vector_count=12,
    )

    application.run()