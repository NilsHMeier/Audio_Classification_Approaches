import numpy as np
import tensorflow as tf


class WaveAugmentation:

    def __init__(self, scale: float = 0.05):
        self.scale = scale

    def apply_noise_np(self, audio_signal: np.ndarray) -> np.ndarray:
        return audio_signal + np.random.normal(loc=0, scale=self.scale, size=audio_signal.shape)

    def apply_noise_tf(self, tensor: tf.Tensor) -> tf.Tensor:
        return tf.add(tensor, tf.random.normal(shape=tensor.shape, mean=0.0, stddev=self.scale, dtype=tensor.dtype))

    def apply_factor_np(self, audio_signal: np.ndarray) -> np.ndarray:
        return audio_signal * np.random.normal(loc=1, scale=self.scale)

    def apply_factor_tf(self, tensor: tf.Tensor) -> tf.Tensor:
        return tf.multiply(tensor, tf.random.normal(shape=[1], mean=1.0, stddev=self.scale, dtype=tensor.dtype))


class SpectrogramAugmentation:

    def __init__(self, percentage: float = 0.2):
        self.percentage = percentage

    def apply_np(self, spectrogram: np.ndarray) -> np.ndarray:
        return spectrogram * np.random.choice(a=[True, False], p=[1 - self.percentage, self.percentage],
                                              size=spectrogram.shape)

    def apply_tf(self, tensor: tf.Tensor) -> tf.Tensor:
        return tf.multiply(tensor, tf.cast(tf.reshape(tf.random.categorical(
            tf.math.log([[self.percentage, 1 - self.percentage]]), num_samples=tf.math.reduce_prod(tensor.shape)),
            shape=tensor.shape), dtype=tensor.dtype))
