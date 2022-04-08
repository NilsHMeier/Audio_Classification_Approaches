import numpy as np
import tensorflow as tf


class WaveAugmentation:

    def __init__(self, noise_scale: float = 0.05, factor_scale: float = 0.1):
        self.noise_scale = noise_scale
        self.factor_scale = factor_scale

    def apply_noise_np(self, audio_signal: np.ndarray) -> np.ndarray:
        return audio_signal + np.random.normal(loc=0, scale=self.noise_scale * np.max(audio_signal),
                                               size=audio_signal.shape)

    def apply_noise_tf(self, tensor: tf.Tensor) -> tf.Tensor:
        return tf.add(tensor, tf.random.normal(shape=tensor.shape, mean=0.0,
                                               stddev=self.noise_scale * tf.reduce_max(tensor), dtype=tensor.dtype))

    def apply_factor_np(self, audio_signal: np.ndarray) -> np.ndarray:
        return audio_signal * np.random.normal(loc=1, scale=self.factor_scale)

    def apply_factor_tf(self, tensor: tf.Tensor) -> tf.Tensor:
        return tf.multiply(tensor, tf.random.normal(shape=[1], mean=1.0, stddev=self.factor_scale, dtype=tensor.dtype))

    def apply_both_np(self, audio_signal: np.ndarray) -> np.ndarray:
        return audio_signal * np.random.normal(loc=1, scale=self.factor_scale) + \
               np.random.normal(loc=0, scale=self.noise_scale * np.max(audio_signal), size=audio_signal.shape)

    def apply_both_tf(self, tensor: tf.Tensor) -> tf.Tensor:
        return tf.add(tf.multiply(tensor, tf.random.normal(shape=[1], mean=1.0, stddev=self.factor_scale,
                                                           dtype=tensor.dtype)),
                      tf.random.normal(shape=tensor.shape, mean=0.0, stddev=self.noise_scale * tf.reduce_max(tensor),
                                       dtype=tensor.dtype))


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
