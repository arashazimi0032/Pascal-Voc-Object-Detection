import tensorflow as tf


class BooleanMask(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BooleanMask, self).__init__(**kwargs)

    def call(self, inputs):
        org_input, masks = inputs
        boolean_mask = tf.cast(tf.not_equal(masks, 0), org_input.dtype)
        matmul = org_input * boolean_mask
        return matmul
