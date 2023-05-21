from __future__ import division
import tensorflow as tf

def smooth_L1_loss(y_true, y_pred):
    absolute_loss = tf.abs(y_true - y_pred)
    square_loss = 0.5 * (y_true - y_pred) ** 2
    l1_loss = tf.where(tf.less(absolute_loss, 1.0), square_loss, absolute_loss - 0.5)
    return tf.reduce_sum(l1_loss, axis=-1)

def log_loss(y_true, y_pred):
    y_pred = tf.maximum(y_pred, 1e-15)
    log_loss = -tf.reduce_sum(y_true * tf.math.log(y_pred), axis=-1)
    return log_loss

def ssd_loss(y_true, y_pred):
    neg_pos_ratio = 3
    n_neg_min = 0
    alpha = 1.0
    neg_pos_ratio = tf.convert_to_tensor(neg_pos_ratio)
    n_neg_min = tf.convert_to_tensor(n_neg_min)
    alpha = tf.convert_to_tensor(alpha)

    batch_size = tf.shape(y_pred)[0]
    n_boxes = tf.shape(y_pred)[1]

    classification_loss = tf.cast(log_loss(y_true[:, :, :-4], y_pred[:, :, :-4]), tf.float32)
    localization_loss = tf.cast(smooth_L1_loss(y_true[:, :, -4:], y_pred[:, :, -4:]), tf.float32)

    negatives = y_true[:, :, 0]
    positives = tf.cast(tf.reduce_max(y_true[:, :, 1:-4], axis=-1), tf.float32)

    n_positive = tf.reduce_sum(positives)

    pos_class_loss = tf.reduce_sum(classification_loss * positives, axis=-1)

    neg_class_loss_all = classification_loss * negatives
    n_neg_losses = tf.math.count_nonzero(neg_class_loss_all, dtype=tf.int32)

    n_negative_keep = tf.minimum(tf.maximum(neg_pos_ratio * tf.cast(n_positive, tf.int32), n_neg_min),
                                 n_neg_losses)

    def f1():
        return tf.zeros([batch_size])

    def f2():
        neg_class_loss_all_1D = tf.reshape(neg_class_loss_all, [-1])

        values, indices = tf.nn.top_k(neg_class_loss_all_1D,
                                      k=n_negative_keep,
                                      sorted=False)

        negatives_keep = tf.scatter_nd(indices=tf.expand_dims(indices, axis=1),
                                       updates=tf.ones_like(indices, dtype=tf.int32),
                                       shape=tf.shape(neg_class_loss_all_1D))
        negatives_keep = tf.cast(tf.reshape(negatives_keep, [batch_size, n_boxes]), tf.float32)

        neg_class_loss = tf.reduce_sum(classification_loss * negatives_keep, axis=-1)
        return neg_class_loss

    neg_class_loss = tf.cond(tf.equal(n_neg_losses, tf.constant(0)), f1, f2)

    class_loss = pos_class_loss + neg_class_loss

    loc_loss = tf.reduce_sum(localization_loss * positives, axis=-1)

    total_loss = (class_loss + alpha * loc_loss) / tf.maximum(1.0, n_positive)

    total_loss = total_loss * tf.cast(batch_size, tf.float32)

    return total_loss
