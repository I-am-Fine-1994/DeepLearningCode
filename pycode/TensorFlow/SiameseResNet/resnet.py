import tensroflow as tf

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

def batch_norm_relu(inputs, is_training, data_format):
    """Perform a batch normalization followed by a ReLU"""
    inputs = tf.layers.batch_normalization(
        inputs=inputs, axis=1 if data_format == 'channels_first' else 3,
        momentum=_BATCH_NORM_DECAY, epsilon=_BATCH_NORM_EPSILON, center=True,
        scale=True, training=is_training, fused=True)

def fixed_padding(inputs, kernel_size):
    pad_total = kernel_size - 1
    pad_beg = pad_total // 2
    pad_end = pad_total - pad_beg
    
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end], 
                                    [pad_beg, pad_end], [0, 0]])
    return padded_inputs

def building_block(inputs, filters, is_training, projection_shortcut,
                   strides, data_format):
    """
    Standard building block for residual networks with BN before conv
    Return:
        The output tensor of the block
    """

    shortcut = inputs
    inputs = batch_norm_relu(inputs, is_training, data_format)

    if projection_shortcut is not None:
