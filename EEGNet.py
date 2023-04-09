import tensorflow as tf


def EEGNet(nb_classes, Chans = 64, Samples = 128, 
             dropoutRate = 0.5, kernLength = 64, F1 = 8, 
             D = 2, F2 = 16, norm_rate = 0.25, dropoutType = 'Dropout'):
    """ Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta
    
    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the 
    kernel lengths for double the sampling rate, etc). Note that we haven't 
    tested the model performance with this rule so this may not work well. 
    
    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
	advised to do some model searching to get optimal performance on your
	particular dataset.
    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D. 
    Inputs:
        
      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.     
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D. 
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.
    """
    
    if dropoutType == 'SpatialDropout2D':
        dropoutType = tf.keras.layers.SpatialDropout2D
    elif dropoutType == 'Dropout':
        dropoutType = tf.keras.layers.Dropout
    else:
        raise ValueError('dropoutType must be one of SpatialDropout2D '
                         'or Dropout, passed as a string.')
    
    input1   = tf.keras.layers.Input(shape = (Chans, Samples, 1))
    ##################################################################
    block1 = tf.keras.layers.Conv2D(F1,
                    (1, kernLength),
                    padding = 'same',
                    input_shape = (Chans, Samples, 1),
                    use_bias = False)(input1)
    block1 = tf.keras.layers.BatchNormalization()(block1)
    block1 = tf.keras.layers.DepthwiseConv2D((Chans, 1),
                             use_bias = False,
                             depth_multiplier = D,
                             depthwise_constraint = tf.keras.constraints.max_norm(1.))(block1)
    block1 = tf.keras.layers.BatchNormalization()(block1)
    block1 = tf.keras.layers.Activation('elu')(block1)
    block1 = tf.keras.layers.AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)
    
    block2 = tf.keras.layers.SeparableConv2D(F2,
                             (1, 16),
                             use_bias = False,
                             padding = 'same')(block1)
    block2 = tf.keras.layers.BatchNormalization()(block2)
    block2 = tf.keras.layers.Activation('elu')(block2)
    block2 = tf.keras.layers.AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)
    flatten = tf.keras.layers.Flatten(name = 'flatten')(block2)
    
    dense = tf.keras.layers.Dense(nb_classes, name = 'dense', 
                         kernel_constraint = tf.keras.constraints.max_norm(norm_rate))(flatten)
    softmax = tf.keras.layers.Activation('softmax', name = 'softmax')(dense)
    
    return tf.keras.modelsModel(inputs=input1, outputs=softmax)