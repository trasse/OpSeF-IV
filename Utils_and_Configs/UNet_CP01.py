"""
Authors:
original cellprofiler plugin Tim Becker, Juan Caicedo, Claire McQuinn
unet_shape_resize workaround for odd pixel dimensions: Eric Czech
Modifications to run as standalone code, Keras API upgrade: Volker Hilsenstein
Also see:
https://github.com/CellProfiler/CellProfiler-plugins/issues/65
https://github.com/jr0th/segmentation
https://github.com/carpenterlab/unet4nuclei
License: BSD-3, see LICENSE.md for details
"""

import numpy as np
from skimage import transform
import keras


def test_run(img):
    input_shape = img.shape
    model = unet_initialize(input_shape)
    result = unet_classify(model, img)
    return (result)


def unet_initialize(input_shape, option_dict_conv, option_dict_bn, weights_filename, automated_shape_adjustment = True):
    unet_shape = unet_shape_resize(input_shape, 3)
    if input_shape != unet_shape and not automated_shape_adjustment:
        raise ValueError(
            f"Shape {input_shape} not compatible with 3 max-pool layers. Consider setting automated_shape_adjustment=True.")

    # create model
    dim1, dim2 = unet_shape
    model = get_model_3_class(dim1, dim2, option_dict_conv, option_dict_bn)

    # load weights
    model.load_weights(weights_filename)

    return model


def unet_shape_resize(shape, n_pooling_layers):
    """Resize shape for compatibility with UNet architecture

    Args:
        shape: Shape of images to be resized in format HW[D1, D2, ...] where any
            trailing dimensions after the first two are ignored
        n_pooling_layers: Number of pooling (or upsampling) layers in network
    Returns:
        Shape with HW sizes transformed to nearest value acceptable by network
    """
    base = 2 ** n_pooling_layers
    rcsh = np.round(np.array(shape[:2]) / base).astype(int)
    # Combine HW axes transformation with trailing shape dimensions
    # (being careful not to return 0-length axes)
    return tuple(base * np.clip(rcsh, 1, None)) + tuple(shape[2:])


def unet_image_resize(image, n_pooling_layers):
    """Resize image for compatibility with UNet architecture

    Args:
        image: Image to be resized in format HW[D1, D2, ...] where any
            trailing dimensions after the first two are ignored
        n_pooling_layers: Number of pooling (or upsampling) layers in network
    Returns:
        Image with HW dimensions resized to nearest value acceptable by network
    Reference:
        https://github.com/CellProfiler/CellProfiler-plugins/issues/65
    """
    shape = unet_shape_resize(image.shape, n_pooling_layers)
    # Note here that the type and range of the image will either not change
    # or become float64, 0-1 (which makes no difference w/ subsequent min/max scaling)
    return image if shape == image.shape else transform.resize(
        image, shape)


def unet_classify(model, input_image, resize_to_model=True):
    dim1, dim2 = input_image.shape
    mdim1, mdim2 = model.input_shape[1:3]
    needs_resize = False if (dim1, dim2) == (mdim1, mdim2) else True
    if needs_resize:
        if resize_to_model:
            input_image = transform.resize(input_image, (mdim1, mdim2))  # , anti_aliasing=True)
        else:
            raise ValueError("image size does not match model size, set resize_to_model=True")
    images = input_image.reshape((-1, mdim1, mdim2, 1))

    # scale min, max to [0.0,1.0]
    images = images.astype(np.float32)
    images = images - np.min(images)
    images = images.astype(np.float32) / np.max(images)

    pixel_classification = model.predict(images, batch_size=1)

    retval = pixel_classification[0, :, :, :]
    if needs_resize:
        retval = transform.resize(retval, (dim1, dim2, retval.shape[2]))
    return retval


def get_core(dim1, dim2, option_dict_conv, option_dict_bn):
    x = keras.layers.Input(shape=(dim1, dim2, 1))

    a = keras.layers.Conv2D(64, (3, 3), **option_dict_conv)(x)
    a = keras.layers.BatchNormalization(**option_dict_bn)(a)

    a = keras.layers.Conv2D(64, (3, 3), **option_dict_conv)(a)
    a = keras.layers.BatchNormalization(**option_dict_bn)(a)

    y = keras.layers.MaxPooling2D()(a)

    b = keras.layers.Conv2D(128, (3, 3), **option_dict_conv)(y)
    b = keras.layers.BatchNormalization(**option_dict_bn)(b)

    b = keras.layers.Conv2D(128, (3, 3), **option_dict_conv)(b)
    b = keras.layers.BatchNormalization(**option_dict_bn)(b)

    y = keras.layers.MaxPooling2D()(b)

    c = keras.layers.Conv2D(256, (3, 3), **option_dict_conv)(y)
    c = keras.layers.BatchNormalization(**option_dict_bn)(c)

    c = keras.layers.Conv2D(256, (3, 3), **option_dict_conv)(c)
    c = keras.layers.BatchNormalization(**option_dict_bn)(c)

    y = keras.layers.MaxPooling2D()(c)

    d = keras.layers.Conv2D(512, (3, 3), **option_dict_conv)(y)
    d = keras.layers.BatchNormalization(**option_dict_bn)(d)

    d = keras.layers.Conv2D(512, (3, 3), **option_dict_conv)(d)
    d = keras.layers.BatchNormalization(**option_dict_bn)(d)

    # UP

    d = keras.layers.UpSampling2D()(d)
    y = keras.layers.merge.concatenate([d, c], axis=3)

    e = keras.layers.Conv2D(256, (3, 3), **option_dict_conv)(y)
    e = keras.layers.BatchNormalization(**option_dict_bn)(e)

    e = keras.layers.Conv2D(256, (3, 3), **option_dict_conv)(e)
    e = keras.layers.BatchNormalization(**option_dict_bn)(e)

    e = keras.layers.UpSampling2D()(e)

    y = keras.layers.merge.concatenate([e, b], axis=3)

    f = keras.layers.Conv2D(128, (3, 3), **option_dict_conv)(y)
    f = keras.layers.BatchNormalization(**option_dict_bn)(f)

    f = keras.layers.Conv2D(128, (3, 3), **option_dict_conv)(f)
    f = keras.layers.BatchNormalization(**option_dict_bn)(f)

    f = keras.layers.UpSampling2D()(f)

    y = keras.layers.merge.concatenate([f, a], axis=3)

    y = keras.layers.Conv2D(64, (3, 3), **option_dict_conv)(y)
    y = keras.layers.BatchNormalization(**option_dict_bn)(y)

    y = keras.layers.Conv2D(64, (3, 3), **option_dict_conv)(y)
    y = keras.layers.BatchNormalization(**option_dict_bn)(y)

    return [x, y]


def get_model_3_class(dim1, dim2, option_dict_conv, option_dict_bn, activation="softmax"):
    [x, y] = get_core(dim1, dim2, option_dict_conv, option_dict_bn)

    y = keras.layers.Conv2D(3, (1, 1), **option_dict_conv)(y)

    if activation is not None:
        y = keras.layers.Activation(activation)(y)

    model = keras.models.Model(x, y)

    return model