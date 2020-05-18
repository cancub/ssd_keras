'''
A Keras port of the original Caffe SSD300 network.

Copyright (C) 2018 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''
import tensorflow as tf

from __future__ import division
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import (Concatenate, Conv2D, Dropout, Flatten,
    Lambda, MaxPooling2D, Reshape, Softmax, ZeroPadding2D)
from tensorflow.keras.regularizers import l2
import numpy as np

from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_L2Normalization import L2Normalization
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast


def ssd_300(image_shape,
            n_classes,
            mode='training',
            l2_reg=0.0005,
            min_scale=None,
            max_scale=None,
            scales=None,
            aspect_ratios_global=None,
            aspect_ratios_per_layer=[[1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5, 3.0, 1.0/3.0],
                                     [1.0, 2.0, 0.5],
                                     [1.0, 2.0, 0.5]],
            two_boxes_for_ar1=True,
            steps=[8, 16, 32, 64, 100, 300],
            offsets=None,
            clip_boxes=False,
            variances=[0.1, 0.1, 0.2, 0.2],
            coords='centroids',
            normalize_coords=True,
            confidence_thresh=0.01,
            iou_threshold=0.45,
            top_k=200,
            nms_max_output_size=400,
            return_predictor_sizes=False):
    '''
    Build a Keras model with SSD300 architecture, see references.

    The base network is a reduced atrous VGG-16, extended by the SSD
    architecture, as described in the paper.

    Most of the arguments that this function takes are only needed for the
    anchor box layers. In case you're training the network, the parameters
    passed here must be the same as the ones used to set up `SSDBoxEncoder`. In
    case you're loading trained weights, the parameters passed here must be the
    same as the ones used to produce the trained weights.

    Some of these arguments are explained in more detail in the documentation of
    the `SSDBoxEncoder` class.

    NOTE: Requires TensorFlow 2 or later.

    Required Arguments:

        image_shape (tuple):
            The input image size in the format `(height, width, channels)`.

        n_classes (int):
            The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS
            COCO.

    Optional Arguments:

        mode ('training', 'inference', 'inference_fast'):
            In 'training' mode, the model outputs the raw prediction tensor,
            while in 'inference' and 'inference_fast' modes, the raw predictions
            are decoded into absolute coordinates and filtered via confidence
            thresholding, non-maximum suppression, and top-k filtering. The
            difference between latter two modes is that 'inference' follows the
            exact procedure of the original Caffe implementation, while
            'inference_fast' uses a faster prediction decoding procedure.

        l2_reg (float):
            The L2-regularization rate.
            Applies to all convolutional layers. Set to zero to deactivate
            L2-regularization.

        min_scale (float):
            The smallest scaling factor for the size of the anchor boxes as a
            fraction of the shorter side of the input images.

        max_scale (float):
            The largest scaling factor for the size of the anchor boxes as a
            fraction of the shorter side of the input images.
            All scaling factors between the smallest and the largest will be
            linearly interpolated.
            NOTE: the second to last of the linearly interpolated scaling
                  factors will actually be the scaling factor for the last
                  predictor layer, while the last scaling factor is used for the
                  second box for aspect ratio 1 in the last predictor layer if
                  `two_boxes_for_ar1` is `True`.

        scales (list):
            A list of floats containing scaling factors per convolutional
            predictor layer.
            This list must be one element longer than the number of predictor
            layers. The first `k` elements are the scaling factors for the `k`
            predictor layers, while the last element is used for the second box
            for aspect ratio 1 in the last predictor layer if
            `two_boxes_for_ar1` is `True`. This additional last scaling factor
            must be passed either way, even if it is not being used. If a list
            is passed, this argument overrides `min_scale` and `max_scale`. All
            scaling factors must be greater than zero.

        aspect_ratios_global (list):
            The list of aspect ratios with which anchor boxes will be generated.
            This list is valid for all prediction layers.

        aspect_ratios_per_layer (list):
            A list containing one aspect ratio list for each prediction layer.
            This allows you to set the aspect ratios for each predictor layer
            individually, which is the case for the original SSD300
            implementation. If a list is passed, it overrides
            `aspect_ratios_global`.
            NOTE: len(`aspect_ratios_per_layer`) == len(prediction layers) == 6


        two_boxes_for_ar1 (bool):
            If `True`, two anchor boxes will be generated for aspect ratio 1.
            The first will be generated using the scaling factor for the
            respective layer, the second one will be generated using geometric
            mean of said scaling factor and next bigger scaling factor.
            NOTE: Only relevant for aspect ratio lists that contain 1. Will be
                  ignored otherwise.

        steps (list):
            [`None`, <list>]
            The elements can be either ints/floats or tuples of two ints/floats.
            These numbers represent for each predictor layer how many pixels
            apart the anchor box center points should be vertically and
            horizontally along the spatial grid over the image. If the list
            contains ints/floats, then that value will be used for both spatial
            dimensions. If the list contains tuples of two ints/floats, then
            they represent `(step_height, step_width)`. If no steps are
            provided, then they will be computed such that the anchor box center
            points will form an equidistant grid within the image dimensions.
            NOTE: len(`steps`) == len(prediction layers) == 6

        offsets (list):
            [`None`, <list>]
            The elements can be either floats or tuples of two floats.
            These numbers represent for each predictor layer how many pixels
            from the top and left boarders of the image the top-most and
            left-most anchor box center points should be as a fraction of
            `steps`. The last bit is important: The offsets are not absolute
            pixel values, but fractions of the step size specified in the
            `steps` argument. If the list contains floats, then that value will
            be used for both spatial dimensions. If the list contains tuples of
            two floats, then they represent
            `(vertical_offset, horizontal_offset)`. If no offsets are provided,
            then they will default to 0.5 of the step size.
            NOTE: len(`offsets`) == len(prediction layers) == 6

        clip_boxes (bool):
            If `True`, clips the anchor box coordinates to stay within image
            boundaries.

        variances (list):
            A list of 4 floats >0.
            The anchor box offset for each coordinate will be divided by its
            respective variance value.

        coords (str):
            The box coordinate format to be used internally by the model.
            Can be either 'centroids' for the format
                `(cx, cy, w, h)` (box center coordinates, width, and height),
                'minmax' for the format `(xmin, xmax, ymin, ymax)`, or
                'corners' for the format `(xmin, ymin, xmax, ymax)`.
            NOTE: this is _not_ the input format of the ground truth labels.

        normalize_coords (bool):
            Set to `True` if the model is supposed to use relative instead of
            absolute coordinates, i.e., if the model predicts box coordinates
            within [0,1] instead of absolute coordinates.

        confidence_thresh (float):
            A float in [0,1), the minimum classification confidence in a
            specific positive class in order to be considered for the
            non-maximum suppression stage for the respective class. A lower
            value will result in a larger part of the selection process being
            done by the non-maximum suppression stage, while a larger value will
            result in a larger part of the selection process happening in the
            confidence thresholding stage.

        iou_threshold (float in [0,1]):
            All boxes that have a Jaccard similarity of greater than
            `iou_threshold` with a locally maximal box will be removed from the
            set of predictions for a given class, where 'maximal' refers to the
            box's confidence score.

        top_k (int):
            The number of highest scoring predictions to be kept for each batch
            item after the non-maximum suppression stage.

        nms_max_output_size (int):
            The maximal number of predictions that will be left over after the
            NMS stage.

        return_predictor_sizes (bool):
            If `True`, this function not only returns the model, but also a list
            containing the spatial dimensions of the predictor layers. This
            isn't strictly necessary since you can always get their sizes easily
            via the Keras API, but it's convenient and less error-prone to get
            them this way. They are only relevant for training anyway
            (SSDBoxEncoder needs to know the spatial dimensions of the predictor
            layers), for inference you don't need them.


    Returns:

        model:
            The Keras SSD300 model.

        predictor_sizes (optional):
            A Numpy array containing the `(height, width)` portion of the output
            tensor shape for each convolutional predictor layer. During
            training, the generator function needs this in order to transform
            the ground truth labels into tensors of identical structure as the
            output tensors of the model, which is in turn needed for the cost
            function.

    References:
        https://arxiv.org/abs/1512.02325v5
    '''

    img_height, img_width, img_channels = image_shape
    # The number of predictor convolutional layers in the network is 6 for the
    # original SSD300.
    n_predictor_layers = 6
    # Account for the background class.
    n_classes += 1

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################

    if aspect_ratios_per_layer is None:
        if aspect_ratios_global is None:
            raise ValueError(
                ('Either `aspect_ratios_global` or `aspect_ratios_per_layer` '
                 'must be specified. Received None for both.')
            )
    elif len(aspect_ratios_per_layer) != n_predictor_layers:
        raise ValueError(
            ('Length of `aspect_ratios_per_layer` ({}) does not match number '
             'of prediction layers ({}).').format(
                len(aspect_ratios_per_layer), n_predictor_layers)
        )

    if scales is None:
        if min_scale is None and max_scale is None:
            raise ValueError(
                ('No values specified for scales (`scales`, `min_scale`, '
                 '`max_scale`).')
            )
        elif ((min_scale is None and max_scale is not None)
                or (min_scale is not None and max_scale is None)):
            raise ValueError('`min_scale` AND `max_scale` must be specified.')
        else:
            # Compute the list of scaling factors from the specified `min_scale`
            # and `max_scale`
            scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)
    elif len(scales) != n_predictor_layers+1:
        raise ValueError(
            ('Length of `scales` ({}) does not match number of prediction '
             ' layers + 1 ({}).').format(len(scales), n_predictor_layers+1)
        )

    if len(variances) != 4:
        raise ValueError(
            ('Four (4) variance values must be provided, but {} were '
             'received.').format(len(variances))
        )
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError(
            ('Found negative variances in provided list ({}). All variances '
             'must be positive (>0)').format(variances)
        )

    if steps is not None and len(steps) != n_predictor_layers:
        raise ValueError(
            ('Length of `steps` ({}) does not match the number of prediction '
             'layers ({}).').format(len(steps), n_predictor_layers)
        )

    if offsets is not None and len(offsets) != n_predictor_layers:
        raise ValueError(
            ('Length of `offsets` ({}) does not match the number of prediction '
             'layers ({}).').format(len(offsets), n_predictor_layers)
        )

    if mode not in ('training', 'inference', 'inference_fast'):
        raise ValueError(
            ('`mode` must be one of "training", "inference" or '
             '"inference_fast", but received "{}".').format(mode)
        )

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for
    # the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor
    # layer. We need this so that we know how many channels the predictor layers
    # need to have.
    # NOTE: +1 for the second box for aspect ratio 1
    n_boxes = [len(ar) + (1 if two_boxes_for_ar1 and 1 in ar else 0)
               for ar in aspect_ratios]

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    ############################################################################
    # Build the network.
    ############################################################################

    def build_Conv(filters, size=3, strides=(1, 1), padding='same',
                   d_rate=(1, 1), activation='relu', k_init='he_normal',
                   k_reg=l2(l2_reg), input_layer=False):
        return Conv2D(
            filters,
            size,
            strides=strides,
            padding=padding,
            dilation_rate=d_rate,
            activation=activation,
            kernel_initializer=k_init,
            kernel_regularizer=k_reg,
            input_shape=image_shape if input_layer else None
        )

    def build_MaxPool(pool_size=2, strides=None, padding='same'):
        return MaxPooling2D(
            pool_size=pool_size, strides=strides, padding=padding)

    def build_AnchorBoxes(index):
        return AnchorBoxes(
            img_height, img_width,
            this_scale=scales[index], next_scale=scales[index+1],
            aspect_ratios=aspect_ratios[index],
            two_boxes_for_ar1=two_boxes_for_ar1,
            this_steps=steps[index],
            this_offsets=offsets[index],
            clip_boxes=clip_boxes,
            variances=variances,
            coords=coords,
            normalize_coords=normalize_coords,
        )

    detector_sources = OrderedDict()

    x = Input(shape=(img_height, img_width, img_channels))

    conv1_1 = build_Conv(64)(x)
    conv1_2 = build_Conv(64)(conv1_1)
    pool1 = build_MaxPool(conv1_2)

    conv2_1 = build_Conv(128)(pool1)
    conv2_2 = build_Conv(128)(conv2_1)
    pool2 = build_MaxPool(conv2_2)

    conv3_1 = build_Conv(256)(pool2)
    conv3_2 = build_Conv(256)(conv3_1)
    conv3_3 = build_Conv(256)(conv3_2)
    pool3 = build_MaxPool(conv3_3)

    conv4_1 = build_Conv(512)(pool3)
    conv4_2 = build_Conv(512)(conv4_1)
    conv4_3 = build_Conv(512)(conv4_2)
    pool4 = build_MaxPool(conv4_3)
    detector_sources['conv4_3'] = conv4_3

    conv5_1 = build_Conv(512)(pool4)
    conv5_2 = build_Conv(512)(conv5_1)
    conv5_3 = build_Conv(512)(conv5_2)
    pool5 = build_MaxPool(pool_size=(3, 3), strides=(1, 1))(conv5_3)

    conv6 = build_Conv(1024, d_rate=(6, 6))(pool5)

    conv7 = build_Conv(1024, size=1)(conv6)
    detector_sources['conv7'] = conv7

    conv8_1 = build_Conv(256, size=1)(conv7)
    conv8_1 = ZeroPadding2D(padding=((1, 1), (1, 1)))(conv8_1)
    conv8_2 = build_Conv(512, strides=(2, 2), padding='valid')(conv8_1)
    detector_sources['conv8_2'] = conv8_2

    conv9_1 = build_Conv(128, size=1)(conv8_2)
    conv9_1 = ZeroPadding2D(padding=((1, 1), (1, 1)))(conv9_1)
    conv9_2 = build_Conv(256, strides=(2, 2), padding='valid')(conv9_1)
    detector_sources['conv9_2'] = conv9_2

    conv10_1 = build_Conv(128, size=1)(conv9_2)
    conv10_2 = build_Conv(256, strides=(1, 1), padding='valid')(conv10_1)
    detector_sources['conv10_2'] = conv10_2

    conv11_1 = build_Conv(128, size=1)(conv10_2)
    conv11_2 = build_Conv(256, strides=(1, 1), padding='valid')(conv11_1)
    detector_sources['conv11_2'] = conv11_2

    # Feed conv4_3 into the L2 normalization layer
    conv4_3_norm = L2Normalization(gamma_init=20)(conv4_3)
    detector_sources['conv4_3_norm'] = conv4_3_norm

    ### Build the convolutional predictor layers on top of the base network

    conf_outputs = []
    loc_outputs = []
    priorboxes_outputs = []

    i = 0
    for name, layer in detector_sources.items():
        conf_outputs.append(
            Reshape((-1, n_classes))(
                build_Conv(n_boxes[i] * n_classes)(layer)
            )
        )

        loc = build_Conv(n_boxes[i] * 4)(layer)

        loc_outputs.append(Reshape((-1, 4))(loc))

        priorboxes_outputs.append(
            Reshape((-1, 8))(
                build_AnchorBoxes(i)(loc)
            )
        )

    mbox_conf = Concatenate(axis=1)(conf_outputs)
    mbox_loc = Concatenate(axis=1)(loc_outputs)
    mbox_priorbox = Concatenate(axis=1)(priorboxes_outputs)

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    mbox_conf_softmax = Softmax(mbox_conf)

    # Concatenate the class and box predictions and the anchors to one large predictions vector
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2)(
        [mbox_conf_softmax, mbox_loc, mbox_priorbox])

    if 'inference' in mode:
        if mode == 'inference':
            decoder = DecodeDetections
        else:
            decoder = DecodeDetectionsFast

        predictions = decoder(
            confidence_thresh=confidence_thresh,
            iou_threshold=iou_threshold,
            top_k=top_k,
            nms_max_output_size=nms_max_output_size,
            coords=coords,
            normalize_coords=normalize_coords,
            img_height=img_height,
            img_width=img_width,
        )(predictions)

    model = Model(inputs=x, outputs=predictions)

    if return_predictor_sizes:
        predictor_sizes = np.array([conv4_3_norm_mbox_conf._keras_shape[1:3],
                                     fc7_mbox_conf._keras_shape[1:3],
                                     conv6_2_mbox_conf._keras_shape[1:3],
                                     conv7_2_mbox_conf._keras_shape[1:3],
                                     conv8_2_mbox_conf._keras_shape[1:3],
                                     conv9_2_mbox_conf._keras_shape[1:3]])
        return model, predictor_sizes
    else:
        return model
