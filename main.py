import cv2
import os.path
import tensorflow as tf
import numpy as np
import helper
import warnings
import movify
from distutils.version import LooseVersion
import project_tests as tests
import sys
import scipy
from matplotlib import pyplot as plt

testing = False

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion(
    '1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if testing:
    if not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #   Use tf.saved_model.loader.load to load the model and weights
    tf.saved_model.loader.load(sess, ['vgg16'], vgg_path)
    graph = tf.get_default_graph()
    return tuple(graph.get_tensor_by_name(name) for name in
                 ('image_input:0', 'keep_prob:0', 'layer3_out:0', 'layer4_out:0', 'layer7_out:0'))


if testing:
    tests.test_load_vgg(load_vgg, tf)


def layers(*args):
    """
    Create the layers for a fully convolutional network. Build skip-layers using the vgg layers.
    To do this, you should pass in the layers that you want to be used in skip layers, followed by
    the number of classes you have.

    So, for instance, if you have a CNN that looks like this (numbers denote layers)

    layer_1 -> layer_2 -> ... -> layer_7

    and you now want to turn that into a fully convolutional network using
    layers 3, 4, and 7 as skip layers, then you would call this function like

    out = layers( layer_3, layer_4, layer_7, num_classes )
    """
    deconv_options = {}
    deconv_options['padding'] = 'same'
    if not __debug__:
        # This is lazy loaded, and really slows down debugging
        # deconv_options['kernel_regularizer'] = tf.contrib.layers.l1_regularizer(scale=1e-4)
        pass
    conv_options = deconv_options.copy()
    conv_options['kernel_size'] = 1

    layers, num_classes = args[-2::-1], args[-1]
    layers4x2, layers16x8 = layers[:2], layers[2:]

    out = None
    for layer in layers4x2:
        conv = tf.layers.conv2d(layer, num_classes, **conv_options)
        if out is not None:
            out = tf.add(out, conv)
        else:
            out = conv
        out = tf.layers.conv2d_transpose(out, num_classes, kernel_size=4, strides=2, **deconv_options)
    for layer in layers16x8:
        conv = tf.layers.conv2d(layer, num_classes, **conv_options)
        out = tf.add(out, conv)
        out = tf.layers.conv2d_transpose(out, num_classes, kernel_size=16, strides=8, **deconv_options)
    return out


if testing:
    tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)
    return logits, train_op, cross_entropy_loss


if testing:
    tests.test_optimize(optimize)


def addText(image, text):
    baseline = 0
    fontFace = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = .5
    thickness = 1
    textSize = cv2.getTextSize(text, fontFace, fontScale, thickness)
    baseline += thickness
    height, width, _ = image.shape
    x = int((width - textSize[0][0]) / 2)
    y = int(height - (height - textSize[0][1]) / 2)
    cv2.putText(image, text, (x, y), fontFace, fontScale, (255, 255, 255), thickness=thickness)


def train_nn(sess,
             epochs,
             batch_size,
             get_batches_fn,
             predictions,
             train_op,
             cross_entropy_loss,
             input_image,
             correct_label,
             keep_prob,
             learning_rate,
             keep_prob_val=0.5,
             learning_rate_val=1e-3,
             model_name='model/model.ckpt',
             validation_images=None,
             validation_labels=None,
             saver=None):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # See if there is already a model and try to load it
    if saver is not None and os.path.exists(model_name + '.meta'):
        saver.restore(sess, model_name)

    valid_dict = {input_image: validation_images,
                  keep_prob: 1}

    # Now run the specified number of epochs
    for epoch in range(epochs):
        batch = 0
        for image, label in get_batches_fn(batch_size):
            batch += 1
            train_dict = {input_image: image,
                          correct_label: label,
                          keep_prob: keep_prob_val,
                          learning_rate: learning_rate_val}
            sess.run(train_op, train_dict)
            details = "Epoch = {}, Batch = {}, Loss = {:.4f}".format(epoch + 1,
                                                                     batch,
                                                                     sess.run(cross_entropy_loss, train_dict))
            print(details)

            if validation_images is not None:
                preds = sess.run(predictions, valid_dict)
                first_image = preds[0, :, :]

                # Convert the predictions to uint8
                img = 0 * np.ones((*first_image.shape, 3), dtype=np.uint8)
                img[first_image > 0] = (0, 255, 0)

                # Overlay the predictions on the original image
                overlay = cv2.addWeighted(validation_images[0, :, :], 1, img, 1, 0)
                addText(img, details)

                # Now stack everything together
                output = np.vstack((img, overlay))

                plotting = False
                if plotting:
                    plt.imshow(output)
                    plt.pause(.05)
                    plt.draw()
                yield cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

        if saver is not None:
            saver.save(sess, model_name)


if testing:
    tests.test_train_nn(train_nn)


def generateColors(num_classes):
    # Generate a list with all of the colors taking a value of 0 or 255 in the rgb fields.
    # e.g. black = (0,0,0), white = (255,255,255), magenta = (255,0,255), blue = (0,0,255)
    colors = []
    if num_classes == 2:
        colors.append((255, 0, 0))  # red
        colors.append((255, 0, 255))  # magenta
    elif num_classes == 3:
        colors.append((255, 0, 0))  # red
        colors.append((255, 0, 255))  # magenta
        colors.append((255, 0, 0))  # black
    else:
        for r in (255, 0):
            for g in (0, 255):
                for b in (0, 255):
                    colors.append((r, g, b))

    return np.array(colors)


def overlay_prediction(sess, input_image, keep_prob, predictions, image_dir, image_shape, output_dir):
    for file in os.listdir(image_dir):
        # Read the image and convert RGB
        image = scipy.misc.imresize(scipy.misc.imread(os.path.join(image_dir, file)), image_shape)

        # Now resize so the the first axis contains the number of images (only 1)
        image.shape = (1, *image.shape)

        # Compute the predicted classes
        prediction = sess.run(predictions, {input_image: image,
                                            keep_prob: 1})

        # Convert the prediction to uint8
        predicted_image = 0 * np.ones((*prediction.shape, 3), dtype=np.uint8)
        predicted_image[prediction > 0] = (0, 255, 0)

        # Overlay the predictions on the original image
        overlaid_image = cv2.addWeighted(image, 1, predicted_image, 1, 0)

        # Save the overlaid images
        scipy.misc.imsave(os.path.join(output_dir, file), overlaid_image[0])


def run(keep_prob_val, batch_size, num_classes):
    learning_rate_val = 1e-3
    num_validation_images = 2
    epochs = 10
    image_shape = (160, 576)
    image_dtype = np.uint8
    data_dir = '../../../Users/matth/Desktop/semantic/'
    if not os.path.exists(data_dir):
        data_dir = '../../../Desktop/semantic'
    runs_dir = './runs'
    if testing:
        tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    with tf.Session() as sess:

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get training and validation batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'),
                                                   image_shape,
                                                   generateColors(num_classes),
                                                   image_dtype)
        valid_batches = helper.gen_batch_function(os.path.join(data_dir, 'data_road/testing'),
                                                  image_shape,
                                                  generateColors(num_classes),
                                                  image_dtype)
        validation_images, validation_labels = next(valid_batches(num_validation_images))

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # Load VGG
        input_image, keep_prob, *vgg_layers = load_vgg(sess, vgg_path)

        # Now create an FCN on top of VGG
        output = layers(*vgg_layers, num_classes)

        # Training functions
        learning_rate = tf.placeholder(tf.float64, shape=(), name="learning_rate")
        correct_label = tf.placeholder(image_dtype, shape=(None, *image_shape, num_classes), name="correct_label")
        logits, train_op, cross_entropy_loss = optimize(output, correct_label, learning_rate, num_classes)

        # Prediction function
        predictions = tf.argmax(output, axis=3)

        # Finally, initial the
        sess.run(tf.global_variables_initializer())

        # Create a saver for training
        # saver = tf.train.Saver(max_to_keep=1)
        saver = None

        # TRAIN!!!
        post_fix = str(batch_size) + str(keep_prob_val)[1:4] + "_" + str(epochs) + "_" + str(num_classes)
        model_name = "models/model" + post_fix + "/model.ckpt"
        frame_generator = train_nn(sess,
                                   epochs,
                                   batch_size,
                                   get_batches_fn,
                                   predictions,
                                   train_op,
                                   cross_entropy_loss,
                                   input_image,
                                   model_name=model_name,
                                   correct_label=correct_label,
                                   keep_prob=keep_prob,
                                   learning_rate=learning_rate,
                                   keep_prob_val=keep_prob_val,
                                   learning_rate_val=learning_rate_val,
                                   validation_images=validation_images,
                                   saver=saver)
        moviename = "videos/movie" + post_fix + ".mp4"
        movify.process(frame_generator, output_path=moviename, fps=20)

        # Process all of the test images
        output_dir = 'images/' + post_fix
        try:
            os.makedirs(output_dir)
        except Exception as e:
            print(e)

        image_dir = os.path.join(data_dir, 'data_road/testing/image_2')
        overlay_prediction(sess,
                           input_image,
                           keep_prob,
                           predictions,
                           image_dir,
                           image_shape,
                           output_dir)

        # Now save those images into a movie
        def frame_generator():
            for file in os.listdir(output_dir):
                yield scipy.misc.imresize(scipy.misc.imread(os.path.join(output_dir, file)), image_shape)

        movify.process(frame_generator(), output_path=output_dir + ".mp4", fps=5)

    # TODO: Save inference data using helper.save_inference_samples
    #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

    # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    # with tf.device('/gpu:0'):
    for batch_size in (2, 3, 4):
        for num_classes in (3, 2):
            for keep_prob_val in (0.2, 0.5, 0.8):
                # for keep_prob_val in (0.5,):
                #     for batch_size in (2,):
                #         for num_classes in (3,):
                run(keep_prob_val, batch_size, num_classes)
