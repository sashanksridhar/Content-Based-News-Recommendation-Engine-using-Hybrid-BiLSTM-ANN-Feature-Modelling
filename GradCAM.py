from tensorflow.keras.models import Model
import tensorflow as tf
import numpy as np
import cv2
class GradCAMS:
    def __init__(self, model, classIdx, layerName=None):
        # store the model, the class index used to measure the class
        # activation map, and the layer to be used when visualizing
        # the class activation map
        self.model = model
        self.classIdx = classIdx
        self.layerName = layerName
        # if the layer name is None, attempt to automatically find
        # the target output layer
        if self.layerName is None:
            self.layerName = self.find_target_layer()


    def find_target_layer(self):
        # attempt to find the final convolutional layer in the network
        # by looping over the layers of the network in reverse order
        for layer in reversed(self.model.layers):
            # print(layer.name)
            # check to see if the layer has a 4D output
            if len(layer.output_shape) == 2 and layer.output_shape[1]!=1:
                return layer.name
        # otherwise, we could not find a 4D layer so the GradCAM
        # algorithm cannot be applied
        raise ValueError("Could not find 4D layer. Cannot apply GradCAM.")

    def compute_heatmap(self, image, eps=1e-8):
        # construct our gradient model by supplying (1) the inputs
        # to our pre-trained model, (2) the output of the (presumably)
        # final 4D layer in the network, and (3) the output of the
        # softmax activations from the model
        gradModel = Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layerName).output,
                     self.model.output])
        # record operations for automatic differentiation
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            # cast the image tensor to a float-32 data type, pass the
            # image through the gradient model, and grab the loss
            # associated with the specific class index
            input1 = tf.cast(image[0], tf.float32)
            input2 = tf.cast(image[1], tf.float32)
            input3 = tf.cast(image[2], tf.float32)
            input4 = tf.cast(image[3], tf.float32)
            tape.watch(input1)
            tape.watch(input2)
            tape.watch(input3)
            tape.watch(input4)

            convOutputs, predictions = gradModel([input1,input2,input2,input3])
            # (convOutputs, predictions) = gradModel([input1, input2, input3, input4])

            loss = predictions[self.classIdx]


        # use automatic differentiation to compute the gradients

        grads = tape.gradient(loss, convOutputs)
        # print(grads)
        # compute the guided gradients
        castConvOutputs = tf.cast(convOutputs > 0, "float32")
        # print(convOutputs)
        castGrads = tf.cast(grads > 0, "float32")
        # print(castGrads)
        guidedGrads = castConvOutputs * castGrads * grads
        # the convolution and guided gradients have a batch dimension
        # (which we don't need) so let's grab the volume itself and
        # discard the batch
        # print("Guided")
        # print(guidedGrads)
        convOutputs = convOutputs[0]
        guidedGrads = guidedGrads[0]
        # compute the average of the gradient values, and using them
        # as weights, compute the ponderation of the filters with
        # respect to the weights
        # print(guidedGrads)
        weights = tf.reduce_mean(guidedGrads, axis=(0, 0))
        # print("weights")
        # print(weights)
        # cam = tf.reduce_sum(tf.multiply(weights, convOutputs), axis=-1)
        cam = tf.multiply(weights,convOutputs)
        # print(cam)
        # print(cam)
        return cam.numpy()
    #     # grab the spatial dimensions of the input image and resize
    #     # the output class activation map to match the input image
    #     # dimensions
    #     (w, h) = (image.shape[2], image.shape[1])
    #     heatmap = cv2.resize(cam.numpy(), (w, h))
    #     # normalize the heatmap such that all values lie in the range
    #     # [0, 1], scale the resulting values to the range [0, 255],
    #     # and then convert to an unsigned 8-bit integer
    #     numer = heatmap - np.min(heatmap)
    #     denom = (heatmap.max() - heatmap.min()) + eps
    #     heatmap = numer / denom
    #     heatmap = (heatmap * 255).astype("uint8")
    #     # return the resulting heatmap to the calling function
    #     return heatmap
    #
    # def overlay_heatmap(self, heatmap, image, alpha=0.5,
    #                     colormap=cv2.COLORMAP_RAINBOW):
    #     # apply the supplied color map to the heatmap and then
    #     # overlay the heatmap on the input image
    #     heatmap = cv2.applyColorMap(heatmap, colormap)
    #     output = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
    #     # return a 2-tuple of the color mapped heatmap and the output,
    #     # overlaid image
    #     return (heatmap, output)