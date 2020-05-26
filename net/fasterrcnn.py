import tensorflow as tf
from tensorflow.keras.layers import *
import riopooling, rpn



class FasterRCNN(tf.keras.Model):
    def __init__(self, _input_shape,_anchor_num , name="FasterRCNN", **kwargs):
        super(FasterRCNN, self).__init__(name=name, **kwargs)
        self._input_shape = _input_shape
        self.input_layer = Input(shape=_input_shape)
        self.feature_extractor = tf.keras.applications.ResNet50V2(
            include_top=False, weights='imagenet', input_tensor=Input(shape=_input_shape),
            pooling=None)
        
        for layer in self.feature_extractor.layers:
            layer.trainable = False
        self.RPN = RPN(_anchor_num)
        self.riopooling = RoiPoolingConv(pool_size=7, num_rois=_anchor_num)

        super(FasterRCNN, self).__init__(
            inputs=self.input_layer,
            outputs=self.out,
            **kwargs)
        def build(self):
            # Initialize the graph
            self._is_graph_network = True
            self._init_graph_network(
                inputs=self.input_layer,
                outputs=self.out
        )

    def call(self, inputs):
        x = self.feature_extractor(inputs)
        x = Flatten()(x)
        return self.classifier(x)

if __name__ == "__main__":
    model = FasterRCNN((222, 222, 3),3)
    model.build((64,222,222,3))
    model.summary()