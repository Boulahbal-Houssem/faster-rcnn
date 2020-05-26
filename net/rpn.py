import tensorflow as tf
from tensorflow.keras.layers import *

class RPN(tf.keras.layers.Layer):
    def __init__(self, num_anchors=1):
        super().__init__()
        self.intermediate = Conv2D(filters=512,kernel_size=[3, 3],kernel_initializer="normal",padding="same")
        self.cls_conv = Conv2D(filters=num_anchors, kernel_size=[1,1],kernel_initializer='uniform',padding="same", name="RPN_cls_out",activation="softmax")
        self.box_reg_conv = Conv2D(filters=num_anchors*4, kernel_size=[1,1],kernel_initializer='zero',padding="same", name="RPN_box_reg",activation="relu")       
    def call(self, input_tensor, training=False):
        x = self.intermediate(input_tensor)
        out_cls = self.cls_conv(x)
        out_box_reg = self.box_reg_conv(x)
        return [out_cls,out_box_reg]

if __name__ == "__main__":
    inputs = Input(shape=(7,7,1048))
    rpn_out = RPN(num_anchors=7)(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=rpn_out, name="RPN TEST")
    model.summary()
