from spektral.layers import ECCConv, GlobalAvgPool, GATConv
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2


class GNN_encoder(Model):
    def __init__(self, fltrs_out=64, l2_reg=1e-3):
        super().__init__()
        
        # ECC Layer
        self.conv1 = ECCConv(fltrs_out, kernel_network=[32], activation="relu", kernel_regularizer=l2(l2_reg))
        
        # GAT Layer
        self.conv2 = GATConv(fltrs_out, activation="relu", kernel_regularizer=l2(l2_reg),
                             attn_kernel_regularizer=l2(l2_reg), return_attn_coef=True)
        
        # Global Average Pooling
        self.flatten = GlobalAvgPool()
        
        # Fully Connected Layer
        self.fc = Dense(32, "relu", kernel_regularizer=l2(l2_reg))

    def call(self, inputs):
        A_in, X_in, E_in = inputs

        x = self.conv1([X_in, A_in, E_in])

        x, attn = self.conv2([x, A_in])

        x = self.flatten(x)

        x = self.fc(x)
        
        return x

class Contrast(Model):
    def __init__(self):
        super(Contrast, self).__init__()

    def call(self, inputs):
        z_1, z_2 = inputs
        return tf.abs(z_1 - z_2)

class Regression(Model):
    def __init__(self):
        super(Regression, self).__init__()

        self.fc = Dense(1, activation = None)

    def call(self, x, version):
        x = self.fc(x)

        # Logistic regression
        if version == "log":
            return tf.sigmoid(x)
        
        # Linear regression
        else:
            return x
