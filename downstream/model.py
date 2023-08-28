from spektral.layers import ECCConv, GlobalAvgPool, GATConv
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.regularizers import l2


class gnn_standard(Model):
    """ 
    A standard graph neural network with all randomly initialized layers.
    
    """
    def __init__(self, fltrs_out=64, l2_reg=1e-3):
        super(gnn_standard, self).__init__()

        # Graph layers
        self.conv1 = ECCConv(fltrs_out, kernel_network=[32], activation="relu", kernel_regularizer=l2(l2_reg))
        
        self.conv2 = GATConv(fltrs_out, activation="relu", kernel_regularizer=l2(l2_reg),
                             attn_kernel_regularizer=l2(l2_reg), return_attn_coef=True)
        self.flatten = GlobalAvgPool()
        
        # Fully connected layers
        self.fc = Dense(32, "relu", kernel_regularizer=l2(l2_reg))
        self.out = Dense(1, "sigmoid", kernel_regularizer=l2(l2_reg))
        
    def call(self, inputs):
        A_in, X_in, E_in = inputs

        x = self.conv1([X_in, A_in, E_in])
        x, attn = self.conv2([x, A_in])
        x = self.flatten(x)
        x = self.fc(x)
        x = self.out(x)        
        
        return x
    

# class gnn_rp(Model):
#     """ 
#     Graph Neural network with pretrained layers from the relative positioning task.
#     """
    
#     def __init__(self, pretrained_model_path, fltrs_out=64, l2_reg=1e-3):
#         super(gnn_rp, self).__init__()

#         # Load the pretrained relative_positioning model
#         pretrained_model = load_model(pretrained_model_path, custom_objects={"ECCConv": ECCConv, "GATConv": GATConv})

#         # Extract the nested gnn_encoder model
#         gnn_encoder = pretrained_model.get_layer(name='gnn_encoder')

#         # Extract layers from the gnn_encoder
#         self.pretrained_conv1 = gnn_encoder.get_layer(name='ECCConv_layer_name')  # Replace with the actual layer name
#         self.pretrained_conv2 = gnn_encoder.get_layer(name='GATConv_layer_name')  # Replace with the actual layer name

#         # Other layers
#         self.flatten = GlobalAvgPool()
#         self.fc = Dense(32, "relu", kernel_regularizer=l2(l2_reg))
#         self.out = Dense(1, "sigmoid", kernel_regularizer=l2(l2_reg))

#     def call(self, inputs):
#         A_in, X_in, E_in = inputs

#         x = self.pretrained_conv1([X_in, A_in, E_in])
#         x, attn = self.pretrained_conv2([x, A_in])
#         x = self.flatten(x)
#         x = self.fc(x)
#         x = self.out(x)
        
#         return x
