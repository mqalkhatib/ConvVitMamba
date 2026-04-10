import os
import tensorflow as tf
import numpy as np
import keras
import matplotlib.pyplot as plt 
import matplotlib.patches as mpts
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, accuracy_score, confusion_matrix
from scipy.io import loadmat
from tqdm import tqdm
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from display_history import display_history
import scipy.io as sio
from keras import layers
import tensorflow_addons as tfa
from operator import truediv


def AA_andEachClassAccuracy(confusion_matrix):
    list_diag = np.diag(confusion_matrix)
    list_raw_sum = np.sum(confusion_matrix, axis=1)
    each_acc = np.nan_to_num(truediv(list_diag, list_raw_sum))
    average_acc = np.mean(each_acc)
    return each_acc, average_acc

DATASET = 'Houston'  ## Tangdaowan, Houston, Pingan, Qingyun


data, tr, te, class_name = loadData(DATASET)

#if DATASET == "Houston":
#    data = np.transpose(data, (1, 0, 2))
#    tr = tr.T
#    te = te.T
    
gt = tr+te
num_classes = gt.max()
_, labels = get_img_indexes(gt, removeZeroindexes = True)

img_display(classes=tr.T,title='Training',class_name=class_name, Location = "upper left")
img_display(classes=te.T,title='Testing',class_name=class_name, Location = "upper left")
img_display(classes=gt.T,title='GT Full',class_name=class_name, Location = "upper left")

window_size = 9
num_PCA = 15




data = applyPCA(data, numComponents = num_PCA, normalization = True)


# Get class map indexes
tr_indexes, tr_labels = get_img_indexes(tr, removeZeroindexes = True)
X_test_idx, y_test = get_img_indexes(te, removeZeroindexes = True)


Tr_Percentage = 70

X_train_idx, X_val_idx, y_train, y_val = splitTrainTestSet(tr_indexes, tr_labels, testRatio = (1 - Tr_Percentage/100 ))

sample_report = f"{'class': ^25}{'train_num':^10}{'val_num': ^10}{'test_num': ^10}{'total': ^10}\n"
for i in range(1,num_classes+1):
    if i == 0: continue
    sample_report += f"{class_name[i]: ^25}{(y_train==i-1).sum(): ^10}{(y_val==i-1).sum(): ^10}{(y_test==i-1).sum(): ^10}{(gt==i).sum(): ^10}\n"
sample_report += f"{'total': ^25}{len(y_train): ^10}{len(y_val): ^10}{len(y_test): ^10}{len(labels): ^10}"
print(sample_report)


x_train = createImageCubes(data, X_train_idx, window_size)
y_train = keras.utils.to_categorical(y_train)

x_val = createImageCubes(data, X_val_idx, window_size)
y_val = keras.utils.to_categorical(y_val)

image_size = window_size  # Final Image Size
patch_size = 3  # Patch Dimension
num_patches = (image_size // patch_size) ** 2
projection_dim = 32
num_heads = 4
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 4
mlp_head_units = [128, 64]  # Size of the dense layers




"""## Implementing Multilayer Perceptron"""
def multilayer_perceptron(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x

"""## Implementing patch creation as a layer"""

class Patches(layers.Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

"""* **Let's display patches for a sample image**"""


#image = x_train[np.random.choice(range(x_train.shape[0]))]
#resized_image = tf.image.resize(
#    tf.convert_to_tensor([image]), size=(image_size, image_size)
#)
#patches = Patches(patch_size)(resized_image)

#n = int(np.sqrt(patches.shape[1]))

"""## Implement the Patch Encoding Layer"""

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

import tensorflow as tf
from keras import layers

def MambaBlock(
    x,
    d_model,
    expand_factor=2,
    conv_kernel_size=3,
    name=None
):
    """
    Mamba-inspired gated sequence mixing block (functional form).

    Input:
        x: Tensor of shape (B, T, d_model)

    Output:
        Tensor of shape (B, T, d_model)

    Structure:
        - LayerNorm
        - Linear projection -> split into content and gate
        - 1D convolution along token dimension (content branch)
        - GELU activation + sigmoid gating
        - Linear projection back to d_model
        - Residual connection
    """
    # Layer normalization
    x_norm = layers.LayerNormalization(epsilon=1e-6, name=None if name is None else name + "_ln")(x)

    # Joint projection
    E = expand_factor * d_model
    x_proj = layers.Dense(
        units=2 * E,
        use_bias=True,
        name=None if name is None else name + "_in_proj"
    )(x_norm)

    # Split content and gate
    x_content, x_gate = tf.split(x_proj, num_or_size_splits=[E, E], axis=-1)

    # Convolutional token mixing
    x_content = layers.Conv1D(
        filters=E,
        kernel_size=conv_kernel_size,
        padding="same",
        activation=None,
        name=None if name is None else name + "_conv"
    )(x_content)

    # Gating and nonlinearity
    x_content = tf.nn.gelu(x_content)
    x_gate = tf.nn.sigmoid(x_gate)
    x_mixed = layers.Multiply(name=None if name is None else name + "_gate")([x_content, x_gate])

    # Projection back to model dimension
    x_out = layers.Dense(
        units=d_model,
        use_bias=True,
        name=None if name is None else name + "_out_proj"
    )(x_mixed)

    # Residual connection
    return layers.Add(name=None if name is None else name + "_residual")([x, x_out])



def conv3d_block(x, filters, kernel):

    x = layers.Conv3D(filters, kernel, padding='same', activation=tf.nn.relu)(x)
    x = layers.Conv3D(filters, kernel, padding='same', activation=tf.nn.relu)(x)
    return x #layers.Add()([x, shortcut])


def MS_FE(x, num_filters = 8):
    
    x = tf.expand_dims(x, axis = 4)

    # Spectral FE
    x_spe = conv3d_block(x, num_filters, (1,1,3))
    
    # Spatial FE
    x_spa = conv3d_block(x, num_filters, (3,3,1))
    
    # Spectral-Spatial FE
    x_ss = conv3d_block(x, num_filters, (3,3,3))
    
    x_concatenated = tf.concat([x_spe, x_spa, x_ss], axis = 4)
    
    x_shape = x_concatenated.shape
    x = layers.Reshape((x_shape[1], x_shape[2], x_shape[3]*x_shape[4]))(x_concatenated)
    #x = layers.DepthwiseConv2D((3, 3), padding='same', activation="relu")(x)
    x = layers.Conv2D(num_filters*3, (1,1), activation=tf.nn.relu, padding='same')(x)
 
    return x

"""## Build the ViT model"""
def ConvViTMamba(X):
    inputs = layers.Input(X.shape[1:])
    
    
    FE = MS_FE(inputs, num_filters=32)


    
    patches = Patches(patch_size)(FE)

    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)

    #x = AddCLS(projection_dim)(encoded_patches)                 # (B, N+1, D)
    x = encoded_patches
    
    for _ in range(transformer_layers):

        x1 = layers.LayerNormalization(epsilon=1e-6)(x)

        attention_output = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)

        x2 = layers.Add()([attention_output, x])

        x3 = layers.LayerNormalization(epsilon=1e-6)(x2)

        x3 = multilayer_perceptron(x3, hidden_units=transformer_units, dropout_rate=0.1)

        x = layers.Add()([x3, x2])
        
        
    representation = layers.LayerNormalization(epsilon=1e-6)(x)


    features = multilayer_perceptron(representation, hidden_units=mlp_head_units, dropout_rate=0.25)
    
    #x = layers.Bidirectional(layers.LSTM(64, return_sequences=True), merge_mode='concat')(features)
    
    x = MambaBlock(features, mlp_head_units[-1], expand_factor=3, conv_kernel_size=1)
    
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(64, activation = tf.nn.gelu)(x)
    logits = layers.Dense(num_classes , activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=logits)
  
    model.compile(optimizer = 'Adam',
                  loss = 'categorical_crossentropy',
                  #loss = tfa.losses.SigmoidFocalCrossEntropy(gamma = 1.5, alpha = 0.35),
                  metrics = 'accuracy')
    
    return model

"""## Compile, Train, and Evaluate the model"""

model = ConvViTMamba(x_train)
print("The number of parameters is: ", model.count_params())
model.summary()
from Flop_Estimator import net_flops
net_flops(model)

Aa = []
Oa = []
K = []
Ea = []
for i in range(5):
    print("Iteration number: ",i)
    
        
    checkpoint = ModelCheckpoint(
        f"ConvViTMamba_{i}_Window_size_{window_size}.h5",
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        verbose=1
        )
    
    # Define a callback to modify the learning rate dynamically
    lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy',
        factor=0.5,
        patience=10,
        min_lr=1e-5
        )
    # # Reset the model parameters at each iteration  
    model = ConvViTMamba(x_train)
        
    history = model.fit(x_train, y_train,
                        epochs = 2,
                        batch_size = 32,
                        validation_data = (x_val, y_val),
                        callbacks=[checkpoint, lr_callback],
                        )
    

    model.load_weights(f"ConvViTMamba_{i}_Window_size_{window_size}.h5")    
    Y_pred = predict_by_batching(model, input_tensor_idx = X_test_idx, batch_size = 1000, X = data, windowSize = window_size)
    y_pred = np.argmax(Y_pred, axis=1)
    confusion = confusion_matrix(y_test, y_pred)
    oa = accuracy_score(y_test, y_pred)
    each_acc, aa = AA_andEachClassAccuracy(confusion)
    kappa = cohen_kappa_score(y_test, y_pred)
    # Display each iteration score
    print("\noa = ", oa) 
    print("aa = ", aa)
    print('Kappa = ', kappa)

   
    Aa.append(float(format((aa)*100, ".2f")))
    Oa.append(float(format((oa)*100, ".2f")))
    K.append(float(format((kappa)*100, ".2f")))
    Ea.append(each_acc*100)
 

print("\n\noa = ", Oa) 
print("aa = ", Aa)
print('Kappa = ', K)
print('\n')
print('Mean OA = ', format(np.mean(Oa), ".2f"), '+', format(np.std(Oa), ".2f"))
print('Mean AA = ', format(np.mean(Aa), ".2f"), '+', format(np.std(Aa), ".2f"))
print('Mean Kappa = ', format(np.mean(K), ".2f"), '+', format(np.std(K), ".2f"))
EA_mean = np.mean(Ea, axis  = 0)
EA_mean = [round(item, 2) for item in EA_mean]
EA_std = np.std(Ea, axis = 0)
EA_std = [round(item, 2) for item in EA_std]

# ###############################################################################
# Get the Class Map
i = input("Enter Best model number:\n")
model.load_weights(f"ConvViTMamba_{i}_Window_size_{window_size}.h5")    

Predicted_Class_Map = get_class_map(model, data, gt, window_size)
gt_binary = gt.copy()
gt_binary[gt_binary>0]=1
img_display(classes=Predicted_Class_Map*gt_binary,title='Predicted Class Map',class_name=class_name, Location = "upper left")


Name = 'ConvViTMamba'
sio.savemat('ConvViTMamba.mat', {Name: Predicted_Class_Map})
