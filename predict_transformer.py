# -*- coding: utf-8 -*-
"""
Created on Wed May 20 11:28:51 2026

@author: user
"""

import sys
import os
import time
import pickle
import math
import configparser
import numpy as np
import csv
import h5py
import tarfile
import shutil
import math
import random

from rdkit import Chem
from rdkit.Chem import SaltRemover
from layers import PositionLayer, MaskLayerLeft, \
                   MaskLayerRight, MaskLayerTriangular, \
                   SelfLayer, LayerNormalization

from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model


#tar = tarfile.open('cv_100_16.tar');
#tar.extractall();
#tar.close();
props = pickle.load( open( "model.pkl", "rb" ));
np.random.seed(100666)

class suppress_stderr(object):
   def __init__(self):
       self.null_fds = [os.open(os.devnull,os.O_RDWR)]
       self.save_fds = [os.dup(2)]
   def __enter__(self):
       os.dup2(self.null_fds[0],2)
   def __exit__(self, *_):
       os.dup2(self.save_fds[0],2)
       for fd in self.null_fds + self.save_fds:
          os.close(fd)



def buildNetwork():

    unfreeze = False;

    l_in = layers.Input( shape= (None,));
    l_mask = layers.Input( shape= (None,));

    l_ymask = [];
    for i in range(len(props)):
       l_ymask.append( layers.Input( shape=(1, )));

    #transformer part
    #positional encodings for product and reagents, respectively
    l_pos = PositionLayer(EMBEDDING_SIZE)(l_mask);
    l_left_mask = MaskLayerLeft()(l_mask);

    #encoder
    l_voc = layers.Embedding(input_dim = vocab_size, output_dim = EMBEDDING_SIZE, input_length = None, trainable = unfreeze);
    l_embed = layers.Add()([ l_voc(l_in), l_pos]);

    for layer in range(n_block):

       #self attention
       l_o = [ SelfLayer(EMBEDDING_SIZE, KEY_SIZE, trainable= unfreeze) ([l_embed, l_embed, l_embed, l_left_mask]) for i in range(n_self)];

       l_con = layers.Concatenate()(l_o);
       l_dense = layers.TimeDistributed(layers.Dense(EMBEDDING_SIZE, trainable = unfreeze), trainable = unfreeze) (l_con);
       if unfreeze == True: l_dense = layers.Dropout(rate=0.1)(l_dense);
       l_add = layers.Add()( [l_dense, l_embed]);
       l_att = LayerNormalization(trainable = unfreeze)(l_add);

       #position-wise
       l_c1 = layers.Conv1D(N_HIDDEN, 1, activation='relu', trainable = unfreeze)(l_att);
       l_c2 = layers.Conv1D(EMBEDDING_SIZE, 1, trainable = unfreeze)(l_c1);
       if unfreeze == True: l_c2 = layers.Dropout(rate=0.1)(l_c2);
       l_ff = layers.Add()([l_att, l_c2]);
       l_embed = LayerNormalization(trainable = unfreeze)(l_ff);

    #end of Transformer's part
    l_encoder = l_embed;

    #text-cnn part
    #https://github.com/deepchem/deepchem/blob/b7a6d3d759145d238eb8abaf76183e9dbd7b683c/deepchem/models/tensorgraph/models/text_cnn.py

    l_in2 =  layers.Input( shape= (None,EMBEDDING_SIZE));

    kernel_sizes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20];
    num_filters=[100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160];

    l_pool = [];
    for i in range(len(kernel_sizes)):
       l_conv = layers.Conv1D(num_filters[i], kernel_size=kernel_sizes[i], padding='valid',
                              kernel_initializer='normal', activation='relu')(l_in2);
       l_maxpool = layers.Lambda(lambda x: tf.reduce_max(x, axis=1))(l_conv);
       l_pool.append(l_maxpool);

    l_cnn = layers.Concatenate(axis=1)(l_pool);
    l_cnn_drop = layers.Dropout(rate = 0.25)(l_cnn);

    #dense part
    l_dense =layers.Dense(N_HIDDEN_CNN, activation='relu') (l_cnn_drop);

    #https://github.com/ParikhKadam/Highway-Layer-Keras
    transform_gate = layers.Dense(units= N_HIDDEN_CNN, activation="sigmoid",
                     bias_initializer=tf.compat.v1.keras.initializers.Constant(-1))(l_dense);

    carry_gate = layers.Lambda(lambda x: 1.0 - x, output_shape=(N_HIDDEN_CNN,))(transform_gate);
    transformed_data = layers.Dense(units= N_HIDDEN_CNN, activation="relu")(l_dense);
    transformed_gated = layers.Multiply()([transform_gate, transformed_data]);
    identity_gated = layers.Multiply()([carry_gate, l_dense]);

    l_highway = layers.Add()([transformed_gated, identity_gated]);

    #Because of multitask we have here a few different outputs and a custom loss.

    def mse_loss(prop):
       def loss(y_true, y_pred):
          y2 = y_true * l_ymask[prop] + y_pred * (1 - l_ymask[prop]);
          return tf.compat.v1.keras.losses.mse(y2, y_pred);
       return loss;

    def binary_loss(prop):
       def loss(y_true, y_pred):
           y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon() );
           r = y_true * K.log(y_pred) + (1.0 - y_true) * K.log(1.0 - y_pred);
           r = -tf.reduce_mean(r * l_ymask[prop] );
           return r;
       return loss;

    l_out = [];
    losses = [];
    for prop in props:
       if props[prop][2] == "regression":
          l_out.append(layers.Dense(1, activation='linear', name="Regression-" + props[prop][1]) (l_highway));
          losses.append(mse_loss(prop));
       else:
          l_out.append(layers.Dense(1, activation='sigmoid', name="Classification-" + props[prop][1]) (l_highway));
          losses.append(binary_loss(prop));

    l_input = [l_in2];
    l_input.extend(l_ymask);

    mdl = tf.compat.v1.keras.Model(l_input, l_out);
    mdl.compile (optimizer = 'adam', loss = losses);

    #mdl.summary();

    K.set_value(mdl.optimizer.lr, 1.0e-4);

    #so far we do not train the encoder part of the model.
    encoder = tf.compat.v1.keras.Model([l_in, l_mask], l_encoder);
    encoder.compile(optimizer = 'adam', loss = 'mse');
    encoder.set_weights(np.load("embeddings.npy", allow_pickle = True));

    #encoder.summary();

    return mdl, encoder;

def gen_data(data):

    batch_size = len(data);

    #search for max lengths
    nl = len(data[0][0]);
    for i in range(1, batch_size, 1):
        nl_a = len(data[i][0]);
        if nl_a > nl:
            nl = nl_a;

    nl = nl + CONV_OFFSET;

    x = np.zeros((batch_size, nl), np.int8);
    mx = np.zeros((batch_size, nl), np.int8);

    z = [];
    ym = [];

    for i in range(len(props)):
       z.append(np.zeros((batch_size, 1), np.float32));
       ym.append(np.zeros((batch_size, 1), np.int8));

    for cnt in range(batch_size):

        n = len(data[cnt][0]);
        for i in range(n):
           x[cnt, i] = char_to_ix[ data[cnt][0][i]] ;
        mx[cnt, :i+1] = 1;

        for i in range(len(props)):
           z[i][cnt] = data[cnt][1][i];
           ym[i][cnt ] = data[cnt][2][i];

    d = [x, mx];

    for i in range(len(props)):
       d.extend([ym[i]]);

    return d, z;

from layers import PositionLayer, MaskLayerLeft, \
                   MaskLayerRight, MaskLayerTriangular, \
                   SelfLayer, LayerNormalization

from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.utils import plot_model
try:
    tf.random.set_random_seed(SEED);
except:
    print ("not supported tf.random.set_random_seed(SEED)")


CONV_OFFSET = 20;
N_HIDDEN = 512;
N_HIDDEN_CNN = 512;
EMBEDDING_SIZE = 64;
KEY_SIZE = EMBEDDING_SIZE;
n_block, n_self = 3, 10;

#our vocabulary
chars = " ^#%()+-./0123456789=@ABCDEFGHIKLMNOPRSTVXYZ[\\]abcdefgilmnoprstuy$";
g_chars = set(chars);
vocab_size = len(chars);

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }




def process(smiles):
    mdl, encoder = buildNetwork();
    mdl.load_weights("model.h5");
    arr=[]
    remover = SaltRemover.SaltRemover();
    with suppress_stderr():
         mol=smiles
         m = Chem.MolFromSmiles(mol);
         m = remover.StripMol(m);
         if m is not None and m.GetNumAtoms() > 0:
            for step in range(10):
               arr.append(Chem.MolToSmiles(m, rootedAtAtom = np.random.randint(0, m.GetNumAtoms()), canonical = False));
               print(np.random.randint(0, m.GetNumAtoms()))
       
    z = np.zeros(len(props), dtype=np.float32);
    ymask = np.ones(len(props), dtype=np.int8);
    d= [];
    for i in range(len(arr)):
        d.append( [arr[i], z, ymask]);

    x, y = gen_data(d);
    internal = encoder.predict( [x[0], x[1]]);


    p = [internal];
    for i in range(len(props)):
        p.extend([x[i+2]]);

    y = mdl.predict( p );
    res = np.zeros( len(props));
    
    for prop in props:
        if len(props) == 1:
           res[prop] = np.mean(y);
        else:
           res[prop] = np.mean(y[prop]);
    if res[prop]>=0.5:
       dec='Active'
    else:
        dec='Inactive'
       
    return dec
  

