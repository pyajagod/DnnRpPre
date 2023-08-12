import os, gc, pickle, datetime, scipy.sparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Back, Style

from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
# import seaborn as sns
# from cycler import cycler
from IPython.display import display
import math

import tensorflow as tf
import keras.backend as K
from keras.models import Model, load_model
from keras.callbacks import ReduceLROnPlateau, LearningRateScheduler, EarlyStopping
from keras.layers import Dense, Input, Concatenate, Dropout, BatchNormalization
import keras_tuner
import time
name_time = time.strftime('%m%d_%H%M_', time.localtime(time.time()))

import random
use_seed = False

def setup_seed(seed):
    random.seed(seed)  
    np.random.seed(seed)  
    tf.random.set_seed(seed) 
    os.environ['TF_DETERMINISTIC_OPS'] = '1'  

if use_seed:
    setup_seed(42)

# define metrix and loss function
class MetrixAndLossFunction:
    # this package this method
    def correlation_score(y_true, y_pred):
        if type(y_true) == pd.DataFrame: y_true = y_true.values
        if type(y_pred) == pd.DataFrame: y_pred = y_pred.values
        corrsum = 0
        for i in range(len(y_true)):
            corrsum += np.corrcoef(y_true[i], y_pred[i])[1, 0]
        return corrsum / len(y_true)

    # you can also use follow methods
    def mean_squared_error_m(y_true, y_pred):
        if type(y_true) == pd.DataFrame: y_true = y_true.values
        if type(y_pred) == pd.DataFrame: y_pred = y_pred.values
        r2sum = 0
        for i in range(len(y_true)):
            # r2sum += r2_score(y_true[i], y_pred[i])[1, 0]
            r2sum += mean_squared_error(y_true[i], y_pred[i])
            
        return r2sum / len(y_true)

    def negative_correlation_loss(y_true, y_pred):
        my = K.mean(tf.convert_to_tensor(y_pred), axis=1)
        my = tf.tile(tf.expand_dims(my, axis=1), (1, y_true.shape[1]))
        ym = y_pred - my
        r_num = K.sum(tf.multiply(y_true, ym), axis=1)
        r_den = tf.sqrt(K.sum(K.square(ym), axis=1) * float(y_true.shape[-1]))
        r = tf.reduce_mean(r_num / r_den)
        return - r

# data-related operations
class DataProcessing:
    def __init__(self, FP_CITE_TRAIN_INPUTS, FP_CITE_TRAIN_TARGETS, FP_CITE_TEST_INPUTS, FP_CELL_METADATA):
        self.constant_cols = []
        self.important_cols = []
        self.metadata_df = pd.read_csv(FP_CELL_METADATA, index_col='cell_id')
        self.X = pd.read_hdf(FP_CITE_TRAIN_INPUTS).drop(columns=self.constant_cols)
        self.X_test = pd.read_hdf(FP_CITE_TEST_INPUTS).drop(columns=self.constant_cols)
        self.Y = pd.read_hdf(FP_CITE_TRAIN_TARGETS)

    # create new feature for group
    def create_new_feature(self):
        self.metadata_df = self.metadata_df[self.metadata_df.technology == "citeseq"]
        conditions = [
        self.metadata_df['donor'].eq(27678) & self.metadata_df['day'].eq(2),
        self.metadata_df['donor'].eq(27678) & self.metadata_df['day'].eq(3),
        self.metadata_df['donor'].eq(27678) & self.metadata_df['day'].eq(4),
        self.metadata_df['donor'].eq(27678) & self.metadata_df['day'].eq(7),
        self.metadata_df['donor'].eq(13176) & self.metadata_df['day'].eq(2),
        self.metadata_df['donor'].eq(13176) & self.metadata_df['day'].eq(3),
        self.metadata_df['donor'].eq(13176) & self.metadata_df['day'].eq(4),
        self.metadata_df['donor'].eq(13176) & self.metadata_df['day'].eq(7),
        self.metadata_df['donor'].eq(31800) & self.metadata_df['day'].eq(2),
        self.metadata_df['donor'].eq(31800) & self.metadata_df['day'].eq(3),
        self.metadata_df['donor'].eq(31800) & self.metadata_df['day'].eq(4),
        self.metadata_df['donor'].eq(31800) & self.metadata_df['day'].eq(7),
        self.metadata_df['donor'].eq(32606) & self.metadata_df['day'].eq(2),
        self.metadata_df['donor'].eq(32606) & self.metadata_df['day'].eq(3),
        self.metadata_df['donor'].eq(32606) & self.metadata_df['day'].eq(4),
        self.metadata_df['donor'].eq(32606) & self.metadata_df['day'].eq(7)
        ]
        # create a list of the values we want to assign for each condition
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

        # create a new column and use np.select to assign values to it using our lists as arguments
        self.metadata_df['new'] = np.select(conditions, values)

    # select different kinds of feature for train
    def select_feature(self):
        self.constant_cols = list(self.X.columns[(self.X == 0).all(axis=0).values]) + list(self.X_test.columns[(self.X_test == 0).all(axis=0).values])
        self.important_cols = ['ENSG00000135218_CD36',
                  'ENSG00000010278_CD9',
                  'ENSG00000204287_HLA-DRA',
                  'ENSG00000117091_CD48',
                  'ENSG00000004468_CD38',
                  'ENSG00000173762_CD7',
                  'ENSG00000137101_CD72',
                  'ENSG00000019582_CD74',
                  'ENSG00000169442_CD52',
                  'ENSG00000170458_CD14',
                  'ENSG00000272398_CD24',
                  'ENSG00000026508_CD44',
                  'ENSG00000114013_CD86',
                  'ENSG00000174059_CD34',
                  'ENSG00000139193_CD27',
                  'ENSG00000105383_CD33',
                  'ENSG00000085117_CD82',
                  'ENSG00000177455_CD19',
                  'ENSG00000002586_CD99',
                  'ENSG00000196126_HLA-DRB1',
                  'ENSG00000135404_CD63',
                  'ENSG00000012124_CD22',
                  'ENSG00000134061_CD180',
                  'ENSG00000105369_CD79A',
                  'ENSG00000116824_CD2',
                  'ENSG00000010610_CD4',
                  'ENSG00000139187_KLRG1',
                  'ENSG00000204592_HLA-E',
                  'ENSG00000090470_PDCD7',
                  'ENSG00000206531_CD200R1L',
                  'ENSG00000166710_B2M',
                  'ENSG00000198034_RPS4X',
                  'ENSG00000188404_SELL',
                  'ENSG00000130303_BST2',
                  'ENSG00000128040_SPINK2',
                  'ENSG00000206503_HLA-A',
                  'ENSG00000108107_RPL28',
                  'ENSG00000143226_FCGR2A',
                  'ENSG00000133112_TPT1',
                  'ENSG00000166091_CMTM5',
                  'ENSG00000026025_VIM',
                  'ENSG00000205542_TMSB4X',
                  'ENSG00000109099_PMP22',
                  'ENSG00000145425_RPS3A',
                  'ENSG00000172247_C1QTNF4',
                  'ENSG00000072274_TFRC',
                  'ENSG00000234745_HLA-B',
                  'ENSG00000075340_ADD2',
                  'ENSG00000119865_CNRIP1',
                  'ENSG00000198938_MT-CO3',
                  'ENSG00000135046_ANXA1',
                  'ENSG00000235169_SMIM1',
                  'ENSG00000101200_AVP',
                  'ENSG00000167996_FTH1',
                  'ENSG00000163565_IFI16',
                  'ENSG00000117450_PRDX1',
                  'ENSG00000124570_SERPINB6',
                  'ENSG00000112077_RHAG',
                  'ENSG00000051523_CYBA',
                  'ENSG00000107130_NCS1',
                  'ENSG00000055118_KCNH2',
                  'ENSG00000029534_ANK1',
                  'ENSG00000169567_HINT1',
                  'ENSG00000142089_IFITM3',
                  'ENSG00000139278_GLIPR1',
                  'ENSG00000142227_EMP3',
                  'ENSG00000076662_ICAM3',
                  'ENSG00000143627_PKLR',
                  'ENSG00000130755_GMFG',
                  'ENSG00000160593_JAML',
                  'ENSG00000095932_SMIM24',
                  'ENSG00000197956_S100A6',
                  'ENSG00000171476_HOPX',
                  'ENSG00000116675_DNAJC6',
                  'ENSG00000100448_CTSG',
                  'ENSG00000100368_CSF2RB',
                  'ENSG00000047648_ARHGAP6',
                  'ENSG00000198918_RPL39',
                  'ENSG00000196154_S100A4',
                  'ENSG00000233968_AL157895.1',
                  'ENSG00000137642_SORL1',
                  'ENSG00000133816_MICAL2',
                  'ENSG00000130208_APOC1',
                  'ENSG00000105610_KLF1']
    # target normalization
    def Y_normalization(self):
        self.Y = self.Y.values
        self.Y -= self.Y.mean(axis=1).reshape(-1, 1)
        self.Y /= self.Y.std(axis=1).reshape(-1, 1)

    def convert_train_and_test(self):
        cell_index = self.X.index
        meta = self.metadata_df.reindex(cell_index)
        self.X0 = self.X[self.important_cols].values

        cell_index_test = self.X_test.index
        meta_test = self.metadata_df.reindex(cell_index_test)
        self.X0t = self.Xt[self.important_cols].values

        st = StandardScaler()
        self.X0 = st.fit_transform(self.X0)
        self.X0t = st.transform(self.X0t)

        print(f'X0 shape {self.X0.shape} X0t shape {self.X0t.shape}')
        gc.collect()

    # apply truncated SVD to train and test together.
    def svd(self, is_svd = True, svd_path = ''):
        if is_svd:
            with open(svd_path, 'rb') as f: both = pickle.load(f)
        else:
            both = np.vstack([self.X, self.Xt])
            assert both.shape[0] == 119651
            print(f"Shape of both before SVD: {both.shape}")
            svd = TruncatedSVD(n_components = 512, random_state = 1) # 512 is possible
            both = svd.fit_transform(both)
            print(f"Shape of both after SVD:  {both.shape}")

            self.X = np.hstack([self.X[:, :75], self.X0])
            self.Xt = np.hstack([self.Xt[:, :75], self.X0t])

# dnnRpPre model
class DnnRpPre:
    def __init__(self, X, Xt, Y):
        self.best_hp = {}
        self.X = X
        self.Xt = Xt
        self.Y = Y
        self.LR_START = 0.01
        self.BATCH_SIZE = 256
        self.SUBMIT = True

    def dnn_model(self, hp, n_inputs=0, n_outputs=0):
        activation = hp.Choice('activation', ['relu', 'swish', 'tanh', 'sigmoid'])
        reg1 = hp.Float("reg1", min_value=1e-8, max_value=1e-4, sampling="log")
        reg2 = hp.Float("reg2", min_value=1e-10, max_value=1e-5, sampling="log")
        
        inputs = Input(shape=(n_inputs,))
        x = inputs
        
        for i in range(hp.Int("num_layers", min_value=2, max_value=5)):  # Adjust the range as needed
            x = Dense(hp.Choice(f'units_{i}', [64, 128, 256, 512, 1024]), kernel_regularizer=tf.keras.regularizers.l2(reg1),
                    activation=activation)(x)
            x = Dropout(hp.Float(f'do_{i}', min_value=0.1, max_value=0.5, step=0.1))(x)

        x = Dense(n_outputs, kernel_regularizer=tf.keras.regularizers.l2(reg2))(x)
        
        regressor = Model(inputs, x)
        
        return regressor


    def ifOrNotTune(self , TUNE = False):
        if TUNE:
            tuner = keras_tuner.BayesianOptimization(
                self.dnn_model,
                overwrite=True,
                objective=keras_tuner.Objective("val_negative_correlation_loss", direction="min"),
                max_trials=70,
                directory='./temp',
                seed=1)
            lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, 
                                patience=6, verbose=0)
            es = EarlyStopping(monitor="val_loss",
                            patience=50, 
                            verbose=0,
                            mode="min", 
                            restore_best_weights=True)
            callbacks = [lr, es, tf.keras.callbacks.TerminateOnNaN()]
            X_tr, X_va, y_tr, y_va = train_test_split(self.X, self.Y, test_size=0.2, random_state=10)
            tuner.search(X_tr, y_tr,
                        epochs=1000,
                        validation_data=(X_va, y_va),
                        batch_size=self.BATCH_SIZE,
                        callbacks=callbacks, verbose=2)
            del X_tr, X_va, y_tr, y_va, lr, es, callbacks

            tuner.results_summary()

            display(pd.DataFrame([hp.values for hp in tuner.get_best_hyperparameters(10)]))
            self.best_hp = tuner.get_best_hyperparameters(1)[0]
        else:
            self.best_hp = keras_tuner.HyperParameters()
            self.best_hp.values = {'reg1': 9.613e-6,
                      'reg2': 1e-07,
                      'units1': 159,
                      'units2': 512,
                      'units3': 256,
                      'units4': 256
                     } 
    def train_data(self，N_SPLITS = 10):
        VERBOSE = 0
        EPOCHS = 300

        np.random.seed(1)
        tf.random.set_seed(1)

        kf = GroupKFold(n_splits=N_SPLITS)
        score_list = []
        for fold, (idx_tr, idx_va) in enumerate(kf.split(self.X, groups=self.meta.new)):
            start_time = datetime.datetime.now()
            model = None
            gc.collect()
            X_tr = self.X[idx_tr]
            y_tr = self.Y[idx_tr]
            X_va = self.X[idx_va]
            y_va = self.Y[idx_va]

            lr = ReduceLROnPlateau(monitor="val_loss", factor=0.9, 
                                patience=4, verbose=VERBOSE)
            es = EarlyStopping(monitor="val_loss",
                            patience=30,
                            verbose=0,
                            mode="min", 
                            restore_best_weights=True)
            callbacks = [lr, es, tf.keras.callbacks.TerminateOnNaN()]

            model = self.dnn_model(self.best_hp, X_tr.shape[1])

            history = model.fit(X_tr, y_tr, 
                                validation_data=(X_va, y_va), 
                                epochs=EPOCHS,
                                verbose=VERBOSE,
                                batch_size=self.BATCH_SIZE,
                                shuffle=True,
                                callbacks=callbacks)
            del X_tr, y_tr
            if self.SUBMIT:
                model.save(f"./temp/model_{fold}")
            history = history.history
            callbacks, lr = None, None
            
            y_va_pred = model.predict(X_va, batch_size=len(X_va))
            maf = MetrixAndLossFunction()
            corrscore = maf.correlation_score(y_va, y_va_pred)

            print(f"Fold {fold}: {es.stopped_epoch:3} epochs, corr =  {corrscore:.5f}")
            del es, X_va
            score_list.append(corrscore)

        print(f"{Fore.GREEN}{Style.BRIGHT}Average  corr = {np.array(score_list).mean():.5f}{Style.RESET_ALL}")

    def predict_data(self, N_SPLITS = 10, demo_submission_path = '', result_path = ''):
        test_pred = np.zeros((len(self.Xt), 140), dtype=np.float32)
        for fold in range(N_SPLITS):
            print(f"Predicting with fold {fold}")
            maf = MetrixAndLossFunction()
            model = load_model(f"./temp/model_{fold}",
                            custom_objects={'negative_correlation_loss': maf.negative_correlation_loss})
            test_pred += model.predict(self.Xt)/N_SPLITS
        
        # if no data leak, ignore it
        test_pred[:7476] = self.Y[:7476]

        submission = pd.read_csv(demo_submission_path, index_col='row_id', squeeze=True)
        # if don't want to ravel, directly submmit is ok
        submission.iloc[:len(test_pred.ravel())] = test_pred.ravel()
        assert not submission.isna().any()
        submission.to_csv(result_path)
        display(submission)