# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 23:47:07 2021

@author: GH
"""
import pandas as pd
import numpy as np
import tensorflow as tf
import keras.backend as k
import tqdm
import random
from keras import optimizers
from keras.layers import *
from keras.models import Input, Model
from keras import regularizers
import keras
from keras.optimizers import Adam
import os
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5 import QtCore, QtGui
import warnings
warnings.filterwarnings("ignore")

def pinball_score(true, pred, number_q):
    q=np.linspace(1/(number_q+1),(1-(1/(number_q+1))),number_q)
    loss = np.where(np.less(true,pred), (1-q)*(np.abs(true-pred)), q*(np.abs(true-pred)))
    return np.mean(loss)

def winkler_score(true, lower,upper, PI):
    score_tmp = np.where(np.less(upper,true),2*(true-upper)/(1-PI),np.zeros_like(true))
    score = np.where(np.less(true,lower),2*(lower-true)/(1-PI),score_tmp)
    return np.mean(score+upper-lower)

def PCIP(true, lower,upper, PI):
    score_tmp = np.where(np.less(upper,true),np.zeros_like(true),np.ones_like(true))
    score = np.where(np.less(true,lower),np.zeros_like(true),score_tmp)
    return np.mean(score)



class ES_QBLSTM:
    def __init__(self,path,step,quantiles=9,learningRate=0.002,epochs=60,batch=100,alpha=0.1):
        self.__filePath = path
        self.__lr = learningRate
        self.__ecoph = epochs
        self.__batchSize = batch
        self.__rawData = None
        self.__esData = None
        self.__newData = None
        self.__esAlpha = alpha
        self.__step = step
        self.__quantile = quantiles
        self.__result = None
        self.__model = None
        self.__tmpModel = None

    def get_esdata(self):
        return self.__esData

    def get_newdata(self):
        return self.__newData

    def get_rawdata(self):
        return self.__rawData

    def get_result(self): 
        return self.__result
    
    def get_step(self):
        return self.__step

    def get_model(self):
        return self.__model

    def get_quantile(self):
        return self.__quantile

    def stopModel(self):
        self.__tmpModel.stop_training = True

    def exponentialSmoothing(self):     
        QApplication.processEvents()
        s_temp=[]
        s_temp.append(self.__rawData[0])
        for i in range(1,len(self.__rawData),1):
            s_temp.append(self.__esAlpha* self.__rawData[i] + (1 - self.__esAlpha) * s_temp[i-1])
        
        self.__esData = np.array(s_temp).reshape(self.__rawData.shape)
    
    def normalization(self):
        QApplication.processEvents()
        self.__newData = self.__rawData / self.__esData
    
    def preprocess(self):
        QApplication.processEvents()
        print('preprocess: exponential smoothing and normalizing')
        try:
            df = pd.read_excel(self.__filePath,header=None)
        except(IOError,OSError):
            print('文件读取异常')
        else:    
            self.__rawData = np.array(df)
            self.exponentialSmoothing()
            self.normalization()

    def datasplit(self,data):
        QApplication.processEvents()
        datasize = data.shape[0]
        x=[]
        y=[]
        for i in range(datasize-self.__step):
            x.append(data[i:i+self.__step].tolist())
            y.append(data[i+self.__step].tolist()*self.__quantile)
        return x,y,len(x)

    def dataSlice(self):
        QApplication.processEvents()
        print('data slicing')
        x_raw,y_raw,size = self.datasplit(self.__newData)
        x_raw2,y_raw2,size2 = self.datasplit(self.__esData)
        train_data = int(size * 0.8)
        val_data = int(size * 0.1)

        xtrain = x_raw[0:train_data]
        ytrain = y_raw[0:train_data]
        x_train=np.array(xtrain)
        y_train=np.array(ytrain).reshape(-1,self.__quantile)

        xval = x_raw[train_data:train_data+val_data]
        yval = y_raw[train_data:train_data+val_data]
        x_val=np.array(xval)
        y_val = np.array(yval).reshape(-1,self.__quantile)

        xtest = x_raw[train_data+val_data:size]
        ytest = y_raw[train_data+val_data:size]
        x_test=np.array(xtest)
        y_test=np.array(ytest).reshape(-1,self.__quantile)

        xtest2 = x_raw2[train_data+val_data:size]
        ytest2 = y_raw2[train_data+val_data:size]
        x_test2=np.array(xtest2)
        y_test2=np.array(ytest2).reshape(-1,self.__quantile)

        return x_train,y_train,x_val,y_val,x_test,y_test,y_test2

    def pinball_loss(self,y_true, y_pred):
        p=0
        q1=tf.linspace(1/(self.__quantile+1),self.__quantile/(self.__quantile+1),self.__quantile)
        q2=1-q1
        r=0.1
        location=tf.less(y_true,y_pred)
        e = tf.abs(y_true - y_pred)
    
        re=tf.ones_like(e)*r
        position=tf.less(e,re)
    
        h1=tf.square(e)/(2*r)
        h2=e-r/2
        h=tf.where(position,h1,h2)
    
        p1=tf.multiply(q2,h)
        p2=tf.multiply(q1,h)
        p=tf.reduce_mean(tf.where(location,p1,p2),axis=0)
        p=tf.reduce_mean(p)
    
        return p
    
    
    def build_model(self,num_steps,num_features):
        QApplication.processEvents()
        print('build model')
        input_layer = Input(name='input_layer',shape=(num_steps,num_features))
        lstm1 = Bidirectional(LSTM(128,name='lstm_layer1',return_sequences=True, activation='relu'))(input_layer)
        lstm2 = Bidirectional(LSTM(128,name='lstm_layer2',return_sequences=False, activation='relu'))(lstm1)
        output_layer = Dense(self.__quantile,name='output_layer',kernel_regularizer=regularizers.l2(0.001))(lstm2)
        model = Model(input_layer,output_layer)
        adam = Adam(self.__lr, clipnorm=1.0,decay=0.1,amsgrad=True)
        model.compile(loss=self.pinball_loss, optimizer=adam)
        
        return model
    
    def trainModel(self,x_train,y_train,x_val,y_val,x_test,y_test,y_test2):
        QApplication.processEvents()
        print('training model')
        self.__tmpModel = self.build_model(num_steps=x_train.shape[1], num_features=x_train.shape[2])

        print(f'x_train.shape = {x_train.shape}')
        print(f'y_train.shape = {y_train.shape}')

        psv = PrintSomeValues(self.__tmpModel,x_test,y_test2,y_test,self.__quantile)
        
        self.__tmpModel.summary()
      
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,patience=10, mode='min')

        early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=2,mode='min')
   
        self.__tmpModel.fit(x_train, y_train, 
            validation_data=(x_val, y_val),
            epochs=self.__ecoph, 
            batch_size=self.__batchSize,
            initial_epoch=0,
            callbacks=[early_stopping,reduce_lr, psv]
            )
        

    def prediction(self,x_test,y_test2,y_test):
        QApplication.processEvents()
        print('prediction')
        pred = self.__tmpModel.predict(x_test)*y_test2
        test = y_test*y_test2

        predict_all = pred.flatten()
        truth_all = test.flatten()
       
        pinball=pinball_score(test,pred,self.__quantile)
        prediction=pred
        truth=test[:,0]
        
        low = int((self.__quantile+1)*0.3-1)
        high = int((self.__quantile+1)*0.7-1)
        low1 = int((self.__quantile+1)*0.1-1)
        high1 =int((self.__quantile+1)*0.9-1)
        winkler40 = winkler_score(truth, prediction[:,low],prediction[:,high], 0.4)
        winkler80 = winkler_score(truth, prediction[:,low1],prediction[:,high1], 0.8)
        pcip40 = PCIP(truth, prediction[:,low],prediction[:,high], 0.4)
        pcip80 = PCIP(truth, prediction[:,low1],prediction[:,high1], 0.8)
        ace40 = np.abs(pcip40 - 0.4)
        ace80 = np.abs(pcip80 - 0.8)
        
        truth_all_reshape=np.reshape(test[:,0],[-1,1])
        predict_all_reshape=np.reshape(predict_all,[-1,self.__quantile])
        y_con = np.concatenate((truth_all_reshape, predict_all_reshape), axis=1)
        #输出真实值和预测值
        column=["true_data"]
        for i in range(1,(self.__quantile+1)):
            column.append("quantile_%.2f"%(i/(self.__quantile+1)))
        y_out = pd.DataFrame(y_con, columns=column)
        #y_out.to_excel(resultPath+Name+'-piball_loss = %.4f,winkler40 = %.4f,winkler80 = %.4f,pcip40 = %.4f,pcip80 = %.4f,ace40 = %.4f,ace80 = %.4f.xlsx'\
        #    % (pinball,winkler40,winkler80,pcip40,pcip80,ace40,ace80))
        print(y_out)
        self.__result=y_con

    def train(self):
        print('strat')
        self.preprocess()
        x_train,y_train,x_val,y_val,x_test,y_test,y_test2 = self.dataSlice()
        self.trainModel(x_train,y_train,x_val,y_val,x_test,y_test,y_test2)
        self.prediction(x_test,y_test2,y_test)
        self.__model = self.__tmpModel
        self.__tmpModel = None

  


class PrintSomeValues(keras.callbacks.Callback):
    def __init__(self,model,x_test,y_test2,y_test,quantile):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.y_test2 = y_test2
        self.__quantile = quantile
        
    def on_train_begin(self, logs={}):
        self.pinball_flag=100
    
    
    def on_epoch_begin(self, epoch, logs={}):
        QApplication.processEvents()
        lr = k.get_value(self.model.optimizer.lr)
        print("current learning rate is {}".format(lr))
        pred = self.model.predict(self.x_test)*self.y_test2
        test = self.y_test*self.y_test2

        predict_all = pred.flatten()
        truth_all = test.flatten()
        
       
        pinball=pinball_score(test,pred,self.__quantile)
        prediction=pred
        truth=test[:,0]
        low = int((self.__quantile+1)*0.3-1)
        high = int((self.__quantile+1)*0.7-1)
        low1 = int((self.__quantile+1)*0.1-1)
        high1 =int((self.__quantile+1)*0.9-1)

        winkler40 = winkler_score(truth, prediction[:,low],prediction[:,high], 0.4)
        winkler80 = winkler_score(truth, prediction[:,low1],prediction[:,high1], 0.8)
        pcip40 = PCIP(truth, prediction[:,low],prediction[:,high], 0.4)
        pcip80 = PCIP(truth, prediction[:,low1],prediction[:,high1], 0.8)
        ace40 = np.abs(pcip40 - 0.4)
        ace80 = np.abs(pcip80 - 0.8)
        print("After %d training step(s),"
             "on test data piball_loss = %.4f,winkler40 = %.4f,winkler80 = %.4f,pcip40 = %.4f,pcip80 = %.4f,ace40 = %.4f,ace80 = %.4f"\
             % (epoch, pinball,winkler40,winkler80,pcip40,pcip80,ace40,ace80))
            
        
    

    

    
