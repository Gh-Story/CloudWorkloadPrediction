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

def exponentialSmoothing(rawData,alpha):
    '''
    ES
    :param alpha:   smoothing coefficient
    :param rawData:    original data
    :return:    exponential smoothing data
    '''
    s_temp=[]
    s_temp.append(rawData[0])
    for i in range(1,len(rawData),1):
        s_temp.append(alpha * rawData[i] + (1 - alpha) * s_temp[i-1])
    return s_temp

def minMaxNormalization(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))

def normalization(data,esdata):
    return data/esdata

def preprocess(filePath,fileName,esDataSavePath,newDataSavePath):
    
    fileName = fileName
    filePath = filePath
    df=pd.read_excel(filePath+fileName,header=None)
    data=np.array(df)
    
    esdata=exponentialSmoothing(data,0.1)
    esdata=np.array(esdata).reshape(len(data))
    newdata = normalization(data.reshape(-1),esdata)

    newdata = pd.DataFrame(newdata)
    newdata.to_excel(newDataSavePath+fileName,header=False,index=False)
    esdata = pd.DataFrame(esdata)
    esdata.to_excel(esDataSavePath+fileName,header=False,index=False)
    return

def pinball_loss(y_true, y_pred):
    p=0
    q1=tf.linspace(0.1,0.9,9)
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



def build_model(num_steps,num_features):
    input_layer = Input(name='input_layer',shape=(num_steps,num_features))
    lstm1 = Bidirectional(LSTM(128,name='lstm_layer1',return_sequences=True, activation='relu'))(input_layer)
    lstm2 = Bidirectional(LSTM(128,name='lstm_layer2',return_sequences=False, activation='relu'))(lstm1)
    output_layer = Dense(9,name='output_layer',kernel_regularizer=regularizers.l2(0.001))(lstm2)
    model = Model(input_layer,output_layer)
    adam = Adam(lr=0.0005, clipnorm=1.0,decay=0.1,amsgrad=True)
    model.compile(loss=pinball_loss, optimizer=adam)
    return model


def read_file(fileName,days):
    data = np.array(pd.read_excel(fileName,header=None))
    datasize=len(data)
    step=int(datasize/days)
    x=[]
    y=[]

    for i in range(datasize-step):
        x.append(data[i:i+step].tolist())
        y.append(data[i+step].tolist()*9)
    return x,y,len(x)    


class PrintSomeValues(keras.callbacks.Callback):
    def __init__(self,model,x_test,y_test2,y_test):
        self.model = model
        self.x_test = x_test
        self.y_test = y_test
        self.y_test2 = y_test2
        
    def on_train_begin(self, logs={}):
        self.pinball_flag=100
    
    
    def on_epoch_begin(self, epoch, logs={}):
        lr = k.get_value(self.model.optimizer.lr)
        print("current learning rate is {}".format(lr))
        pred = self.model.predict(self.x_test)*self.y_test2
        test = self.y_test*self.y_test2

        predict_all = pred.flatten()
        truth_all = test.flatten()
        
       
        pinball=pinball_score(test,pred,9)
        prediction=pred
        truth=test[:,0]
        winkler40 = winkler_score(truth, prediction[:,2],prediction[:,6], 0.4)
        winkler80 = winkler_score(truth, prediction[:,0],prediction[:,8], 0.8)
        pcip40 = PCIP(truth, prediction[:,2],prediction[:,6], 0.4)
        pcip80 = PCIP(truth, prediction[:,0],prediction[:,8], 0.8)
        ace40 = np.abs(pcip40 - 0.4)
        ace80 = np.abs(pcip80 - 0.8)
        print("After %d training step(s),"
             "on test data piball_loss = %.4f,winkler40 = %.4f,winkler80 = %.4f,pcip40 = %.4f,pcip80 = %.4f,ace40 = %.4f,ace80 = %.4f"\
             % (epoch, pinball,winkler40,winkler80,pcip40,pcip80,ace40,ace80))
            
        if(epoch%20==0):
            truth_all_reshape=np.reshape(truth_all,[-1,1])
            predict_all_reshape=np.reshape(predict_all,[-1,1])
            y_con = np.concatenate((truth_all_reshape, predict_all_reshape), axis=1)
            #输出真实值和预测值
            y_out = pd.DataFrame(y_con, columns=["true_data","pre_data"])
            y_out.to_excel('../ES-QBLSTM/ProbabilityTrain/'+Name+'%d-piball_loss = %.4f,winkler40 = %.4f,winkler80 = %.4f,pcip40 = %.4f,pcip80 = %.4f,ace40 = %.4f,ace80 = %.4f.xlsx'\
            % (epoch,pinball,winkler40,winkler80,pcip40,pcip80,ace40,ace80))
        


def dataSlice(newDataPath,esDataPath,fileName,days):
    file = newDataPath+fileName
    file2 = esDataPath+fileName
        
     
    x_raw,y_raw,size = read_file(file,days)
    x_raw2,y_raw2,size2 = read_file(file2,days)
    train_data = int(size * 0.8)
    val_data = int(size * 0.1)

    xtrain = x_raw[0:train_data]
    ytrain = y_raw[0:train_data]
    x_train=np.array(xtrain)
    y_train=np.array(ytrain).reshape(-1,9)

    xval = x_raw[train_data:train_data+val_data]
    yval = y_raw[train_data:train_data+val_data]
    x_val=np.array(xval)
    y_val = np.array(yval).reshape(-1,9)

    xtest = x_raw[train_data+val_data:size]
    ytest = y_raw[train_data+val_data:size]
    x_test=np.array(xtest)
    y_test=np.array(ytest).reshape(-1,9)

    xtest2 = x_raw2[train_data+val_data:size]
    ytest2 = y_raw2[train_data+val_data:size]
    x_test2=np.array(xtest2)
    y_test2=np.array(ytest2).reshape(-1,9)

    return x_train,y_train,x_val,y_val,x_test,y_test,y_test2
    
    
def trainModel(x_train,y_train,x_val,y_val,x_test,y_test,y_test2,Name):
    model = build_model(num_steps=x_train.shape[1], num_features=x_train.shape[2])


    print(f'x_train.shape = {x_train.shape}')
    print(f'y_train.shape = {y_train.shape}')

    psv = PrintSomeValues(model,x_test,y_test2,y_test)

        # Using sparse softmax.
        # http://chappers.github.io/web%20micro%20log/2017/01/26/quick-models-in-keras/
    model.summary()


    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.3,patience=10, mode='min')

    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=50, verbose=2,mode='min')

        #for i in range(1000):
            #callbacks=[psv]
    model.fit(x_train, y_train, 
            validation_data=(x_val, y_val),
            epochs=220, 
            batch_size=100,
            initial_epoch=0,
            callbacks=[early_stopping,reduce_lr, psv]
            )
    return model


def prediction(model,x_test,y_test2,y_test,Name,resultPath):
    pred = model.predict(x_test)*y_test2
    test = y_test*y_test2

    predict_all = pred.flatten()
    truth_all = test.flatten()
        
       
    pinball=pinball_score(test,pred,9)
    prediction=pred
    truth=test[:,0]
    winkler40 = winkler_score(truth, prediction[:,2],prediction[:,6], 0.4)
    winkler80 = winkler_score(truth, prediction[:,0],prediction[:,8], 0.8)
    pcip40 = PCIP(truth, prediction[:,2],prediction[:,6], 0.4)
    pcip80 = PCIP(truth, prediction[:,0],prediction[:,8], 0.8)
    ace40 = np.abs(pcip40 - 0.4)
    ace80 = np.abs(pcip80 - 0.8)
        
    
    truth_all_reshape=np.reshape(truth_all,[-1,1])
    predict_all_reshape=np.reshape(predict_all,[-1,1])
    y_con = np.concatenate((truth_all_reshape, predict_all_reshape), axis=1)
        #输出真实值和预测值
    y_out = pd.DataFrame(y_con, columns=["true_data","pre_data"])
    y_out.to_excel(resultPath+Name+'-piball_loss = %.4f,winkler40 = %.4f,winkler80 = %.4f,pcip40 = %.4f,pcip80 = %.4f,ace40 = %.4f,ace80 = %.4f.xlsx'\
        % (pinball,winkler40,winkler80,pcip40,pcip80,ace40,ace80))
    
filePath = '../dataset/'
newDataPath = '../new dataset/'
esDataPath = '../es dataset/'
resultPath = '../ES-QBLSTM/ProbabilityResult/'
fileName = 'M1_per_15min.xlsx'
days = 29
Name = 'M1_per_15min'

preprocess(filePath,fileName,esDataPath,newDataPath)
x_train,y_train,x_val,y_val,x_test,y_test,y_test2 = dataSlice(newDataPath,esDataPath,fileName,days)
model = trainModel(x_train,y_train,x_val,y_val,x_test,y_test,y_test2,Name)
prediction(model,x_test,y_test2,y_test,Name,resultPath)
os.system("pause")
