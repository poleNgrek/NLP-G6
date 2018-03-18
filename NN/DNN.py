import pandas as pd
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
import random
from keras.layers import Embedding

class Classifier(object):

    def __init__(self,uniques,n):
        self.uniques=uniques
        self.n=n
        self.vacab=['<UNSEEN>']
        self.model=Sequential()
        self.model.add(Embedding(50000,int(np.log(50000)+1),input_length=self.n))
        self.model.add(Flatten())
        self.model.add(Dense(6))
        self.model.add(Activation('sigmoid'))
        self.model.add(Dense(3))
        self.model.add(Activation('softmax'))
        self.model.summary()
        self.model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

    def predict(self,x,candidate=None):
        ret=[]
        for i in x:
            if i in self.vacab:
                ret.append(self.vacab.index(i))
            else:
                ret.append(self.vacab.index('<UNSEEN>'))

        #temp=self.model.predict_on_batch(np.array([ret for i in range(self.n)]))[0].tolist()
        temp=self.model.predict_on_batch(np.array([ret]))[0].tolist()
        ret=self.uniques[temp.index(max(temp))]

        if ret not in candidate:
            if len(candidate)==1:
                return candidate[0]
            elif len(candidate)==2:
                return candidate[random.randint(0,1)]
        else:
            return ret

    def update(self,x,y):
        ret=[]
        for i in x:
            if i not in self.vacab:
                self.vacab.append(i)
            ret.append(self.vacab.index(i))
        #self.model.train_on_batch(np.array([ret for i in range(self.n)]),np.array([y for i in range(self.n)]))
        self.model.train_on_batch(np.array([ret]),np.array([y]))
