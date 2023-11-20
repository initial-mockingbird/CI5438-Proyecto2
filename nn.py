import pandas as pd
import numpy as np
import random
from src.nn import *

if __name__=="__main__":
  random.seed(5)
  def min_max_normalize(df):
    df_max = df.max()
    df_min = df.min()
    df = (df - df_min) / (df_max - df_min)
    return df
  observations = int(1e4)

  # w0 = 1
  # w1 = 20
  # w2 = 50

  f = lambda x,y : 1 + 20 * x + 50 * y

  #w0T = [1 for _ in range(observations)]
  w1T = [random.uniform(0,1) for _ in range(observations)]
  w2T = [random.uniform(0,1) for _ in range(observations)]

  T_train = np.array([f(x,y) for (x,y) in zip(w1T,w2T)])
  X_train = np.array([w1T,w2T]).T

  tests = int(1e4)
  w1Test = [random.uniform(0,1) for _ in range(tests)]
  w2Test = [random.uniform(0,1) for _ in range(tests)]

  T_Test = np.array([f(x,y) for (x,y) in zip(w1Test,w2Test)])
  X_Test = np.array([w1Test,w2Test]).T

  T_train = min_max_normalize(T_train.reshape(len(T_train),1))
  T_Test = min_max_normalize(T_Test.reshape(len(T_Test),1))

  nn = NN(X_train,T_train,0.1,1000)
  nn.add_layer(Lineal_Layer(2,3))
  nn.add_layer(Logistic())
  nn.add_layer(Lineal_Layer(3,3))
  nn.add_layer(Logistic())
  nn.add_layer(Lineal_Layer(3,1))
  nn.add_output_layer(OutputLayer(Logistic(),"mse"))
  nn.train()
  predictions = nn.predict(X_Test)
  print(predictions - T_Test)

  
  rss = sum(np.square(predictions - T_Test))
  tss = sum(np.square(predictions - np.average(predictions)))
  r2_score = 1 - rss/tss
  print(r2_score)