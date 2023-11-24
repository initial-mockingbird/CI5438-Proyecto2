import pandas as pd
import numpy as np
import random
from src.nn import *
from src.config.preprocessing import *
from math import nan
from math import floor
import matplotlib.pyplot as plt
import matplotlib as mpl

random.seed(5)

def min_max_normalize(df):
  df_max = df.max()
  df_min = df.min()
  df = (df - df_min) / (df_max - df_min)
  return df

def iris_parsing(path):
  col_dict = [
    { "column"        : "sepal_length"
    , "type"          : "numeric"
    , "categoric"     : False
    , "fill_strategy" : "REMOVE"
    },
    { "column"        : "sepal_width"
    , "type"          : "numeric"
    , "categoric"     : False
    , "fill_strategy" : "REMOVE"
    },
    { "column"        : "petal_width"
    , "type"          : "numeric"
    , "categoric"     : False
    , "fill_strategy" : "REMOVE"
    },
    { "column"        : "species"
    , "type"          : "string"
    , "categoric"     : True
    , "fill_strategy" : "REMOVE"
    }
  ]
  return builder(path,"species",True,None,col_dict)

def iris(path):
  iris_csv = iris_parsing(path)
  targets = [f"species:{s}" for s in iris_csv["species"].unique()]
  iris_csv = iris_csv.drop("species",axis=1)
  

  
  train=iris_csv.sample(frac=0.8,random_state=200)
  train_t = train.copy()[targets].to_numpy()
  test=iris_csv.copy().drop(train.index)
  
  train = train.drop(targets,axis=1).to_numpy()
  test_t = test.copy()[targets].to_numpy()
  test = test.drop(targets,axis=1).to_numpy()

  def cont(f,learning_rate=0.1):
    return f(train,train_t,test,test_t,targets,learning_rate)

  return cont


def iris_binary(path,targets):
  ts = ["species:Iris-setosa","species:Iris-versicolor","species:Iris-virginica"]
  iris_csv = iris_parsing(path)
  iris_csv = iris_csv.drop("species",axis=1)
  train=iris_csv.sample(frac=0.8,random_state=200)
  train_t = train.copy()[targets].to_numpy()
  test=iris_csv.copy().drop(train.index)
  
  train = train.drop(ts,axis=1).to_numpy()
  test_t = test.copy()[targets].to_numpy()
  test = test.drop(ts,axis=1).to_numpy()

  def cont(f,learning_rate=0.1):
    return f(train,train_t,test,test_t,targets,learning_rate)

  return cont

def iris_binary_setosa(train,train_t,test,test_t,targets,learning_rate):
  train = train.astype(float)
  train_t = train_t.astype(float) 
  test = test.astype(float)
  test_t = test_t.astype(float)


  nn = NN(train,train_t,learning_rate,10000)
  nn.add_layer(Lineal_Layer(3,1))
  #nn.add_output_layer(SoftMaxOutput())
  nn.add_output_layer(LogisticOutput())
  (train_cost,batch_cost) = nn.train()
  predictions = nn.predict(test)

  delta = predictions - test_t
  df = pd.DataFrame(delta, columns=targets)
  
  return (train_cost,resultados(df,targets,learning_rate,0.2,""))

def iris_binary_one_layer(train,train_t,test,test_t,targets,learning_rate):
  train = train.astype(float)
  train_t = train_t.astype(float) 
  test = test.astype(float)
  test_t = test_t.astype(float)


  nn = NN(train,train_t,learning_rate,10000)
  nn.add_layer(Lineal_Layer(3,5))
  nn.add_layer(Logistic())
  nn.add_layer(Lineal_Layer(5,1))
  nn.add_output_layer(LogisticOutput())
  (train_cost,batch_cost) = nn.train()
  predictions = nn.predict(test)

  delta = predictions - test_t
  df = pd.DataFrame(delta, columns=targets)
  
  return (train_cost,resultados(df,targets,learning_rate,0.2,""))

def iris_binary_two_layer(train,train_t,test,test_t,targets,learning_rate):
  train = train.astype(float)
  train_t = train_t.astype(float) 
  test = test.astype(float)
  test_t = test_t.astype(float)


  nn = NN(train,train_t,learning_rate,10000)
  nn.add_layer(Lineal_Layer(3,5))
  nn.add_layer(Logistic())
  nn.add_layer(Lineal_Layer(5,5))
  nn.add_layer(Logistic())
  nn.add_layer(Lineal_Layer(5,1))
  nn.add_output_layer(LogisticOutput())
  (train_cost,batch_cost) = nn.train()
  predictions = nn.predict(test)

  delta = predictions - test_t
  df = pd.DataFrame(delta, columns=targets)
  
  return (train_cost,resultados(df,targets,learning_rate,0.2,""))

def iris_multiclass_single_layer(train,train_t,test,test_t,targets,learning_rate):
  train = train.astype(float)
  train_t = train_t.astype(float)
  test = test.astype(float)
  test_t = test_t.astype(float)

  nn = NN(train,train_t,learning_rate,10000)
  nn.add_layer(Lineal_Layer(3,4))
  nn.add_layer(Logistic())
  nn.add_layer(Lineal_Layer(4,3))
  nn.add_output_layer(SoftMaxOutput())
  (train_cost,batch_cost) = nn.train()
  predictions = nn.predict(test)

  delta = predictions - test_t
  df = pd.DataFrame(delta, columns=targets)
  
  return (train_cost, resultados(df,targets,learning_rate,output=""))

def iris_multiclass_single_layer_logistic(train,train_t,test,test_t,targets,learning_rate):
  train = train.astype(float)
  train_t = train_t.astype(float)
  test = test.astype(float)
  test_t = test_t.astype(float)

  nn = NN(train,train_t,learning_rate,10000)
  nn.add_layer(Lineal_Layer(3,5))
  nn.add_layer(Logistic())
  nn.add_layer(Lineal_Layer(5,3))
  nn.add_output_layer(LogisticOutput())
  (train_cost,batch_cost) = nn.train()
  predictions = nn.predict(test)

  delta = predictions - test_t
  df = pd.DataFrame(delta, columns=targets)
  
  return (train_cost, resultados(df,targets,learning_rate,output=""))

def iris_multiclass_two_layer(train,train_t,test,test_t,targets,learning_rate):
  train = train.astype(float) 
  train_t = train_t.astype(float) 
  test = test.astype(float) 
  test_t = test_t.astype(float) 

  nn = NN(train,train_t,learning_rate,10000)
  nn.add_layer(Lineal_Layer(3,4))
  nn.add_layer(Logistic())
  nn.add_layer(Lineal_Layer(4,5))
  nn.add_layer(Logistic())
  nn.add_layer(Lineal_Layer(5,3))
  nn.add_output_layer(SoftMaxOutput())
  (train_cost,batch_cost) = nn.train()
  predictions = nn.predict(test)

  delta = predictions - test_t
  df = pd.DataFrame(delta, columns=targets)
  return (train_cost,resultados(df,targets,learning_rate,output=""))

def resultados(df,targets,learning_rate,threshold=1e-7,output="console"):
  totales = len(df)
  results = []
  for target in targets:
    true_positive = len(df[(df[target] >= 0) & (df[target] < threshold)])
    true_negative = len(df[(df[target] <= 0) & (df[target] > -threshold)])
    false_positive = len(df[df[target] > threshold])
    false_negative = len(df[df[target] < -threshold])
    predichos      = len(df[np.abs(df[target]) < threshold])
    accuracy = predichos/totales
    try:
      precision = true_positive / (true_positive + false_positive)
    except:
      precision = nan
    try:
      recall = true_positive / (true_positive + false_negative)
    except:
      recall = nan
    s = "=============================================\n"
    s += f"learning_rate: {learning_rate}\n"
    s += f"Para: {target}\n"
    s += f"totales: {totales}\n"
    s += f"falsos_positivos: {false_positive}\n"
    s += f"falsos_negativos: {false_negative}\n"
    s += f"verdaderos positivos: {true_positive}\n"
    s += f"verdaderos negativos: {true_negative}\n"
    s += f"predichos correctamente: {predichos}\n"
    s += f"accuracy: {accuracy}\n"
    s += f"precision: {precision}\n"
    s += f"recall: {recall}\n"
    if (output=="console"):
      print(s)
    results.append((target,accuracy,precision,recall,false_positive,false_negative))
  return results

def hex_to_RGB(hex_str):
  """ #FFFFFF -> [255,255,255]"""
  #Pass 16 to the integer function for change of base
  return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]

def get_color_gradient(c1, c2, n):
  """
  Given two hex colors, returns a color gradient
  with n colors.
  """
  assert n > 1
  c1_rgb = np.array(hex_to_RGB(c1))/255
  c2_rgb = np.array(hex_to_RGB(c2))/255
  mix_pcts = [x/(n-1) for x in range(n)]
  rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
  return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]

def plot_errors_its(errores,titulo,nombre):
  plt.style.use('seaborn-darkgrid')
  fig, ax = plt.subplots()
  
  colors = get_color_gradient("#E81123","#FFF100",len(errores)) if len(errores) > 1 else "blueviolet"
  for (i,r0) in enumerate(errores):
    (a,r) = r0
    (cost,_) = r
    x = np.array([ i + 1 for i in range(len(cost))])
    ax.plot(x, cost, marker='', color=colors[i], linewidth=2, alpha=0.7,label=f"a={a:.3f}")
  ax.set_title(titulo,fontsize=12, fontweight=1)
  ax.set_ylabel("error")
  ax.set_xlabel("iteracion")
  ax.legend()
  plt.savefig(f"imgs/{nombre}.PNG")


def plot_measure(_info,attr,titulo,nombre):
  plt.style.use('seaborn-darkgrid')
  fig, ax = plt.subplots()
  info = _info.copy().sort_values(by=[attr])
  # barras = target distinto
  # stack  = learning rate distinto
  # eje y  = attributo 
  lr = info["lr"].unique()
  targets = info["target"].unique()
  stacks = pd.DataFrame(dict([(f"{_lr:.3f}",list(info[info["lr"] == _lr][attr])) for _lr in lr]))
  colors = get_color_gradient("#E81123","#FFF100",len(lr)) if len(lr) > 1 else "blueviolet"

  fig, ax = plt.subplots()
  bottom = np.zeros(len(targets))

  for lr, r in stacks.items():
    p = ax.bar(targets, r, 0.4, label=lr, bottom=bottom)
    #bottom += r
  
  ax.set_title(titulo,fontsize=12, fontweight=1)
  ax.legend(loc="upper right")
  
  """ plt.bar(alphas, attrs, color = colors, width = 0.4)
  ax.set_title(titulo,fontsize=12, fontweight=1)
  ax.set_ylabel(nombre) """
  plt.savefig(f"imgs/{nombre}.PNG")


def solve_and_tune_iris(path):
  def max(xs):
    try:
      return list(map(np.max,xs))
    except:
      return xs
  def min(xs):
    try:
      return list(map(np.min,xs))
    except:
      return xs
  def avr(xs):
    try:
      return list(map(np.average,xs))
    except: 
      return xs
  
  _results_multiclass_2 = []
  _results_multiclass_1 = []
  _results_single_setosa = []
  _results_single_versicolor = []
  _results_single_virginica = []
  _results_single_1l_setosa = []
  _results_single_1l_versicolor = []
  _results_single_1l_virginica = []
  _results_single_2l_setosa = []
  _results_single_2l_versicolor = []
  _results_single_2l_virginica = []
  _results_multiclass_1_logistic = []


  # multiclass two layer
  for i in np.arange(0.001,0.7,0.05):
    _results_multiclass_2.append((i,iris(path)(iris_multiclass_two_layer,i)))
    _results_multiclass_1.append((i,iris(path)(iris_multiclass_single_layer,i)))
    # ["species:Iris-setosa","species:Iris-versicolor","species:Iris-virginica"]
    _results_single_setosa.append((i,iris_binary(path,["species:Iris-setosa"])(iris_binary_setosa,i)))
    _results_single_versicolor.append((i,iris_binary(path,["species:Iris-versicolor"])(iris_binary_setosa,i)))
    _results_single_virginica.append((i,iris_binary(path,["species:Iris-virginica"])(iris_binary_setosa,i)))
    _results_single_1l_setosa.append((i,iris_binary(path,["species:Iris-setosa"])(iris_binary_one_layer,i)))
    _results_single_1l_versicolor.append((i,iris_binary(path,["species:Iris-versicolor"])(iris_binary_one_layer,i)))
    _results_single_1l_virginica.append((i,iris_binary(path,["species:Iris-virginica"])(iris_binary_one_layer,i)))
    _results_single_2l_setosa.append((i,iris_binary(path,["species:Iris-setosa"])(iris_binary_two_layer,i)))
    _results_single_2l_versicolor.append((i,iris_binary(path,["species:Iris-versicolor"])(iris_binary_two_layer,i)))
    _results_single_2l_virginica.append((i,iris_binary(path,["species:Iris-virginica"])(iris_binary_two_layer,i)))
    _results_multiclass_1_logistic.append((i,iris(path)(iris_multiclass_single_layer_logistic,i)))
    

  plot_errors_its(_results_multiclass_1_logistic,"Errores por iteracion","mc_1c_log.PNG")
  aux = [ (lr,tc,target,acc,p,r,falso_positivo,falso_negativo) for (lr,(tc,result)) in _results_multiclass_1_logistic for (target,acc,p,r,falso_positivo,falso_negativo) in result]
  targets    =  pd.DataFrame(aux,columns=["lr","tc","target","acc","p","r","falso_positivo","falso_negativo"])
  plot_measure(targets,"acc","Accuracy para multiclass de 1 capa","mc_1c_log_acc.PNG")
  grouped = targets.groupby(by=["lr","target"])
  grouped.agg([max,min,avr]).to_csv("./results/mc_1c_log.csv")
  
  plot_errors_its(_results_multiclass_2,"Errores por iteracion","mc_2c.PNG")
  aux = [ (lr,tc,target,acc,p,r,falso_positivo,falso_negativo) for (lr,(tc,result)) in _results_multiclass_2 for (target,acc,p,r,falso_positivo,falso_negativo) in result]
  targets    =  pd.DataFrame(aux,columns=["lr","tc","target","acc","p","r","falso_positivo","falso_negativo"])
  plot_measure(targets,"acc","Accuracy para multiclass de 2 capas","mc_2c_acc.PNG")
  grouped = targets.groupby(by=["lr","target"])
  grouped.agg([max,min,avr]).to_csv("./results/mc_2c.csv")
  


  plot_errors_its(_results_multiclass_1,"Errores por iteracion","mc_1c.PNG")
  aux = [ (lr,tc,target,acc,p,r,falso_positivo,falso_negativo) for (lr,(tc,result)) in _results_multiclass_1 for (target,acc,p,r,falso_positivo,falso_negativo) in result]
  targets    =  pd.DataFrame(aux.copy(),columns=["lr","tc","target","acc","p","r","falso_positivo","falso_negativo"])
  plot_measure(targets,"acc","Accuracy para multiclass de 1 capa","mc_1c_acc.PNG")
  grouped = targets.groupby(by=["lr","target"])
  grouped.agg([max,min,avr]).to_csv("./results/mc_1c.csv")
  

  plot_errors_its(_results_single_setosa,"Errores por iteracion","sc_setosa.PNG")
  aux = [ (lr,tc,target,acc,p,r,falso_positivo,falso_negativo) for (lr,(tc,result)) in _results_single_setosa for (target,acc,p,r,falso_positivo,falso_negativo) in result]
  targets    =  pd.DataFrame(aux.copy(),columns=["lr","tc","target","acc","p","r","falso_positivo","falso_negativo"])
  plot_measure(targets,"acc","Accuracy para clasificador binario setosa","sc_setosa_acc.PNG")
  grouped = targets.groupby(by=["lr"])
  grouped.agg([max,min,avr]).to_csv("./results/sc_setosa.csv")

  plot_errors_its(_results_single_versicolor,"Errores por iteracion","sc_versicolor.PNG")
  aux = [ (lr,tc,target,acc,p,r,falso_positivo,falso_negativo) for (lr,(tc,result)) in _results_single_versicolor for (target,acc,p,r,falso_positivo,falso_negativo) in result]
  targets    =  pd.DataFrame(aux.copy(),columns=["lr","tc","target","acc","p","r","falso_positivo","falso_negativo"])
  plot_measure(targets,"acc","Accuracy para clasificador binario versicolor","sc_versicolor_acc.PNG")
  grouped = targets.groupby(by=["lr"])
  grouped.agg([max,min,avr]).to_csv("./results/sc_versicolor.csv")

  plot_errors_its(_results_single_virginica,"Errores por iteracion","sc_virginica.PNG")
  aux = [ (lr,tc,target,acc,p,r,falso_positivo,falso_negativo) for (lr,(tc,result)) in _results_single_virginica for (target,acc,p,r,falso_positivo,falso_negativo) in result]
  targets    =  pd.DataFrame(aux.copy(),columns=["lr","tc","target","acc","p","r","falso_positivo","falso_negativo"])
  plot_measure(targets,"acc","Accuracy para clasificador binario virginica","sc_virginica_acc.PNG")
  grouped = targets.groupby(by=["lr"])
  grouped.agg([max,min,avr]).to_csv("./results/sc_virginica.csv") 

  plot_errors_its(_results_single_1l_setosa,"Errores por iteracion","sc_l1_setosa.PNG")
  aux = [ (lr,tc,target,acc,p,r,false_positive,false_negative) for (lr,(tc,result)) in _results_single_1l_setosa for (target,acc,p,r,false_positive,false_negative) in result]
  targets    =  pd.DataFrame(aux.copy(),columns=["lr","tc","target","acc","p","r","false_positive","false_negative"])
  plot_measure(targets,"acc","Accuracy para clasificador binario setosa Una capa oculta","sc_setosa_l1_acc.PNG")
  grouped = targets.groupby(by=["lr"])
  grouped.agg([max,min,avr]).to_csv("./results/sc_setosa_l1.csv")

  plot_errors_its(_results_single_1l_versicolor,"Errores por iteracion","sc_l1_versicolor.PNG")
  aux = [ (lr,tc,target,acc,p,r,false_positive,false_negative) for (lr,(tc,result)) in _results_single_1l_versicolor for (target,acc,p,r,false_positive,false_negative) in result]
  targets    =  pd.DataFrame(aux.copy(),columns=["lr","tc","target","acc","p","r","false_positive","false_negative"])
  plot_measure(targets,"acc","Accuracy para clasificador binario versicolor Una capa oculta","sc_versicolor_l1_acc.PNG")
  grouped = targets.groupby(by=["lr"])
  grouped.agg([max,min,avr]).to_csv("./results/sc_versicolor_l1.csv")

  plot_errors_its(_results_single_1l_virginica,"Errores por iteracion","sc_l1_virginica.PNG")
  aux = [ (lr,tc,target,acc,p,r,false_positive,false_negative) for (lr,(tc,result)) in _results_single_1l_virginica for (target,acc,p,r,false_positive,false_negative) in result]
  targets    =  pd.DataFrame(aux.copy(),columns=["lr","tc","target","acc","p","r","false_positive","false_negative"])
  plot_measure(targets,"acc","Accuracy para clasificador binario virginica Una capa oculta","sc_virginica_l1_acc.PNG")
  grouped = targets.groupby(by=["lr"])
  grouped.agg([max,min,avr]).to_csv("./results/sc_virginica_l1.csv")


  plot_errors_its(_results_single_2l_setosa,"Errores por iteracion","sc_l2_setosa.PNG")
  aux = [ (lr,tc,target,acc,p,r,false_positive,false_negative) for (lr,(tc,result)) in _results_single_2l_setosa for (target,acc,p,r,false_positive,false_negative) in result]
  targets    =  pd.DataFrame(aux.copy(),columns=["lr","tc","target","acc","p","r","false_positive","false_negative"])
  plot_measure(targets,"acc","Accuracy para clasificador binario setosa Dos capas ocultas","sc_setosa_l2_acc.PNG")
  grouped = targets.groupby(by=["lr"])
  grouped.agg([max,min,avr]).to_csv("./results/sc_setosa_l2.csv")

  plot_errors_its(_results_single_2l_versicolor,"Errores por iteracion","sc_l2_versicolor.PNG")
  aux = [ (lr,tc,target,acc,p,r,false_positive,false_negative) for (lr,(tc,result)) in _results_single_2l_versicolor for (target,acc,p,r,false_positive,false_negative) in result]
  targets    =  pd.DataFrame(aux.copy(),columns=["lr","tc","target","acc","p","r","false_positive","false_negative"])
  plot_measure(targets,"acc","Accuracy para clasificador binario versicolor Dos capas ocultas","sc_versicolor_l2_acc.PNG")
  grouped = targets.groupby(by=["lr"])
  grouped.agg([max,min,avr]).to_csv("./results/sc_versicolor_l2.csv")

  plot_errors_its(_results_single_2l_virginica,"Errores por iteracion","sc_l2_virginica.PNG")
  aux = [ (lr,tc,target,acc,p,r,false_positive,false_negative) for (lr,(tc,result)) in _results_single_2l_virginica for (target,acc,p,r,false_positive,false_negative) in result]
  targets    =  pd.DataFrame(aux.copy(),columns=["lr","tc","target","acc","p","r","false_positive","false_negative"])
  plot_measure(targets,"acc","Accuracy para clasificador binario virginica Dos capas ocultas","sc_virginica_l2_acc.PNG")
  grouped = targets.groupby(by=["lr"])
  grouped.agg([max,min,avr]).to_csv("./results/sc_virginica_l2.csv")
