from src.iris import *
import random
import numpy as np

if __name__=="__main__":
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
  random.seed(5)
  solve_and_tune_iris("./iris.csv")
