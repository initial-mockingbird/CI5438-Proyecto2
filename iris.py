from src.iris import *
import random
import numpy as np

if __name__=="__main__":
  random.seed(5)
  solve_and_tune_iris("./iris.csv")
