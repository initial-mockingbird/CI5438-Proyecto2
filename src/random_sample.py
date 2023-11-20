import random
import pandas as pd

random.seed(5)

observations = int(1e4)

# w0 = 1
# w1 = 20
# w2 = 50

f = lambda x,y : 1 + 20 * x + 50 * y

w1s = [random.uniform(0,1) for _ in range(observations)]
w2s = [random.uniform(0,1) for _ in range(observations)]
ys = [f(x,y) for (x,y) in zip(w1s,w2s)]
testing_data = pd.DataFrame(\
  { "w1s": w1s
  , "w2s": w2s
  , "ys": ys
  }
)
