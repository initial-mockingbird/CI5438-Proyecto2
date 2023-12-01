import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import src.config.preprocessing as pp
from iris import get_color_gradient
import src.spam_cols as spam
from src.nn import *
import pandas as pd
import numpy as np
import sys

import matplotlib.pyplot as plt

# only works for a single column output!
def accuracy(Y_pred, Y_true):
    tp, fp, fn = 0, 0, 0
    for i in range(0, len(Y_pred)):
        p, r = Y_pred[i], Y_true[i]
        if p == r and p == 1:
            tp += 1; continue

        if p == 1 and r == 0: 
            fp += 1; continue

        if p == 0 and r == 1:
            fn += 1; continue


    print(f"True Positives {tp}") 
    print(f"False Positives {fp}")
    print(f"False Negatives {fn}")

    try:
        acc = tp / (tp + fp)
    except:
        acc = 0

    try:
        rec = tp / (tp + fn)
    except:
        rec = 0

    return acc, rec


def do_experiment(data, inputs, targets, hidden_layers, learn_rate=[0.1], iterations = 5000, out_name = "fig"):
    colors = get_color_gradient("#E81123","#FFF100",len(learn_rate)) if len(learn_rate) > 1 else "blueviolet"
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.title("Error por iteracion")
    plt.ylabel("Log Loss")
    plt.xlabel("Iteracion")

    data_train, data_test = np.split(data, [int(0.7 * len(data))])

    X_train = data_train.drop(labels=targets, axis=1)[inputs]
    T_train = data_train.take(targets, axis=1)
    T_train = T_train.to_numpy().reshape(len(T_train),len(targets))

    X_test = data_test.drop(labels=targets, axis=1)[inputs]
    T_test = data_test[targets[0]].to_numpy()

    for i in range(0, len(learn_rate)):
        lr = learn_rate[i]
        net = NN(X_train, T_train, lr, max_iter = iterations)
        if len(hidden_layers) != 0:
            net.add_layer(Lineal_Layer(len(inputs), hidden_layers[0]))
        else:
            net.add_layer(Lineal_Layer(len(inputs), 1))
        
        for i in range(0, len(hidden_layers) - 1):
            net.add_layer(Logistic())
            net.add_layer(Lineal_Layer(hidden_layers[i], hidden_layers[i+1]))
        
        if len(hidden_layers) != 0:
            net.add_layer(Logistic())
            net.add_layer(Lineal_Layer(hidden_layers[-1], 1))

        net.add_output_layer(LogisticOutput())

        train_cost, _ = net.train()

        predictions = round(net.predict(X_test.to_numpy())).to_numpy().reshape(X_test.shape[0])
        acc, rec = accuracy(predictions, T_test)

        print(f'Accuracy {acc} Recall {rec} Learning Rate {lr}')

        plt.plot(
            [i for i in range(0, len(train_cost))],
            train_cost,
            marker='',
            color=colors[i],
            linewidth=4, 
            alpha=0.7,
            label=f"a={lr:.3f}"
        )

    plt.legend()
    plt.savefig(out_name)
    plt.clf()
    

if __name__ != '__main__':
    sys.exit()

random.seed(5)

spam_df = pp.min_max_normalize(
    pd.read_csv("./spambase.data", 
                header=None, 
                engine='c', 
                na_values=[""])).sample(frac=1)


all_hyp = [i for i in range(spam.word_freq_make, spam.is_spam)]
sus_hyp = [
    spam.word_freq_address,
    spam.word_freq_internet,
    spam.word_freq_order,
    spam.word_freq_mail,
    spam.word_freq_receive,
    spam.word_freq_free,
    spam.word_freq_business,
    spam.word_freq_email,
    spam.word_freq_credit,
    spam.word_freq_money,
    spam.word_freq_original,
    spam.word_freq_project,
    spam.char_freq_exc,
    spam.char_freq_dollar,
    spam.capital_run_length_average,
    spam.capital_run_length_longest,
    spam.capital_run_length_total
]



print('All Input Hypothesis [5, 2]')
print("-"*80, flush=True)
do_experiment(spam_df, all_hyp, [spam.is_spam], [5,2], learn_rate=[0.1, 0.01, 0.001], out_name = "fig_ai_[5 2].png")


print('Suspicious Input Hypothesis [5, 2]')
print("-"*80, flush=True)
do_experiment(spam_df, sus_hyp, [spam.is_spam], [5,2], learn_rate=[0.1, 0.01, 0.001], out_name = "fig_si_[5 2].png")
