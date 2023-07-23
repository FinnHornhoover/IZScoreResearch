import json

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sconst
import scipy.special as scp

from ff_metrics import mean_percentage_error


def prot1(max_score, max_pods, max_time, pods_collected):
    return pods_collected * max_score * 0.19261251 / max_pods


def prot2(max_score, max_pods, max_time, pods_collected):
    return (np.minimum(pods_collected * 1.0239929, max_pods) * (((((max_score + ((1.0457652 ** pods_collected) - max_time)) - max_time) / (max_pods + -1.291869)) * 0.19140056) + 1.6877022))


def prot3(max_score, max_pods, max_time, pods_collected):
    return (((max_score * pods_collected) / np.minimum(np.exp(5.0680304), max_pods)) / 5.1895375)


def prot4(max_score, max_pods, max_time, pods_collected):
    return 0.1876364 * (max_score + 2 ** (0.06440807 * max_pods)) * pods_collected / max_pods


def prot5(max_score, max_pods, max_time, pods_collected):
    return 0.19696088 * (max_score - max_time + 2 ** (0.064131744 * max_pods)) * pods_collected / max_pods


def prot6(max_score, max_pods, max_time, pods_collected):
    return (max_score / 5.2791777) * pods_collected / np.minimum(np.maximum(max_pods, 16.617645), 155.22107)


def prot7(max_score, max_pods, max_time, pods_collected):
    return ((np.ceil(np.ceil(((max_score - max_time) / 4.5803595) / np.minimum(max_pods, 157.18338)) / 1.184314) - 0.7384641 ** 2) * (1.0066032 ** 9)) * pods_collected


def prot8(max_score, max_pods, max_time, pods_collected):
    return ((np.ceil((((max_score - max_time) / np.ceil(np.minimum(max_pods, 160.37965))) - 2.3073153) / 10.80305) / 0.47741216) + np.exp(-0.057032812 * max_pods)) * pods_collected


def prot9(max_score, max_pods, max_time, pods_collected):
    return (np.maximum(np.minimum(0.1896468 / max_pods, 0.0114018265), (0.018011156 ** 3) * max_pods) * max_score) * pods_collected


def prot10(max_score, max_pods, max_time, pods_collected):
    return ((np.maximum(np.minimum(np.tanh(np.tanh(0.18727496)) / ((max_pods - (max((scp.erf(scp.erf((0.026683195 ** 3)) ** 3) * max_score) + 0.26115447, 0.6002886))) + -0.23691316), 0.010817147), (0.017940171 ** 3) * max_pods) * max_score) + 0.5715052) * pods_collected


def main():
    with open('data/merged_scores.json') as r:
        score_dict = json.load(r)

    for k, score_info in score_dict.items():
        objs = [v for v in score_info['scores'] if 'fm' in v]

        if len(objs) == 0:
            continue

        fig = plt.figure()
        ax = fig.add_subplot()

        max_score = score_info['max_score']
        max_pods = score_info['max_pods']
        max_time = score_info['max_time']
        x = np.arange(0, max_pods + 1)

        pods = np.array([o['pods'] for o in objs]).astype(float)
        fms = np.array([o['fm'] for o in objs]).astype(float)
        func = prot10

        ax.scatter(pods, fms, c='red')
        ax.plot(x, func(max_score, max_pods, max_time, x), alpha=0.5)

        ax.set_title(k)

        plt.show()

        pred_tup = list(zip(func(max_score, max_pods, max_time, pods), fms))
        print(k)
        print('(Pred, True):', pred_tup)
        print('MAPE:', np.mean([mean_percentage_error(a, b, w=1, safe=True) for a, b in pred_tup]))
        print('MSPE:', np.mean([mean_percentage_error(a, b, w=1, safe=True, ord=2) for a, b in pred_tup]))
        print()


if __name__ == '__main__':
    main()
