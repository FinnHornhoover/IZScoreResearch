import json

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants as sconst

from ff_metrics import mean_percentage_error


def prot1(max_score, max_pods, max_time, pods_collected, time):
    return np.minimum((max_score ** 1.6311166) / time, max_score + (pods_collected - max_pods) / 0.015940396)


def prot2(max_score, max_pods, max_time, pods_collected, time):
    return np.minimum((max_score ** 1.6540185) * (pods_collected + 2.825367) / ((max_pods * time) + max_score), max_score)


def prot3(max_score, max_pods, max_time, pods_collected, time):
    return np.minimum((max_score ** sconst.golden) * (pods_collected + sconst.e) / ((max_pods * time) + max_score), max_score)


def prot4(max_score, max_pods, max_time, pods_collected, time):
    return np.minimum(max_score, (max_score ** 1.6320213) / time) * ((pods_collected / max_pods) ** 0.5296513)


def prot5(max_score, max_pods, max_time, pods_collected, time):
    return max_score - ((max_pods - pods_collected) + 2451.869 / max_score) * (time - max_pods)


def prot6(max_score, max_pods, max_time, pods_collected, time):
    return np.minimum(max_score * np.sqrt(pods_collected) / np.sqrt(max_pods - pods_collected * 0.23019499) - time ** 1.3428546, max_score)


def prot7(max_score, max_pods, max_time, pods_collected, time):
    return np.minimum((max_score ** 1.6541654) * pods_collected / (max_score + pods_collected * time), max_score)


def prot8(max_score, max_pods, max_time, pods_collected, time):
    return np.minimum((max_score * np.sqrt(pods_collected) / np.sqrt((time + max_pods) * np.sqrt(max_pods))) ** 1.2462883, max_score)


def prot9(max_score, max_pods, max_time, pods_collected, time):
    return np.minimum((max_time * pods_collected / (max_pods * (time + 29.458052)) + 0.6773201) * 0.338596 * max_score, max_score)


def prot10(max_score, max_pods, max_time, pods_collected, time):
    return np.minimum((pods_collected - 8.911205) * (max_score - time ** 1.4300926) / max_pods + max_time ** 1.2894257 - max_time, max_score)


def prot11(max_score, max_pods, max_time, pods_collected, time):
    return np.maximum(pods_collected / max_pods, 0.3529223) * (max_score - (np.maximum(time, np.exp(4.4260664)) ** 2.3920426) / max_time) + max_time


def prot12(max_score, max_pods, max_time, pods_collected, time):
    return max_score / (np.log(max_pods) - np.log(np.maximum(1.0, pods_collected)) + (time / max_time) + 0.66664994)


def prot13(max_score, max_pods, max_time, pods_collected, time):
    return np.minimum(max_score * 0.44801974 * (max_time + 2 * pods_collected - max_pods) / np.minimum(np.floor((time / (pods_collected + 0.12837368) + 0.23308727) * max_pods + 38.29458), max_time) + time, max_score)


def prot14(max_score, max_pods, max_time, pods_collected, time):
    return ((((max_score ** ((pods_collected / (max_pods + 7.8712173)) ** 0.17007987)) - (((time ** 2.6447675) / (max_time / 0.24446912)) - (max_score * 0.38823175))) + max_time) * 0.7767193)


def prot15(max_score, max_pods, max_time, pods_collected, time):
    return ((max_score * np.exp((pods_collected + -2.0973341) / (max_pods / 1.1246198))) / np.exp(np.exp((time / max_time) ** 1.7093706) ** 0.66899526))


def prot16(max_score, max_pods, max_time, pods_collected, time):
    return (max_score - (time ** 1.2164748)) * np.sqrt(pods_collected / max_pods)


def prot17(max_score, max_pods, max_time, pods_collected, time):
    return np.minimum(((((np.maximum(0.0, pods_collected + -2.7010932) / max_pods) ** 1.5081385) + 0.48413983) / np.exp(time / max_time)) * max_score, max_score)


def prot18(max_score, max_pods, max_time, pods_collected, time):
    return np.minimum(max_score * ((pods_collected / max_pods) ** 1.5348951 + 0.44351) / np.exp(time / max_time), max_score)


def prot19(max_score, max_pods, max_time, pods_collected, time):
    return np.minimum((np.exp(((-2.0789618 + pods_collected) / max_pods) ** 1.3020364) * max_score) / np.exp((time / max_time) + 0.6050899), max_score)


def prot20(max_score, max_pods, max_time, pods_collected, time):
    return np.minimum((np.exp(((((-0.7866236 / max_pods) / 0.009003487) + pods_collected) - (-0.09650744 * pods_collected)) / (max_pods - np.log(max_time))) * max_score) / np.exp((time / max_time) + 0.8132038), max_score)


def prot21(max_score, max_pods, max_time, pods_collected, time):
    return np.minimum(max_score, np.exp(pods_collected / max_pods) * 0.4575826 * max_time * max_score / (time + max_time) - time)


def prot22(max_score, max_pods, max_time, pods_collected, time):
    return (((max_score / (2.625248 - ((pods_collected * 1.4832045) / max_pods))) - (time ** 1.2477597)) + max_time)


def prot23(max_score, max_pods, max_time, pods_collected, time):
    return max_score / np.exp(0.822641 - pods_collected / (max_pods ** 0.9637691) + time / max_time)


def prot24(max_score, max_pods, max_time, pods_collected, time):
    return np.minimum(max_score / np.exp(0.82721686 - pods_collected / (max_pods / 1.2334778 + 2.4358647) + time / max_time), max_score)


def main():
    with open('data/merged_scores.json') as r:
        score_dict = json.load(r)

    rank_fracs = [0.8, 0.7, 0.5, 0.3, 0.29]
    rank_colors = ['yellow', 'silver', 'orange', 'darkgray', 'red']

    for k, score_info in score_dict.items():
        objs = score_info['scores']
        max_score = score_info['max_score']
        max_pods = score_info['max_pods']
        max_time = score_info['max_time']

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        x = np.arange(0, max_pods + 1)
        y = np.arange(0, max_time + 1)
        xx, yy = np.meshgrid(x, y)

        pods = np.array([o['pods'] for o in objs]).astype(float)
        times = np.array([o['time'] for o in objs]).astype(float)
        scores = np.array([o['score'] for o in objs]).astype(float)
        func = prot24

        ax.scatter(pods, times, scores, c='red')
        ax.plot_surface(xx, yy, func(max_score, max_pods, max_time, xx, yy), alpha=0.5)

        ax.set_title(k)

        for rank_frac, rank_color in zip(rank_fracs, rank_colors):
            lim = max_score * rank_frac
            ax.plot_surface(xx, yy, np.ones_like(xx) * lim, c=rank_color, alpha=0.5)

        plt.show()

        pred_tup = list(zip(func(max_score, max_pods, max_time, pods, times), scores))
        print(k)
        print('(Pred, True):', pred_tup)
        print('MAPE:', np.mean([mean_percentage_error(a, b, w=1) for a, b in pred_tup]))
        print('MSPE:', np.mean([mean_percentage_error(a, b, w=1, ord=2) for a, b in pred_tup]))
        print()


if __name__ == '__main__':
    main()
