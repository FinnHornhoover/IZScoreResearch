import sys
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import MDS, Isomap, TSNE
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.svm import LinearSVR
import matplotlib.colors as mcolors
import matplotlib.widgets as mwidgets


DATA_PATH = Path('data')
SPLITS_PATH = DATA_PATH / 'splits'


def show_iz_points():
    with open(DATA_PATH / 'merged_scores.json') as r:
        score_dict = json.load(r)

    for k, score_info in score_dict.items():
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        score_list = score_info['scores']
        pods = np.array([o['pods'] for o in score_list]).astype(float)
        times = np.array([o['time'] for o in score_list]).astype(float)
        scores = np.array([o['score'] for o in score_list]).astype(float)
        ax.scatter(pods, times, scores, c='red')

        ax.set_title(f"{k}\nMS: {score_info['max_score']}, MP: {score_info['max_pods']}, MT: {score_info['max_time']}")

        plt.show()


def show_iz_old_new():
    for k in ['KND Training Area (F)', "Megas' Last Stand", 'Loch Mess', 'Sand Castle', 'Dinosaur Graveyard']:
        k_format = k.replace(' ', '_').replace('(', '').replace(')', '').lower()

        with open(SPLITS_PATH / f'old_{k_format}.json') as r:
            old_values = json.load(r)

        with open(SPLITS_PATH / f'new_{k_format}.json') as r:
            new_values = json.load(r)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for lst in [new_values, old_values]:
            ax.scatter([o['pods'] for o in lst], [o['time'] for o in lst], [o['score'] for o in lst])

        ax.set_title(k)

        plt.show()


def test_pca():
    with open(DATA_PATH / 'merged_scores.json') as r:
        score_dict = json.load(r)

    for k, score_info in score_dict.items():
        objs = score_info['scores']
        max_score = score_info['max_score']
        max_pods = score_info['max_pods']
        max_time = score_info['max_time']

        pca = TSNE(n_components=2)
        x = np.array([[o['pods'] / max_pods for o in objs], [o['time'] / max_time for o in objs]]).T.astype(float)
        y = np.array([o['score'] for o in objs]).astype(float)

        xt = pca.fit_transform(x)

        fig = plt.figure()
        ax = fig.add_subplot()

        ax.scatter(xt[:, 0],y)
        ax.set_title(k)

        plt.show()


def test_angle():
    with open(DATA_PATH / 'merged_scores.json') as r:
        score_dict = json.load(r)

    for k, score_info in score_dict.items():
        objs = score_info['scores']

        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        ax1 = fig.add_subplot(122)

        x = np.array([[o['pods'] for o in objs], [o['time'] for o in objs]]).T.astype(float)
        y = np.array([o['score'] for o in objs]).astype(float)

        ax.scatter(x[:, 0], x[:, 1], y)
        ax.set_title(k)

        def move_view(event):
            xm = np.mean(ax.get_xlim3d())
            ym = np.mean(ax.get_ylim3d())

            azimuth = ax.azim  # Example azimuth value
            line_direction = np.array([[np.cos(np.radians(azimuth)), np.sin(np.radians(azimuth))], [-np.sin(np.radians(azimuth)), np.cos(np.radians(azimuth))]])
            xn = ((x - np.array([xm, ym])) @ line_direction).sum(axis=-1)

            ax1.scatter(xn, y)
            plt.pause(0.05)

            fig.canvas.draw_idle()

        fig.canvas.mpl_connect("button_release_event", move_view)

        plt.show()


def test_points(visualize=False):
    with open(DATA_PATH / 'merged_scores.json') as r:
        score_dict = json.load(r)

    outliers = {}

    banned_points = {}

    for k, score_info in score_dict.items():
        objs = score_info['scores']
        max_score = score_info['max_score']
        max_pods = score_info['max_pods']
        max_time = score_info['max_time']

        def transform(x, y):
            return np.stack((
                x,
                y,
                np.ones_like(x) * (np.log(max_score / 2)),
            ), axis=0)

        p = np.array([o['pods'] / max_pods for o in objs]).astype(float)
        t = np.array([o['time'] / max_time for o in objs]).astype(float)
        y_normal = np.array([o['score'] for o in objs]).astype(float)

        x = transform(p, t).T
        y = np.log(y_normal)

        mask = ~np.isclose(y_normal, max_score)

        if k in banned_points:
            ban_mask = np.ones_like(mask)
            ban_mask[banned_points[k]] = 0
            mask = mask & ban_mask

        clf = LinearRegression(fit_intercept=False)
        clf.fit(x[mask, :], y[mask])

        y_pred = clf.predict(x)
        y_pred_normal = np.exp(y_pred)

        signed_diff = y_normal - y_pred_normal
        diff = np.abs(signed_diff)
        diff_mask = (diff < 0.5) | (np.isclose(y_normal, max_score) & (y_pred_normal > max_score))
        diff[diff_mask] = 0.0

        print(k, 'Average L1 Eps Insensitive', diff.mean(), 'Outliers', np.sum(diff > 0.0), '/', diff.shape[0], '(', np.round(100 * np.mean(diff > 0.0), 3), '%)')
        print(clf.coef_)
        print(signed_diff[~diff_mask])
        print(np.nonzero( (diff > 0.5) & (~np.isclose(y_normal, max_score) | (y_pred_normal <= max_score)) )[0])
        print()
        outliers[k] = [{**o, 'pred_loss': d} for o, d, leave in zip(objs, signed_diff, diff_mask) if not leave]

        if not visualize:
            continue

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        xx, yy = np.meshgrid(np.linspace(0.0, 1.0, 101), np.linspace(0.0, 1.0, 101))
        x_all = transform(xx, yy).reshape(x.shape[1], -1).T

        color_norm = mcolors.Normalize(vmin=-diff.max(), vmax=diff.max())
        signed_diff[diff_mask] = 0.0
        sc = ax.scatter(x[:, 0], x[:, 1], y_normal, s=3, c=signed_diff, norm=color_norm, cmap='RdBu', depthshade=False)
        ax.plot_surface(xx, yy, np.exp(clf.predict(x_all).reshape(*xx.shape)), color='white', alpha=.4)
        fig.colorbar(sc, ax=ax)

        ax.set_title(k)

        plt.show()

    with open(DATA_PATH / 'score_pred_outliers.json', 'w') as w:
        json.dump(outliers, w, indent=4)


def test_line_search():
    with open(DATA_PATH / 'merged_scores.json') as r:
        score_dict = json.load(r)

    def predictor(max_score, pods, time, multp, factor):
        return np.minimum(np.floor(max_score * np.exp(factor[:, :, np.newaxis] * pods[np.newaxis, :] - time - multp[:, :, np.newaxis])), max_score)

    for k, score_info in score_dict.items():
        objs = score_info['scores']
        max_score = score_info['max_score']
        max_pods = score_info['max_pods']
        max_time = score_info['max_time']

        p = np.array([o['pods'] / max_pods for o in objs]).astype(float)
        t = np.array([o['time'] / max_time for o in objs]).astype(float)
        y_normal = np.array([o['score'] for o in objs]).astype(float)

        steps = 501
        xmin, xmax = 0.0, 1.0
        ymin, ymax = 0.0, 5.0

        multp = np.linspace(xmin, xmax, num=steps)
        factor = np.linspace(ymin, ymax, num=steps)

        mm, ff = np.meshgrid(multp, factor)

        preds = predictor(max_score, p, t, mm, ff)
        errors = np.abs(preds - y_normal)
        loss = np.mean(errors, axis=-1)
        err_mean = np.mean(errors > 0, axis=-1)

        fig, ax = plt.subplots()

        ax2 = fig.add_subplot(111, zorder=2)
        ax2.set_navigate(False)

        colormesh = [ax2.pcolormesh(mm, ff, loss, shading='auto')]
        ax2.axis('off')

        ax2.set_title(k)
        ax2.set_xlabel('multp')
        ax2.set_ylabel('factor')

        ax.set_xlim(ax2.get_xlim())
        ax.set_ylim(ax2.get_ylim())

        def on_lims_changed(event):
            ax2.set_xlim(ax.get_xlim())
            ax2.set_ylim(ax.get_ylim())

            multp = np.linspace(*ax.get_xlim(), num=steps)
            factor = np.linspace(*ax.get_ylim(), num=steps)

            mm, ff = np.meshgrid(multp, factor)

            preds = predictor(max_score, p, t, mm, ff)
            errors = np.abs(preds - y_normal)
            loss = np.mean(errors, axis=-1)
            err_mean = np.mean(errors > 0, axis=-1)

            colormesh[0].remove()
            colormesh[0] = ax2.pcolormesh(mm, ff, loss, shading='auto')
            fig.canvas.draw_idle()

        ax.callbacks.connect('xlim_changed', on_lims_changed)
        ax.callbacks.connect('ylim_changed', on_lims_changed)

        plt.show()


if __name__ == '__main__':
    funcs = {f.__name__: f for f in [
        show_iz_points,
        show_iz_old_new,
        test_pca,
        test_angle,
        test_points,
        test_line_search,
    ]}

    if len(sys.argv) > 1:
        kwargs = {n: True for n in sys.argv[2:]} if len(sys.argv) > 2 else {}
        funcs[sys.argv[1]](**kwargs)
