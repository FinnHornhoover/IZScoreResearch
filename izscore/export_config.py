import os
import json
from pathlib import Path
from typing import Any, Dict
from argparse import ArgumentParser, Namespace

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox


RANK_COEF = [0.8, 0.7, 0.5, 0.3, 0.29]


def predict_score(
    max_score: int,
    max_pods: int,
    max_time: int,
    pods: np.ndarray,
    time: np.ndarray,
    scale_factor: float,
    pod_factor: float,
    time_factor: float,
) -> np.ndarray:
    return np.minimum(max_score, np.floor(np.exp(
        pod_factor * pods / max_pods - time_factor * time / max_time + scale_factor)))


def predict_fm(
    max_pods: int,
    pods: np.ndarray,
    scale_factor: float,
    pod_factor: float,
) -> np.ndarray:
    return np.floor((1. + np.exp(scale_factor - 1.) * pod_factor * pods) / max_pods)


def get_values_for_iz(params: Dict[str, Any], iz_name: str) -> Dict[str, Any]:
    max_score = params['max_score']
    max_pods = params['max_pods']
    max_time = params['max_time']
    scale_factor = params['scale_factor']
    pod_factor = params['pod_factor']
    time_factor = params['time_factor']

    out_param = {
        '!RankScores': [int(rc * max_score) for rc in RANK_COEF],
        '!TimeLimit': max_time,
        'ScoreCap': max_score,
        'TotalPods': max_pods,
        'ScaleFactor': scale_factor,
        'PodFactor': pod_factor,
        'TimeFactor': time_factor,
        'EPName': iz_name,
    }

    p = np.arange(0, max_pods + 1)
    t = np.arange(0, max_time + 1)
    pp, tt = np.meshgrid(p, t, indexing='ij')

    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(122, projection='3d')
    ax_fm = fig.add_subplot(121)

    fig.suptitle(iz_name + ' [Exit to Save]')

    ax.set_xlabel('Pods')
    ax.set_ylabel('Time')
    ax.set_zlabel('Score')
    ax.view_init(elev=20., azim=135.)

    ax_fm.set_xlabel('Pods')
    ax_fm.set_ylabel('Fusion Matter')

    fig.subplots_adjust(bottom=0.3)
    fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)

    istride = p.shape[0] // 3
    jstride = t.shape[0] // 4
    score_pred = predict_score(max_score, max_pods, max_time, pp, tt, scale_factor, pod_factor, time_factor)
    fm_pred = predict_fm(max_pods, p, scale_factor, pod_factor)
    plots = [
        ax.plot_surface(pp, tt, score_pred, cmap=plt.cm.coolwarm),
        ax_fm.plot(p, fm_pred, color='lightgreen'),
    ]
    score_annots = [[ax.text(pv, tv, score_pred[-i * istride - 1, j * jstride], str(int(score_pred[-i * istride - 1, j * jstride])), size=10, color='black')
                     for j, tv in enumerate(t[::jstride])]
                     for i, pv in enumerate(p[::-istride])]
    fm_annots = [ax_fm.text(pv, fm_pred[-i * istride - 1], str(int(fm_pred[-i * istride - 1])), size=10, color='black')
                 for i, pv in enumerate(p[::-istride])]

    widget_color = 'lightgoldenrodyellow'
    sf_slider_ax  = fig.add_axes([0.2, 0.175, 0.7, 0.03], facecolor=widget_color)
    sf_slider = Slider(sf_slider_ax, 'ScaleFactor', 0.0, 20.0, valinit=scale_factor)

    pf_slider_ax = fig.add_axes([0.2, 0.125, 0.7, 0.03], facecolor=widget_color)
    pf_slider = Slider(pf_slider_ax, 'PodFactor', 0.0, 3.0, valinit=pod_factor)

    tf_slider_ax = fig.add_axes([0.2, 0.075, 0.7, 0.03], facecolor=widget_color)
    tf_slider = Slider(tf_slider_ax, 'TimeFactor', 0.0, 3.0, valinit=time_factor)

    reset_button_ax = fig.add_axes([0.8, 0.025, 0.1, 0.03])
    reset_button = Button(reset_button_ax, 'Reset', color=widget_color, hovercolor='0.975')

    ms_box_ax = fig.add_axes([0.2, 0.025, 0.1, 0.03])
    ms_box = TextBox(ms_box_ax, 'MaxScore', initial=str(max_score), color=widget_color, hovercolor='0.975', label_pad=0.15)
    current_max_score = [max_score]

    def sliders_on_changed(val):
        plots[0].remove()
        score_pred = predict_score(current_max_score[0], max_pods, max_time, pp, tt, sf_slider.val, pf_slider.val, tf_slider.val)
        plots[0] = ax.plot_surface(pp, tt, score_pred, cmap=plt.cm.coolwarm)

        plots[1][0].remove()
        fm_pred = predict_fm(max_pods, p, sf_slider.val, pf_slider.val)
        plots[1] = ax_fm.plot(p, fm_pred, color='lightgreen')
        ax_fm.relim()
        ax_fm.autoscale_view()

        for i, pv in enumerate(p[::-istride]):
            fm_annots[i].remove()
            fm_annots[i] = ax_fm.text(pv, fm_pred[-i * istride - 1], str(int(fm_pred[-i * istride - 1])), size=10, color='black')

            for j, tv in enumerate(t[::jstride]):
                score_annots[i][j].remove()
                score_annots[i][j] = ax.text(pv, tv, score_pred[-i * istride - 1, j * jstride], str(int(score_pred[-i * istride - 1, j * jstride])), size=10, color='black')

        fig.canvas.draw_idle()

    sf_slider.on_changed(sliders_on_changed)
    pf_slider.on_changed(sliders_on_changed)
    tf_slider.on_changed(sliders_on_changed)

    def reset_button_on_clicked(mouse_event):
        ms_box.set_val(str(max_score))
        current_max_score[0] = max_score
        sf_slider.reset()
        pf_slider.reset()
        tf_slider.reset()

    reset_button.on_clicked(reset_button_on_clicked)

    def textbox_on_submit(text):
        try:
            box_val = int(text)
            if box_val < 0:
                raise ValueError('Negative!')
            current_max_score[0] = box_val
        except ValueError:
            ms_box.set_val(str(current_max_score[0]))

        sliders_on_changed(-1)

    ms_box.on_submit(textbox_on_submit)

    plt.show()

    out_param.update({
        'ScoreCap': current_max_score[0],
        'ScaleFactor': sf_slider.val,
        'PodFactor': pf_slider.val,
        'TimeFactor': tf_slider.val,
    })

    return out_param


def make_values(args: Namespace) -> None:
    with open(args.param_path) as r:
        param_dict = {o['iz_name'].replace(' (New)', ''): o
                      for o in json.load(r)
                      if '(Old)' not in o['iz_name']}

    out_dict = {
        'Racing': {str(i): get_values_for_iz(params, iz_name)
                   for i, (iz_name, params) in enumerate(param_dict.items())}
    }

    os.makedirs(args.out_path, exist_ok=True)
    with open(Path(args.out_path) / 'drops.json', 'w') as w:
        json.dump(out_dict, w, indent=4)


def main() -> None:
    parser = ArgumentParser(description='FF Score Function Modifier.')
    parser.add_argument('--param-path', dest='param_path', type=str,
                        default='data/params.json')
    parser.add_argument('--out-path', dest='out_path', type=str,
                        default='data/ogracingpatch')
    parser.add_argument('--izs-to-view', dest='izs_to_view', type=str, nargs='+')
    args = parser.parse_args()

    make_values(args)


if __name__ == '__main__':
    main()
