import json
import math


def main():
    with open('score_yt.json') as r:
        score_dict = json.load(r)

    with open('params.json') as r:
        params = {o['iz_name']: o for o in json.load(r)}

    for k, score_list in score_dict.items():
        param = params[k]

        def score_predictor(obj):
            return int(min(param['max_score'], math.floor(math.exp(param['pod_factor'] * obj['pods'] / param['max_pods'] - param['time_factor'] * obj['time'] / obj.get('tlim', param['max_time']) + param['scale_factor']))))

        def fm_predictor(obj):
            return int(math.floor((1 + math.exp(param['scale_factor'] - 1) * param['pod_factor'] * obj['pods']) / param['max_pods']))

        for i in range(len(score_list)):
            if 'og_score' in score_list[i]:
                score_list[i]['score'] = score_predictor(score_list[i])

            if 'og_fm' in score_list[i]:
                score_list[i]['fm'] = fm_predictor(score_list[i])

            score_list[i] = dict(sorted(score_list[i].items()))

    with open('score_yt.json', 'w') as w:
        json.dump(score_dict, w, indent=4)


if __name__ == '__main__':
    main()
