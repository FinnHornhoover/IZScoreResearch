import json
from pathlib import Path
from operator import itemgetter

import matplotlib.pyplot as plt


def main():
    with open('raw_leaderboard.json') as r:
        score_dict = json.load(r)

    for k, objs in score_dict.items():
        if k not in ['KND Training Area (F)', "Megas' Last Stand", 'Loch Mess', 'Sand Castle', 'Dinosaur Graveyard']:
            continue

        k_format = k.replace(" ", "_").replace("(", "").replace(")", "").lower()

        if Path(f'splits/old_{k_format}.json').is_file() or Path(f'splits/new_{k_format}.json').is_file():
            continue

        pod_counts_unique = {o['pods'] for o in objs}
        time_unique = {o['time'] for o in objs}

        lists = {
            'pods': {
                'set': pod_counts_unique,
                'old': [],
                'new': [],
            },
            'time': {
                'set': time_unique,
                'old': [],
                'new': [],
            },
        }

        val_key = 'pods'
        other_key = 'time'

        for scope in lists[val_key]['set']:
            scoped_objs = [o for o in objs if scope == o[val_key]]
            other_x = [o[other_key] for o in scoped_objs]
            score_y = [o['score'] for o in scoped_objs]

            if len(set(score_y)) < 2 or (len(set(score_y)) == 2 and len(set(other_x)) == 2):
                lists[val_key]['new'].extend(scoped_objs)
                continue

            _, ax = plt.subplots()
            ax.set_title(f'{k} {val_key.title()} = {scope}')
            ax.scatter(other_x, score_y)
            plt.pause(0.05)

            if not input('Is everything b here? (y/n): ').startswith('n'):
                ax.scatter(other_x, score_y, c='red')
                lists[val_key]['new'].extend(scoped_objs)
            else:
                sorted_scores = sorted(scoped_objs, key=itemgetter('score'))

                for i, o in enumerate(sorted_scores):
                    ax.scatter([o[other_key]], [o['score']], c='yellow')
                    plt.pause(0.05)

                    print('Prev:', [(oo['score'], oo[other_key], oo.get('player'), oo.get('date')) for oo in sorted_scores[max(0, i - 2):i]],
                          '\nCur:', (o['score'], o[other_key], o.get('player'), o.get('date')),
                          '\nNext:', [(oo['score'], oo[other_key], oo.get('player'), oo.get('date')) for oo in sorted_scores[min(i + 1, len(sorted_scores) - 1):(i + 3)]])

                    if input('Please enter t or b: ').startswith('t'):
                        lists[val_key]['old'].append(o)
                        ax.scatter([o[other_key]], [o['score']], c='green')
                    else:
                        lists[val_key]['new'].append(o)
                        ax.scatter([o[other_key]], [o['score']], c='red')

            plt.pause(1)
            plt.close()

        with open(f'splits/old_{k_format}.json', 'w') as w:
            json.dump(lists[val_key]['old'], w, indent=4)

        with open(f'splits/new_{k_format}.json', 'w') as w:
            json.dump(lists[val_key]['new'], w, indent=4)


if __name__ == '__main__':
    main()
