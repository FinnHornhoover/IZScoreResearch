import json
from operator import itemgetter
from collections import defaultdict


def main():
    with open('nopod_scores.txt') as r:
        scores_text = r.read()

    score_dict = defaultdict(list)
    for i, split in enumerate(scores_text.split('\n\n')):
        str_set = set()
        lines = split.split('\n')
        key = None if i == 0 else lines[0]

        for j, line in enumerate(lines):
            print(f'Group {i:2d} Line {j:4d}', end='\r')

            if line in str_set or line == key:
                continue

            str_set.add(line)
            pieces = line.split('\t')
            score_idx = 2
            if i == 0:
                key = pieces[2]
                score_idx = 3

            if key == 'Monkey Skypad':
                key = 'Monkey Summit'

            score_dict[key].append({'order': int(pieces[1]), 'score': int(pieces[score_idx])})

        score_dict[key] = sorted(score_dict[key], key=itemgetter('score'), reverse=True)

    with open('nopod_score_sanitized.json', 'w') as w:
        json.dump(score_dict, w, indent=4)


if __name__ == '__main__':
    main()
