import sys
import json
import glob
from pathlib import Path
from itertools import groupby
from operator import itemgetter
from collections import defaultdict
from bs4 import BeautifulSoup


TDATA_PATH = Path.home() / 'source' / 'repos' / 'OpenFusion' / 'tdata'
SPLITS_PATH = Path('splits')
FFPAGES_PATH = Path('ffpages')
ep_pods = {
    'KND Training Area (F) (New)': 14,    # 46 sec approx
    'KND Training Area (F) (Old)': 14,    # 46 sec approx
    'Pokey Oaks Junior High (F)': 38,     # 3 min approx
    "Mandark's House (F)": 26,            # 2 min 36 approx
    'Delightful Developments (F)': 62,    # 3 min 47 approx
    "Megas' Last Stand (F)": 83,          # 6 min 20 approx
    'Pokey Oaks Junior High': 103,
    "Mandark's House": 88,
    'Sweet Revenge': 53,
    'Delightful Developments': 64,
    "Megas' Last Stand (New)": 137,       # 6 min 45 approx
    "Megas' Last Stand (Old)": 137,       # 6 min 45 approx
    'The Boneyard': 128,                  # 8 min 30 approx
    'Reactor Works': 132,
    'Charles Darwin Middle School': 76,
    'Dizzy World': 122,
    'Sunny Bridges Auditorium': 97,
    'Jungle Training Area': 84,           # 4 min approx
    'Loch Mess (New)': 53,                # 3 min approx
    'Loch Mess (Old)': 53,                # 3 min approx
    'The Fissure': 115,
    'Cutts and Bruises Skate Park': 49,   # 2 min 30 approx
    'Sand Castle (New)': 156,             # 9 min approx
    'Sand Castle (Old)': 156,             # 9 min approx
    'Crystalline Caverns': 50,
    'Construction Site': 81,
    'Hani-Baba Temple': 69,
    'Tyrannical Gardens': 123,
    'Skypad Space Port': 220,
    'Nowhere Triangle': 97,               # 5 min approx
    'The Canopy': 75,
    'Monkey Summit': 59,
    'Dinosaur Graveyard (New)': 87,
    'Dinosaur Graveyard (Old)': 87,
    'Inferno Fields': 89,
    'Dark Tree Clearing': 168,
    'Green Gullet': 74,                   # one more is inaccessible
}
ep_times = {
    'KND Training Area (F) (New)': 150,
    'KND Training Area (F) (Old)': 165,
    'Pokey Oaks Junior High (F)': 430,    # 548 inexact youtu.be/hNb8R69NaSw
    "Mandark's House (F)": 192,
    'Delightful Developments (F)': 590,
    "Megas' Last Stand (F)": 982,
    'Pokey Oaks Junior High': 647,
    "Mandark's House": 507,
    'Sweet Revenge': 300,
    'Delightful Developments': 425,
    "Megas' Last Stand (New)": 1030,
    "Megas' Last Stand (Old)": 1030,
    'The Boneyard': 1277,
    'Reactor Works': 1537,
    'Charles Darwin Middle School': 682,
    'Dizzy World': 705,
    'Sunny Bridges Auditorium': 685,
    'Jungle Training Area': 575,
    'Loch Mess (New)': 367,               # 461 pre-update
    'Loch Mess (Old)': 367,               # 461 pre-update
    'The Fissure': 677,
    'Cutts and Bruises Skate Park': 347,
    'Sand Castle (New)': 1257,
    'Sand Castle (Old)': 1257,
    'Crystalline Caverns': 702,
    'Construction Site': 1082,
    'Hani-Baba Temple': 825,
    'Tyrannical Gardens': 1590,
    'Skypad Space Port': 1750,
    'Nowhere Triangle': 640,
    'The Canopy': 1167,
    'Monkey Summit': 925,
    'Dinosaur Graveyard (New)': 587,
    'Dinosaur Graveyard (Old)': 587,
    'Inferno Fields': 1370,
    'Dark Tree Clearing': 1372,
    'Green Gullet': 652,
}
epids = {
    'KND Training Area (F) (New)': 1,
    'KND Training Area (F) (Old)': 1,
    'Pokey Oaks Junior High (F)': 2,
    "Mandark's House (F)": 3,
    'Delightful Developments (F)': 4,
    "Megas' Last Stand (F)": 5,
    'Pokey Oaks Junior High': 7,
    "Mandark's House": 8,
    'Sweet Revenge': 9,
    'Delightful Developments': 10,
    "Megas' Last Stand (New)": 11,
    "Megas' Last Stand (Old)": 11,
    'The Boneyard': 12,
    'Reactor Works': 13,
    'Charles Darwin Middle School': 14,
    'Dizzy World': 15,
    'Sunny Bridges Auditorium': 16,
    'Jungle Training Area': 17,
    'Loch Mess (New)': 18,
    'Loch Mess (Old)': 18,
    'The Fissure': 19,
    'Cutts and Bruises Skate Park': 20,
    'Sand Castle (New)': 21,
    'Sand Castle (Old)': 21,
    'Crystalline Caverns': 22,
    'Construction Site': 23,
    'Hani-Baba Temple': 24,
    'Tyrannical Gardens': 25,
    'Skypad Space Port': 26,
    'Nowhere Triangle': 27,
    'The Canopy': 28,
    'Monkey Summit': 29,
    'Dinosaur Graveyard (New)': 30,
    'Dinosaur Graveyard (Old)': 30,
    'Inferno Fields': 31,
    'Dark Tree Clearing': 32,
    'Green Gullet': 33,
}
ep_levels = {
    'KND Training Area (F) (New)': 1,
    'KND Training Area (F) (Old)': 1,
    'Pokey Oaks Junior High (F)': 2,
    "Mandark's House (F)": 2,
    'Delightful Developments (F)': 3,
    "Megas' Last Stand (F)": 4,
    'Pokey Oaks Junior High': 5,
    "Mandark's House": 7,
    'Sweet Revenge': 7,
    'Delightful Developments': 8,
    "Megas' Last Stand (New)": 9,
    "Megas' Last Stand (Old)": 9,
    'The Boneyard': 9,
    'Reactor Works': 11,
    'Charles Darwin Middle School': 12,
    'Dizzy World': 13,
    'Sunny Bridges Auditorium': 14,
    'Jungle Training Area': 14,
    'Loch Mess (New)': 15,
    'Loch Mess (Old)': 15,
    'The Fissure': 16,
    'Cutts and Bruises Skate Park': 17,
    'Sand Castle (New)': 18,
    'Sand Castle (Old)': 18,
    'Crystalline Caverns': 17,
    'Construction Site': 19,
    'Hani-Baba Temple': 20,
    'Tyrannical Gardens': 21,
    'Skypad Space Port': 24,
    'Nowhere Triangle': 24,
    'The Canopy': 25,
    'Monkey Summit': 27,
    'Dinosaur Graveyard (New)': 28,
    'Dinosaur Graveyard (Old)': 28,
    'Inferno Fields': 30,
    'Dark Tree Clearing': 32,
    'Green Gullet': 35,
}


def scrape_values():
    seen = set()
    raw_values = defaultdict(list)
    fake_values = defaultdict(list)
    record_fields = ['player', 'date', 'score', 'pods', 'time']

    for file in glob.iglob(str(FFPAGES_PATH / 'httpfusionfallcomcommunityleaderboardplayerphp*.html')):
        try:
            with open(file, encoding='utf-8', errors='ignore') as r:
                bs = BeautifulSoup(r, features='lxml')

            date = file.split('_')[1].split('.')[0]
            pname = str(bs.findChild('div', attrs={'id': 'player_info'}).findChild('p', attrs={'class': 'name'}).contents[0])

            for obj in bs.find_all('div', attrs={'class': 'race_score'}):
                k = str(obj.findChild('li', attrs={'class': 'name'}).contents[0])
                name_suffix = ' (F)' if str(obj.findChild('li', attrs={'class': 'region'}).contents[0]) == 'Future' else ''
                full_name = k + name_suffix

                topscore = int(obj.findChild('li', attrs={'class': 'topscore'}).contents[0])

                time_str = str(obj.findChild('li', attrs={'class': 'time'}).contents[0])
                time = sum([multp * int(piece) for multp, piece in zip([60, 1], time_str.split(':', 1))])

                pods = int(obj.findChild('li', attrs={'class': 'pods'}).contents[0])

                record = (pname, date, topscore, pods, time, full_name)

                if topscore == 0 or record in seen:
                    continue

                seen.add(record)
                dest = fake_values if topscore == time and topscore == pods else raw_values
                dest[full_name].append(dict(zip(record_fields, record)))

            print('Processed', file)
        except Exception as e:
            print('Error with', file, e)

    with open('raw_leaderboard.json', 'w') as w:
        json.dump(raw_values, w, indent=4)

    with open('fake_leaderboard.json', 'w') as w:
        json.dump(fake_values, w, indent=4)

    return raw_values, fake_values


def split_outliers(raw_values):
    # after fetching, use manually separated old and new values in certain IZs
    values = raw_values.copy()

    for k in ['KND Training Area (F)', "Megas' Last Stand", 'Loch Mess', 'Sand Castle', 'Dinosaur Graveyard']:
        k_format = k.replace(' ', '_').replace('(', '').replace(')', '').lower()

        with open(SPLITS_PATH / f'old_{k_format}.json') as r:
            values[f'{k} (Old)'] = json.load(r)

        with open(SPLITS_PATH / f'new_{k_format}.json') as r:
            values[f'{k} (New)'] = json.load(r)

        del values[k]

    with open('leaderboard.json', 'w') as w:
        json.dump(values, w, indent=4)

    return values


def get_limits():
    with open(TDATA_PATH / 'xdt1013.json') as r:
        xdt = json.load(r)

    max_scores_idx = {o['m_iIsEP']: o['m_ScoreMax'] for o in xdt['m_pInstanceTable']['m_pInstanceData'] if o['m_iIsEP'] > 0}
    max_scores = {k: max_scores_idx[i] for k, i in epids.items()}

    # the old max values
    max_scores['KND Training Area (F) (Old)'] = 649
    max_scores["Megas' Last Stand (Old)"] = 15158
    max_scores['Loch Mess (Old)'] = 7647
    max_scores['Sand Castle (Old)'] = 26775
    max_scores['Dinosaur Graveyard (Old)'] = 19180

    limit_dict = {k: {
        'epid': epids[k],
        'ep_level': ep_levels[k],
        'max_score': max_scores[k],
        'max_pods': ep_pods[k],
        'max_time': ep_times[k],
    } for k in ep_times}

    with open('limits.json', 'w') as w:
        json.dump(limit_dict, w, indent=4)

    return limit_dict


def merge_scores_and_limits(values, limit_dict):
    # merge yt values
    with open('score_yt.json') as r:
        yt_values = json.load(r)

    merged_scores = {k: {
        **limit_dict[k],
        'scores': values[k] + [v for v in yt_values[k] if 'og_score' not in v],
    } for k in ep_times}

    with open('merged_scores.json', 'w') as w:
        json.dump(merged_scores, w, indent=4)

    return merged_scores


def run_checks(merged_scores):
    for k, score_info in merged_scores.items():
        # over max time check
        over_max_times = [o['time'] for o in score_info['scores'] if o['time'] >= ep_times[k]]
        if over_max_times:
            print(k, 'contains possibly erroneous', over_max_times, 'values!')

        # monotonicty check
        non_monotonic_score = []

        for _, objs in groupby(sorted(score_info['scores'], key=itemgetter('pods', 'time')), key=itemgetter('pods')):
            max_seen_score = 1e8
            last_obj = None

            for o in objs:
                if max_seen_score < o['score']:
                    non_monotonic_score.append((o, 'time', last_obj))
                max_seen_score = o['score']
                last_obj = o


        for _, objs in groupby(sorted(score_info['scores'], key=itemgetter('time', 'pods')), key=itemgetter('time')):
            min_seen_score = 0
            last_obj = None

            for o in objs:
                if min_seen_score > o['score']:
                    non_monotonic_score.append((o, 'pods', last_obj))
                min_seen_score = o['score']
                last_obj = o

        for err_obj, reason, other_obj in non_monotonic_score:
            print(f'{k} contains possibly erroneous object\n{err_obj}\ndue to ({reason}) because\n{other_obj}\nexits!\n')


def main(skip_scrape=False):
    if skip_scrape:
        with open('raw_leaderboard.json') as r:
            raw_values = json.load(r)
    else:
        raw_values, _ = scrape_values()

    values = split_outliers(raw_values)
    limit_dict = get_limits()
    merged_scores = merge_scores_and_limits(values, limit_dict)
    run_checks(merged_scores)


if __name__ == '__main__':
    main(skip_scrape=(len(sys.argv) > 1 and sys.argv[1] == 'skipscrape'))
