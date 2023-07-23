import os
from collections import defaultdict
from pprint import pprint
from operator import itemgetter


def main():
    QUERIES = [
        # Keywords go here
        'Pods',
    ]
    URL_FILTER = [
        # remove entries if they contain these words in the url
        'xtreme',
        'fusionfallcentral',
    ]
    # number of results to show
    LIST_COUNT = 100

    scores = defaultdict(int)

    for filename in os.listdir('ffpages'):
        skip = False
        for url_filt in URL_FILTER:
            if url_filt in filename:
                skip = True
                break

        if skip:
            continue

        with open(os.path.join('ffpages', filename), encoding='utf-8', errors='ignore') as r:
            pstr = r.read().lower()

        for query in QUERIES:
            scores[filename] += pstr.count(query.lower())

    pprint(sorted(scores.items(), key=itemgetter(1), reverse=True)[:LIST_COUNT])


if __name__ == '__main__':
    main()
