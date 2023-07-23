import requests
import time


def main():
    wayback_url = 'https://web.archive.org/cdx/search/cdx'
    wayback_html_url = 'https://web.archive.org/web/{timestamp}/{original}'
    params = {
        # replace the url here to the site you want to fetch
        'url': 'fusionfall.com/community/leaderboard-player.php',
        'matchType': 'prefix',
        'output': 'json',
        'filter': 'mimetype:text/html',
        'collapse': 'digest',
        'from': '20081005000000',
        'to': '20091005000000',
    }
    response = requests.get(wayback_url, params=params)

    if response.status_code != 200:
        print('Response returned code', response.status_code)
        return

    rj = response.json()
    print(len(rj) - 1, 'pages found')
    time.sleep(10)
    for i, row in enumerate(rj[1:]):
        d = dict(zip(rj[0], row))

        if d['statuscode'] != "200":
            continue

        while True:
            try:
                inner_resp = requests.get(wayback_html_url.format(**d))
                break
            except:
                # you're overwhelming Wayback Machine servers, be nicer
                print('sleeping 120 secs ...')
                time.sleep(120)

        if inner_resp.status_code != 200:
            continue

        filename = 'ffpages/' + ''.join(x for x in d['original'] if x.isalnum()) + '_' + d['timestamp'] + '.html'

        with open(filename, 'wb') as w:
            w.write(inner_resp.content)

        print(filename, 'done!')

        if (i + 1) % 10 == 0:
            # be nice, don't overwhelm Wayback Machine servers
            print('sleeping 10 secs ...')
            time.sleep(10)


if __name__ == '__main__':
    main()
