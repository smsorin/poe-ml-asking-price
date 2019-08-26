import requests
import json

def get(change_id=None):
    url = 'http://api.pathofexile.com/public-stash-tabs'
    if change_id:
        url += '/?id=' + change_id
    page = requests.get(url)
    return json.loads(page.content)


