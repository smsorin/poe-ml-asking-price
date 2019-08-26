import requests
import json

class ID(object):
    def __init__(self, change_id=''):
        if not change_id:
            self.num = [0]*5
        else:
            self.num = [int(x) for x in change_id.split('-')]

    def __repr__(self):
        return '-'.join(['%d' % x for x in self.num])
    def __add__(self, other):
        ret = ID()
        ret.num = [x+y for x,y in zip(self.num, other.num)]
        return ret
    def __truediv__(self, divisor):
        ret = ID()
        ret.num = [x / divisor for x in self.num]
        return ret
    def __lt__(self, other):
        return ('%s' % self) < ('%s' % other)

def Get(change_id=None):
    url = 'http://api.pathofexile.com/public-stash-tabs'
    if change_id:
        url += '?id=%s' % change_id
    page = requests.get(url)
    return json.loads(page.content)


