from os import listdir
import pickle
import re
import json
import sys
from collections import defaultdict

def Stashes(files):
    for file_name in files:
        print('Start reading %s of %d' %(file_name, len(files)))
        sys.stdout.flush()
        with open(file_name, 'rb') as f:
            try:
                while True:
                    yield pickle.load(f)
            except EOFError: pass
    

def CountItems(files):
    counts = defaultdict(int)
    try:
        for stash in Stashes(files):
            league = stash['league']
            if not league: continue
            item_count = len(stash['items'])
            
            counts['total'] += 1
            counts['total_items'] += item_count
            counts[league] += 1
            counts[league + '_items'] += item_count
            if stash['public']:
                counts['total_public'] += 1
                counts[league + '_public'] += 1
            for item in stash['items']:
                if 'note' not in item: continue
                if '~price ' in item['note']:                
                    counts['total_priced_items'] += 1
                    counts[league + '_priced_items'] += 1
                elif '~b/o ' in item['note']:
                    counts['total_bo_items'] += 1
                    counts[league + '_bo_items'] += 1
    except KeyboardInterrupt: pass
    print(json.dumps(counts, indent=2, sort_keys=True))


def main():
    files = []
    for file_name in listdir('./'):
        if re.match('stashes-\d+.pickle', file_name):
            files.append(file_name)
    files.sort()
    #CountItems(files)
    with open('priced_items.pickle', 'wb') as out_file:
        for stash in Stashes(files):
            if stash['league'] != 'Legion': continue
            for item in stash['items']:
                if 'note' not in item or '~price ' not in item['note']: continue
                pickle.dump(item, out_file)                

if __name__ == '__main__':
    main()
