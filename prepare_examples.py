import sys
import pickle
import re
import itertools
import tensorflow as tf
from collections import defaultdict

def Items(file_name):
    with open(file_name, 'rb') as f:
        try:
            while True:
                yield pickle.load(f)
        except EOFError: pass

# Ignore these for the model training.
BAD_CATEGORIES = set(['currency', 'maps', 'jewels', 'monsters', 'gems', 'cards', 'flasks'])

def UpdateMod(mods, mod_name, value):
    if mod_name not in mods:
        mods[mod_name] = (value, value)
    else:
        p_min, p_max = mods[mod_name]
        mods[mod_name] = (min(p_min, value),
                          max(p_max, value))

def GetPrice(note):
    match = re.match(r'~price (\d+) chaos', note)
    if not match: return None
    return float(match[1])

def GetScaledMods(mod):
    m = re.fullmatch(r'([+-]?\d+\.?\d*)([^\d]*(?:for \d+ second)?(?:per \d+)?(?:at least \d+)?(?:over \d+ second)?[^\d]*)', mod)
    if m:
        return [(m[2].strip(), float(m[1]))]
    m = re.fullmatch(r'([^\d]+) ([+-]?\d+\.?\d*)([^\d]*(?:for \d+ second)?[^\d]*)', mod)
    if m:
        return [(m[1].strip() + ' X ' + m[3].strip(),
                 float(m[2]))]
    m = re.fullmatch(r'([^\d]*)(\d+) to (\d+)(?: added)? (.*)', mod)
    if m:
        return [
            (m[1] + m[4] + '_min', float(m[2])),
            (m[1] + m[4] + '_max', float(m[3])),
            ]
    return None


def main():
    categories = set()
    names = set()
    mods = {}
    bool_mods = set()
    counters = defaultdict(int)
    selected_examples = []
    try:
        for item in Items('priced_items.pickle'):
            if [x for x in item['category'].keys() if x in BAD_CATEGORIES]:
                counters['bad_items'] +=1
                continue
            if item['frameType'] != 2:
                counters['bad_frame_%s' % item['frameType']] += 1
                continue
            if not item['identified']:
                counters['not_identified'] += 1
            price = GetPrice(item['note'])
            if not price:
                # print(item['note'], flush=True)            
                counters['bad_price'] += 1
                continue
            
            e = tf.train.Example()                            

            e.features.feature['name'].bytes_list.value.append(
                item['name'].encode('utf-8'))
            names.add(item['name'])
            
            e.features.feature['price'].float_list.value.append(price)
            e.features.feature['elder'].int64_list.value.append(
                1 if item.get('elder', False) else 0)
            e.features.feature['shaper'].int64_list.value.append(
                1 if item.get('shaper', False) else 0)
            e.features.feature['corrupted'].int64_list.value.append(
                1 if item.get('corrupted', False) else 0)
            e.features.feature['num_sockets'].int64_list.value.append(
                len(item.get('sockets', [])))
            
            for mod in itertools.chain(item.get('implicitMods', []),
                                       item.get('explicitMods', []),
                                       item.get('enchantMods', []),
                                       item.get('craftedMods', [])):                
                scaled_mods = GetScaledMods(mod)
                if not scaled_mods:
                    e.features.feature['bool_mods'].bytes_list.value.append(mod.encode('utf-8'))
                    if mod in bool_mods: continue
                    print(mod, flush=True)
                    bool_mods.add(mod)                    
                    continue
                for k, v in scaled_mods:
                    if k == 'to':
                        print(mod)
                        return
                    UpdateMod(mods, k, v)
                    e.features.feature['scaled_mods_name'].bytes_list.value.append(k.encode('utf-8'))
                    e.features.feature['scaled_mods_value'].float_list.value.append(v)
                    
            counters['good_items'] += 1
            selected_examples.append(e)
            for x,y in item['category'].items():
                categories.add('%s - %s' % (x,y))
    except KeyboardInterrupt:
        print('Stoped reading the items for debug purposes.')
        print('Interrupt again if you really want to quit.')
        print('Continuing with the training on partial data.')
    return
    # Rescale the value of the mods to be in [0, 1]
    for e in selected_examples:
        for i, mod in enumerate(e.features.feature['scaled_mods_name'].bytes_list.value):
            mod = str(mod, 'utf-8')
            original_value = e.features.feature['scaled_mods_value'].float_list.value[i]
            low, high = mods[mod]
            if high == low: high += 1
            scaled_value = (original_value - low) / (high - low)
            e.features.feature['scaled_mods_value'].float_list.value[i] = scaled_value
            
    print('\n\n Scalable mods list:\n')
    print('\n'.join(['%s: %s -> %s' %(k, low, high)
                     for k, (low, high) in sorted(mods.items())]))
    print('\n\n')
    print('Counters: \n',
          '\n'.join(['  %20s: %8d' % (k, v) for k,v in counters.items()])) 
    
    print('Dumping examples')
    with tf.io.TFRecordWriter('examples.rio') as writer:
        for e in selected_examples:
            writer.write(e.SerializeToString())
    with open('names-dict.txt', 'wb') as f:
        f.write(('\n'.join([x for x in sorted(names) if x])).encode('utf-8'))
    with open('scaled_mods-dict.txt', 'wb') as f:
        f.write(('\n'.join([x for x in sorted(mods.keys()) if x])).encode('utf-8'))
    with open('bool_mods-dict.txt', 'wb') as f:
        f.write(('\n'.join([x for x in sorted(bool_mods)])).encode('utf-8'))
                
    print('Done')


if __name__ == '__main__':
    main()
