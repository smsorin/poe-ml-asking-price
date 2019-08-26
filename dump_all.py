import constants
import api
import sys
import pickle

def getFile(index):
    return 'stashes-%05d.pickle' % index

def main():
    change_id = api.ID(constants.LEGION_LEAGUE_START)
    index = 0
    writer = open(getFile(index), 'wb')
    call_count = 0
    while True:
        print(change_id)
        sys.stdout.flush()
        data = api.Get(change_id)
        if change_id == data['next_change_id']:
            print('DONE')
            break
        change_id = data['next_change_id']
        for stash in data['stashes']:
            pickle.dump(stash, writer)
        call_count += 1
        if call_count % 376 == 0:
            writer.close()
            index += 1
            with open('next_change_id', 'wt') as f:
                f.write(change_id)
            writer = open(getFile(index), 'wb')
    writer.close()


if __name__ == '__main__':
    main()
