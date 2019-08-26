import sys
import api

def main():
  start = api.ID()
  end = api.ID('469922119-486744496-459190039-525283228-498741791')

  while start < end:
    mid = (start + end) / 2
    data = api.Get('%s' % mid)
    true_mid = api.ID(data['next_change_id'])
    print(start.num, '\n ->', end, '\n: ', true_mid, '\n ', mid)
    sys.stdout.flush()
    if start == true_mid or end == true_mid:
        print('Exhausted!!!')
        
    leagues = set()
    for stash in data['stashes']:
      if 'league' in stash: leagues.add(stash['league'])
      for item in stash.get('items', []):
        if 'league' in item: leagues.add(stash['league'])
    print('Leagues found: ', leagues)
    if 'Legion' in leagues or  'Hardcore Legion' in leagues:
        end = true_mid
    else:
        start = true_mid
    
          

if __name__ == '__main__':
    main()
