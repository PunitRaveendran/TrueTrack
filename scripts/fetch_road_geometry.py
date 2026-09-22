import urllib.request
import urllib.parse
import json

url = 'https://overpass-api.de/api/interpreter'
q = """[out:json];
(
  way["name"="HITEC City Road"](17.440,78.373,17.452,78.385);
  way["name"~"Mindspace|Cyber|Hitec"](17.440,78.373,17.452,78.385);
);
out geom;"""

data = urllib.parse.urlencode({'data': q}).encode('utf-8')
req = urllib.request.Request(url, data=data, headers={'User-Agent': 'TrueTrack-Geo/1.0'})

try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        res = json.loads(resp.read().decode('utf-8'))
    print(f"Found {len(res['elements'])} road elements.")
    with open('scripts/real_road_elements.json', 'w') as f:
        json.dump(res, f, indent=2)
    for e in res['elements'][:10]:
        print(f"Way {e['id']}: {e.get('tags', {}).get('name')} - {len(e.get('geometry', []))} pts")
        if e.get('geometry'):
            print(f"   Start: {e['geometry'][0]} -> End: {e['geometry'][-1]}")
except Exception as exc:
    print(f"Error fetching from Overpass: {exc}")
