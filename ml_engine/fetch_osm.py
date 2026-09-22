import urllib.request
import json

query = """[out:json][timeout:15];
way["name"="HITEC City Road"](17.435,78.370,17.455,78.385);
out geom;
"""

import urllib.parse

for endpoint in [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
    "https://maps.mail.ru/osm/tools/overpass/api/interpreter"
]:
    try:
        print(f"Trying {endpoint}...")
        data_encoded = urllib.parse.urlencode({"data": query}).encode('utf-8')
        req = urllib.request.Request(
            endpoint,
            data=data_encoded,
            headers={"User-Agent": "TrueTrackNav/1.0"}
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode('utf-8'))
            elements = data.get('elements', [])
            print(f"Success! Found {len(elements)} ways.")
            with open("ml_engine/osm_ways.json", "w") as f:
                json.dump(data, f, indent=2)
            break
    except Exception as e:
        print(f"Failed {endpoint}: {e}")
