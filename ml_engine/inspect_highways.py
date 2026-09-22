import json

with open('ml_engine/osm_bbox_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

nodes = {e['id']: (e['lat'], e['lon']) for e in data['elements'] if e['type'] == 'node'}
ways = [e for e in data['elements'] if e['type'] == 'way' and 'tags' in e and 'highway' in e['tags']]

print(f"Total highway ways: {len(ways)}")
for w in ways:
    tags = w['tags']
    name = tags.get('name', 'unnamed')
    hw = tags.get('highway', '')
    tunnel = tags.get('tunnel', 'no')
    layer = tags.get('layer', '0')
    if hw in ['primary', 'secondary', 'trunk', 'tertiary', 'primary_link'] or tunnel == 'yes' or 'HITEC' in name:
        coords = [nodes[nid] for nid in w['nodes'] if nid in nodes]
        if len(coords) > 1:
            print(f"Way {w['id']}: name='{name}', hw={hw}, tunnel={tunnel}, layer={layer}, count={len(coords)}, start=({coords[0][0]:.5f},{coords[0][1]:.5f}), end=({coords[-1][0]:.5f},{coords[-1][1]:.5f})")
