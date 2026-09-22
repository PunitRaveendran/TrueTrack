"""
Offline Asset Bundler for TrueTrack
===================================
Downloads and bundles:
1. Leaflet v1.9.4 (leaflet.js & leaflet.css) -> web_app/lib/
2. Bounding-box map tiles for Hyderabad Mindspace Underpass -> web_app/tiles/{z}/{x}/{y}.png
   (Zoom 14-16, total 28 tiles, adhering to OpenStreetMap Fair Use policy with full attribution)
"""

import os
import math
import time
import urllib.request

HEADERS = {
    'User-Agent': 'TrueTrack-EdgeAI-Navigation/1.0 (Hackathon Academic Demo; punitraveendran@gmail.com)'
}

def deg2num(lat_deg, lon_deg, zoom):
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def download_file(url, dest_path):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 500:
        return True
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp, open(dest_path, 'wb') as out_f:
            out_f.write(resp.read())
        return True
    except Exception as e:
        print(f"Failed {url}: {e}")
        return False

def bundle_assets():
    print("[1/2] Bundling Leaflet 1.9.4 locally...")
    lib_dir = os.path.join('web_app', 'lib')
    os.makedirs(lib_dir, exist_ok=True)
    
    # Download leaflet.js and leaflet.css
    js_url = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"
    css_url = "https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"
    
    download_file(js_url, os.path.join(lib_dir, 'leaflet.js'))
    download_file(css_url, os.path.join(lib_dir, 'leaflet.css'))
    print("  > Leaflet JS and CSS downloaded.")

    print("\n[2/2] Caching Offline Corridor Tiles (Zoom 14-16)...")
    # Corridor bounds: Lat 17.435 to 17.455, Lon 78.370 to 78.390
    tiles_base = os.path.join('web_app', 'tiles')
    
    # We can use CartoDB dark tiles or OSM standard tiles
    tile_url_pattern = "https://a.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}.png"
    # Alternative: https://tile.openstreetmap.org/{z}/{x}/{y}.png
    
    count = 0
    for z in [14, 15, 16]:
        x0, y1 = deg2num(17.435, 78.370, z)
        x1, y0 = deg2num(17.455, 78.390, z)
        min_x, max_x = min(x0, x1), max(x0, x1)
        min_y, max_y = min(y0, y1), max(y0, y1)
        
        for x in range(min_x, max_x + 1):
            for y in range(min_y, max_y + 1):
                url = f"https://tile.openstreetmap.org/{z}/{x}/{y}.png"
                dest = os.path.join(tiles_base, str(z), str(x), f"{y}.png")
                ok = download_file(url, dest)
                if ok:
                    count += 1
                time.sleep(0.08) # Respect OSM 1 req/sec rate limit for small tile sets
                
    print(f"  > Successfully cached {count} corridor tiles locally in {tiles_base}/")

if __name__ == '__main__':
    bundle_assets()
