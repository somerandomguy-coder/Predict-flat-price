"""Bounding-box pull of the public NSW cadastre Lot layer. Sends only a box, never supplied rows."""
import json, sys, time, urllib.parse, urllib.request

OUT = sys.argv[1]
URL = "https://portal.spatial.nsw.gov.au/server/rest/services/NSW_Land_Parcel_Property_Theme/FeatureServer/8/query"
FIELDS = "cadid,createdate,modifieddate,planlabel,plannumber,lotnumber,lotidstring,planlotarea,planlotareaunits,startdate,enddate,lastupdate,classsubtype,itstitlestatus,hasstratum,stratumlevel"
base = dict(where="1=1", geometry="151.15,-33.935,151.245,-33.845", geometryType="esriGeometryEnvelope", inSR="4326",
            spatialRel="esriSpatialRelIntersects", outFields=FIELDS, returnGeometry="true", outSR="4326", f="geojson",
            orderByFields="objectid", resultRecordCount="2000")
feats, off = [], 0
while True:
    q = urllib.parse.urlencode({**base, "resultOffset": str(off)})
    req = urllib.request.Request(URL, data=q.encode(), headers={"User-Agent": "flat-rent-hackathon bbox query"})
    for attempt in range(4):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=120))
            break
        except Exception as e:
            print("retry", off, e, flush=True)
            time.sleep(5 * (attempt + 1))
    else:
        raise SystemExit("gave up")
    got = d.get("features", [])
    feats += got
    print(off, len(got), flush=True)
    if len(got) < 1:
        break
    off += len(got)
    time.sleep(0.5)
json.dump(dict(type="FeatureCollection", features=feats), open(OUT, "w"))
print("total", len(feats))
