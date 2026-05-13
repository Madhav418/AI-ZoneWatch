import json
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

class ZoneManager:

    def __init__(self, zone_file):

        with open(zone_file, 'r') as f:
            data = json.load(f)

        self.zones = []

        for zone in data['zones']:
            polygon = Polygon(zone['points'])

            self.zones.append({
                'name': zone['name'],
                'type': zone['type'],
                'polygon': polygon,
                'loiter_time': zone.get('loiter_time', 0)
            })

    def check_zones(self, track):

        x1, y1, x2, y2 = track['bbox']

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)

        point = Point(center_x, center_y)

        matched = []

        for zone in self.zones:
            if zone['polygon'].contains(point):
                matched.append(zone)

        return matched