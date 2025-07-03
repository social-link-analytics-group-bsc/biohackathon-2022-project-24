import csv
import json

with open('raw.csv', 'r') as f:
    reader = csv.reader(f)
    headers = next(reader)
    out = [dict(zip(headers, i)) for i in reader]

with open('raw.jsonl', 'w') as o:
    for line in out:
        json.dump(line, o)
        o.write('\n')