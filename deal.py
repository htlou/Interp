import json
from pathlib import Path
path = "output.json"

with open(path, 'r') as f:
    data = json.load(f)

data_new = []
for piece in data:
    piece["completion"] = (" "+piece["completion"])*20
    data_new.append(piece)

output_file = Path('output_new.json')
output_file.write_text(json.dumps(data_new, indent=2, ensure_ascii=False))