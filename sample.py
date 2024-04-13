import json
from pathlib import Path
import random
path = "output_new.json"

with open(path, 'r') as f:
    data = json.load(f)

random_numbers = random.sample(range(1, 2001), 50)

data_new = []
for i in random_numbers:
    piece = data[i]
    data_new.append(piece)

output_file = Path('output_test.json')
output_file.write_text(json.dumps(data_new, indent=2, ensure_ascii=False))