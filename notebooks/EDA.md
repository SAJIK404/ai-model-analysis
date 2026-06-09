# Exploratory Data Analysis (notes)

This file outlines quick steps to reproduce basic EDA locally.

1. Activate venv and install deps

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

2. Load data into SQLite

```bash
python scripts/load_data.py
```

3. Start an interactive Python session or Jupyter notebook and run:

```python
import pandas as pd
import sqlite3
from pathlib import Path
p = Path("data/ai_models.db")
conn = sqlite3.connect(p)
models = pd.read_sql_query("SELECT * FROM models", conn)
bench = pd.read_sql_query("SELECT * FROM benchmarks", conn)
pricing = pd.read_sql_query("SELECT * FROM pricing", conn)

models.head()
bench.head()
pricing.tail()
```

4. Use `scripts/visualize.py` to generate charts to `output/` and open PNGs.

5. Run `scripts/evaluate.py` to produce bootstrap stability report.
