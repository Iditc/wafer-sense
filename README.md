# WaferSense

End-to-end anomaly detection for semiconductor wafer maps: defect pattern
recognition, discovery of unseen defect types, lot-level tool excursion
alerts, and an LLM agent that proposes root-cause hypotheses.

## Dataset
[WM-811K](https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map) –
811,457 real wafer maps from semiconductor fabs (Wu et al., IEEE TSM 2015).

## Project Log
Steps are documented here in chronological order.

### Step 0 – Project setup
Repository structure, configuration and conventions.

### Step 1 – Data loading (`src/data/load.py`)
Raw pickle (2.1 GB, saved with old pandas) converted to a flat parquet cache (178 MB).
- 811,457 wafers from 46,293 lots
- 172,950 labeled (21%): 147,431 `none` + 25,519 defects in 8 patterns
- Highly imbalanced: Edge-Ring 9,680 vs Near-full 149

**Why parquet:** the pickle is slow, large and can execute code when loaded.
It is read once; all later steps load the parquet (12x smaller, columnar, fast).

**Transformations:**
- **Compatibility:** module paths from old pandas are aliased to their new location
  (fallback only, used if loading fails).
- **Labels unwrapped:** `[['Scratch']]` -> `'Scratch'`, `[]` -> `None`.
  Note the difference: `none` = inspected, no defect; `None` = never labeled.
  Unlabeled wafers feed self-supervised learning; `none` wafers are normal examples.
- **Columns renamed and typed:**

  | Original | New | Purpose |
  |---|---|---|
  | `lotName` | `lot_name` | Lot-based train/test split |
  | `waferIndex` | `wafer_index` (int) | Position of the wafer within its lot |
  | `dieSize` | `die_size` | Yield impact estimation |
  | `trianTestLabel` | `orig_split` | Original split (typo in source); not used, since it is not lot-based |
  | – | `wafer_id` | Unique wafer identifier |

- **Wafer maps:** values are 0 = outside wafer, 1 = normal die, 2 = defective die.
  Maps vary in shape (e.g. 26x26, 52x53), so each is stored flattened (`map_flat`,
  `uint8`) with its `map_h` / `map_w`; `map_from_row()` restores the 2D map.
