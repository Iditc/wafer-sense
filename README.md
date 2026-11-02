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
  | `dieSize` | `die_size` | Number of dies on the wafer (verified: equals the die count on the map); used for yield impact |
  | `trianTestLabel` | `orig_split` | Original split (typo in source); not used, since it is not lot-based |
  | – | `wafer_id` | Unique wafer identifier |

- **Wafer maps:** values are 0 = outside wafer, 1 = normal die, 2 = defective die.
  Maps vary in shape (e.g. 26x26, 52x53), so each is stored flattened (`map_flat`,
  `uint8`) with its `map_h` / `map_w`; `map_from_row()` restores the 2D map.

### Step 2 – EDA (`src/data/eda.py`, `src/data/eda_lots.py`)
Numbers in `results/metrics/eda_summary.json`.

A **lot** is a batch of wafers (up to 25) that move through the fab together and are
processed by the same tools, so a tool problem tends to hit the whole lot.

**1. Only 21% of wafers are labeled** (172,950 of 811,457).
Decision: use the unlabeled 79% for self-supervised pretraining.

![Labeled share](results/figures/01a_labeled_share.png)

**2. Strong class imbalance:** O.K. 147,431 vs Near-full 149 (~1000x); among defects,
Edge-Ring 9,680 vs Near-full 149 (65x).
Decision: F1 macro as primary metric, class weighting.

![Class distribution](results/figures/01_class_distribution.png)

**3. Each pattern has a distinct look.** Three typical wafers per class (closest to the
class median defect share, not hand-picked). `none` is shown as **O.K.** in all figures.
Scratches are thin and often faint, making Scratch visually the hardest class.
Some wafers look elliptical: the map only counts dies per row/column and draws each die
as a square, so wafers with rectangular dies appear stretched. Resizing removes this.

![Example maps](results/figures/02_example_maps.png)

**4. Each product has its own die grid** (632 distinct shapes). Wafers are the same
physical size, but products with smaller dies fit more of them: from 515 dies (25x27
grid) to 5,633 dies (88x81 grid).
Decision: resize all maps to one fixed size before the CNN (nearest-neighbour, so the
0/1/2 values are preserved).

![Map sizes](results/figures/03_map_sizes.png)

**5. Defect rate is flat across wafer position** (14–15.5%).
Decision: no global position effect; excursions must be detected per lot.

![Defect rate by index](results/figures/04_defect_rate_by_index.png)

**6. A full lot is 25 wafers** (standard lot size): 47% of lots are full, and 75% of
all wafers sit in lots of 24–25. Smaller lots (peaks at 1 and 12–13 wafers) are likely
split or partially recorded lots.

![Wafers per lot](results/figures/05_wafers_per_lot.png)

**7. Labeling was mostly done per lot (87% of lots are all-or-nothing):** 35,531 lots fully unlabeled, 4,685 fully labeled, 6,077 mixed.
Decision: a lot-based split matches how labels were produced.

![Labeling per lot](results/figures/06_labeling_per_lot.png)

**8. Patterns repeat within a lot:** in 51.5% of lots with 2+ defective wafers,
all defects share one pattern. The peak at 0.5 is mostly lots with exactly two
defects of different types.
Decision: a random split would leak (near-identical wafers in train and test);
the lot-level excursion feature is grounded in the data.

![Pattern repetition](results/figures/07_pattern_repetition.png)
![Example lot](results/figures/08_example_lot.png)

**9. Ambiguous labels:** the weakest `Center` wafers have ~1% defective dies (a tiny
central cluster), while some `none` wafers have ~33% scattered defects that look like
`Random`. WM-811K labels are known to be inconsistent; this caps achievable accuracy
and is a reason to inspect model errors visually.

![Label noise](results/figures/09_label_noise.png)
