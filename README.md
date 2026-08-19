# AI Cup 2026 — Bird-Group Classification

Radar tracks of things flying over the Wadden Sea, and the job is to work out which
kind of bird made each one. There are 9 bird groups plus a "Clutter" bucket for
non-bird returns. You submit a probability for each class and the leaderboard scores
you on macro mean Average Precision (mAP), which weights every class equally so the
rare birds matter as much as the common ones.

Classes: `Clutter`, `Cormorants`, `Pigeons`, `Ducks`, `Geese`, `Gulls`,
`Birds of Prey`, `Waders`, `Songbirds`.

## Pipeline

Three steps, run in order:

1. **`coordinate-into-csv.py`** — the `trajectory` column comes out of the Robin radar
   as a WKB hex string (a LineString ZM in SRID 4326). This decodes it into a list of
   `(lon, lat, altitude, RCS)` points. It writes straight back over `train.csv` and
   `test.csv`, so run it once up front and keep a copy of the originals somewhere safe.

2. **`features.py`** — turns each decoded track into a fixed-length vector of numeric
   features (see below). This is now the single home for all feature engineering.

3. **`model_v4.py`** — the model. It imports `add_features` from `features.py`, builds
   the features on train and test, trains the ensemble, and writes the submission. It
   no longer computes any features of its own.

## `features.py`

The public entry point is `add_features(df)`: hand it a DataFrame with decoded
trajectories and it returns the same frame with all the feature columns attached. The
full list of columns lives in `BROAD_COLS`. The features group into:

- **Geometry** — bearing, duration, track length, sinuosity, net displacement, bounding-box aspect ratio.
- **RCS (radar cross-section)** — a body-size proxy, summarised on both the dBsm scale (good for tree splits) and a linear m² scale (needed wherever a ratio has to be physical): mean/std/percentiles, trend, flicker, autocorrelation.
- **Altitude** — mean/spread/percentiles, trend and how much it wanders off that trend, fractions flown low (<50 m) and high (>200 m).
- **Vertical velocity** — climb/descent rates, time spent climbing vs. level, and how often the bird bobs up and down.
- **Speed** — true 3D airspeed through space (not ground speed), its variation, trend, and acceleration.
- **Turning / heading** — turn rates and cumulative turning, computed in metre-space so lat/lon distortion cancels out.
- **Cross-domain** — combinations that mix motion and RCS (scintillation, a kinetic-energy proxy, agility, etc.).
- **Bio-temporal** — where the track sits in the day/night cycle, using real sunrise/sunset at the radar site (Eemshaven): night flag, hours after sunset/sunrise, day length, twilight flag, and a nocturnal-migration-window flag.

You can also run `features.py` on its own (`python features.py`) to compute the features
once and cache them back into the CSVs, which saves recomputing them on every model run.

## `model_v4.py`

A CatBoost multiclass classifier, tuned on the heavy-regularization side so it doesn't
overfit the smallish training set. Rather than trust one model, it trains an ensemble:
5 stratified CV folds times 5 random seeds, so 25 models, and averages their
probabilities. Class imbalance is handled with softened inverse-frequency weights, with
a few manual boosts on top for the classes that are hard or worth getting right
(Cormorants, Birds of Prey, Waders, and so on).

Feature selection is deliberately simple: everything is a feature except identifiers,
labels, raw text, and the raw trajectory/timestamp columns. That means the model trains
on the engineered `BROAD_COLS` plus a few leftover raw radar summaries (`airspeed`,
`min_z`, `max_z`, and the categorical `radar_bird_size`, which CatBoost handles natively).

Scoring is done out-of-fold so the mAP number is honest. The run also prints a per-class
AP breakdown and the top feature importances, which is handy for seeing where it's weak,
then writes `submission_v4.csv` with one probability column per class in the order Kaggle
wants and `track_id` up front.

## Running it

You'll need Python 3.9+ and:

```bash
pip install pandas numpy scipy scikit-learn catboost astral pytz
```

Then, in order:

```bash
python coordinate-into-csv.py
```

```bash
python model_v4.py
```

The first command decodes the trajectories in place (back up the CSVs first). The second
builds the features via `features.py`, trains the 25-model ensemble, and writes
`submission_v4.csv`. If you'd rather cache the features so repeated model runs don't
recompute them, run `python features.py` once in between.

## Files

- `coordinate-into-csv.py` — decodes the raw WKB-hex `trajectory` column into point tuples (edits CSVs in place).
- `features.py` — all feature engineering; exposes `add_features(df)` and `BROAD_COLS`.
- `model_v4.py` — the CatBoost ensemble; imports features from `features.py` and writes the submission.
- `train.csv` / `test.csv` — the labelled and unlabelled radar tracks.
- `submission_v4.csv` — the model's output.
