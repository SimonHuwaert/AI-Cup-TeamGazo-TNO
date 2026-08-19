# AI-Cup-TeamGazo-TNO

TeamGazo's entry for the 2026 AI Cup on Kaggle, a bird classification challenge run with TNO.

The setup: you get radar tracks of things flying over the Wadden Sea and have to figure out which kind of bird made each one. There are 9 bird groups plus a "Clutter" bucket for non-bird returns. You submit a probability for each class and the leaderboard scores you on macro mean Average Precision (mAP), which weights every class equally so the rare birds matter as much as the common ones.

Heads up: this is the general model and methodology the team used, not necessarily the exact file behind the final submission. We couldn't pin down which version went in for the winning run, so `model_v4.py` is the representative pipeline.

## Classes

Clutter, Cormorants, Pigeons, Ducks, Geese, Gulls, Birds of Prey, Waders, Songbirds.

## What's in here

- `model_v4.py` — the whole pipeline: features, training, and writing out the submission.
- `coordinate-into-csv.py` — a preprocessing step that decodes the raw `trajectory` column from hex into actual coordinates. It edits the CSVs in place.
- `train.csv` — labelled tracks (~2,600 rows), with `bird_group` and `bird_species`.
- `test.csv` — the tracks you have to predict (~1,870 rows).
- `submission_v4.csv` — what `model_v4.py` spits out.

Both CSVs carry the radar fields: `track_id`, the two `timestamp_*_radar_utc` columns, `trajectory`, `trajectory_time`, `radar_bird_size`, `airspeed`, `min_z`, `max_z`. The training file also has the labels and the observation metadata (`observation_id`, `observer_position`, `observer_comment`, `n_birds_observed`, `bird_group`, `bird_species`).

## How it works

### Decoding the trajectories

The `trajectory` column comes out of the Robin radar as a WKB hex string (a LineString ZM in SRID 4326). `coordinate-into-csv.py` reads the binary and turns each track into a list of `(Lon, Lat, Alt, M)` points. It writes straight back over `train.csv` and `test.csv`, so run it once up front and keep a copy of the originals somewhere safe.

### Features

Bird activity follows the clock and the tide, so most of the feature work is about encoding time properly. Raw hour and month are a trap because they jump at the boundaries (hour 23 sits right next to hour 0 in reality but they're miles apart as numbers), so time of day and season both get a sin/cos pair instead. There's also an `is_night` flag.

The interesting bit is the tides. Instead of pulling in real tide data, the M2 semi-diurnal cycle (~12.42 h) gets reconstructed straight from the timestamp as a sin/cos phase, plus a spring/neap envelope on the ~14.77-day cycle. That gives the model a sense of high vs low water for free.

A bunch of columns get thrown out on purpose. Anything that leaks observer-specific geometry (distances to the observer, lat/lon, bearing, approach angle) tends not to generalise, so those go. Same for the raw calendar fields once the cyclic versions exist, all the `weather_*` columns, and duplicate `_calc` copies of features that already exist.

### The model

It's CatBoost, multiclass, tuned on the heavy-regularization side so it doesn't overfit the smallish training set. Rather than trust one model, it trains an ensemble: 5 stratified CV folds times 5 random seeds, so 25 models, and averages their probabilities. Class imbalance is handled with softened inverse-frequency weights, with a few manual boosts on top for the classes that are hard or worth getting right (Cormorants, Birds of Prey, Waders, and so on).

Scoring is done out-of-fold so the mAP number is honest. The run also prints a per-class AP breakdown and the top feature importances, which is handy for seeing where it's weak.

## Running it

You'll need Python 3.9+ and:

```bash
pip install pandas numpy scikit-learn catboost
```

Then, in order:

```bash
python coordinate-into-csv.py
```

```bash
python model_v4.py
```

The first command decodes the trajectories in place (back up the CSVs first). The second trains the ensemble and writes `submission_v4.csv`, with one probability column per class in the order Kaggle wants and `track_id` up front. It logs the per-seed and final OOF mAP, the per-class table, and the top features as it goes.

## Team

TeamGazo, 2026 AI Cup, with TNO.
