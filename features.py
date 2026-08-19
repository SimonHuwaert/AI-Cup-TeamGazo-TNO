"""
Feature engineering for the AI Cup 2026 bird-group classifier.

This module turns a raw radar track into a fixed-length vector of numeric
features. Each track arrives as a decoded trajectory — a list of
(lon, lat, altitude, RCS) points sampled over time — and this file distils it
down to summary statistics the model can learn from: how the bird moved
(geometry, speed, turning), how big it looked to the radar (RCS), how it used
altitude, and when it was flying relative to the sun (bio-temporal).

The public entry point is `add_features(df)`, which attaches all of the columns
in `BROAD_COLS` to a DataFrame and returns it. `model_v4.py` imports that so the
model never has to know how a feature is computed. Running this file directly
(`python features.py`) instead extracts the features and overwrites train.csv /
test.csv in place — handy for caching the results.

Requires the trajectory column to already be decoded into point tuples (see
coordinate-into-csv.py); it cannot read the raw WKB hex.
"""

import pandas as pd
import numpy as np
import math
import ast
import warnings
from scipy.stats import skew, kurtosis
from astral import LocationInfo
from astral.sun import sun
import pytz

warnings.filterwarnings('ignore')

# Reference location for sunrise/sunset. The radar sits near Eemshaven on the
# Dutch Wadden Sea coast; bird activity keys strongly on daylight there.
EEMSHAVEN = LocationInfo("Eemshaven", "Netherlands", "Europe/Amsterdam", 53.45, 6.83)
UTC = pytz.UTC

# The full ordered list of feature columns this module produces. The order here
# must match the order values are appended in `extract_broad_features`, since the
# two are zipped together by position when the row is written back.
BROAD_COLS = [
    # ── Geometry: the overall shape and extent of the ground track ──
    'bearing', 'duration', 'n_points', 'total_distance',
    'sinuosity', 'net_displacement', 'net_displacement_rate',
    'track_aspect_ratio',
    # ── RCS in dBsm (radar cross-section, a proxy for apparent body size) ──
    'rcs_mean', 'rcs_std', 'rcs_min', 'rcs_max',
    'rcs_median', 'rcs_iqr', 'rcs_skew', 'rcs_kurt',
    'rcs_p10', 'rcs_p90', 'rcs_range_90_10',
    'rcs_autocorr_1', 'rcs_trend', 'rcs_rate_mean', 'rcs_rate_std',
    'rcs_cv',
    # ── RCS on a linear scale (m²), needed wherever a ratio must be physical ──
    'rcs_linear_mean', 'rcs_linear_std', 'rcs_linear_cv',
    # ── Altitude: how high and how steadily the bird flew ──
    'z_mean', 'z_std', 'z_min', 'z_max', 'z_range',
    'z_cv', 'z_trend', 'z_residual_std',
    'z_p25', 'z_p75', 'z_iqr',
    'z_stable_frac',
    'frac_below_50m', 'frac_above_200m',
    # ── Vertical velocity: climbing / descending behaviour ──
    'vz_mean', 'vz_std', 'vz_abs_mean',
    'frac_climbing', 'frac_descending', 'frac_level',
    'max_climb_rate', 'max_descent_rate',
    'vz_oscillation_rate',
    # ── Speed through 3D space and its acceleration ──
    'speed_mean', 'speed_std', 'speed_max', 'speed_min',
    'speed_cv', 'speed_trend',
    'accel_mean', 'accel_std',
    # ── Turning / heading (computed in metres so lat/lon distortion cancels) ──
    'heading_rate', 'heading_change_cv',
    'cumulative_turn_angle',
    'turn_rate_mean', 'turn_rate_std', 'turn_rate_max',
    # ── Cross-domain combinations (use linear RCS where physics requires it) ──
    'scintillation_index',
    'kinetic_proxy',
    'turn_to_speed_ratio',
    'climb_efficiency',
    'rcs_speed_ratio',
    'formation_stability',
    'pseudo_energy',
    'flight_volatility',
    'z_speed_interaction',
    # ── Bio-temporal: timing of the track relative to the sun ──
    'is_night',
    'hours_after_sunset', 'hours_after_sunrise',
    'day_length_hours',
    'is_twilight',
    'is_nocturnal_migration_window',
]
NUM_BROAD_FEATURES = len(BROAD_COLS)


def calculate_bearing(lon1, lat1, lon2, lat2):
    """Initial great-circle bearing from point 1 to point 2, in degrees (0–360)."""
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    x = math.sin(dlon) * math.cos(lat2)
    y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


def compute_sun_times(timestamp):
    """Return (sunrise, sunset) at Eemshaven for the timestamp's date, or (None, None)."""
    try:
        s = sun(EEMSHAVEN.observer, date=timestamp.date(), tzinfo=UTC)
        return s['sunrise'], s['sunset']
    except Exception:
        return None, None


def extract_broad_features(row):
    """Compute every feature in `BROAD_COLS` for a single track (one DataFrame row).

    Returns a pandas Series of length `NUM_BROAD_FEATURES` in `BROAD_COLS` order.
    Tracks that are too short to be meaningful, or that fail to parse, yield a row
    of NaNs so the model can treat them as missing rather than crashing the run.
    """
    try:
        # The trajectory is stored as a bare comma-separated list of point tuples;
        # wrapping it in brackets makes it a valid Python literal to parse.
        traj_str = f"[{row['trajectory']}]"
        trajectory = ast.literal_eval(traj_str)
        times = np.array(ast.literal_eval(row['trajectory_time']))

        # Need at least two points to define any motion at all.
        if len(trajectory) < 2:
            return pd.Series([np.nan] * NUM_BROAD_FEATURES)

        lons = np.array([p[0] for p in trajectory])
        lats = np.array([p[1] for p in trajectory])
        zs   = np.array([p[2] for p in trajectory])   # altitude in metres
        rcs  = np.array([p[3] for p in trajectory])   # radar cross-section in dBsm

        n_points = len(lons)
        eps = 1e-6   # guards every division against zero

        # ── Geometry ──
        bearing  = calculate_bearing(lons[0], lats[0], lons[-1], lats[-1])
        duration = max(times[-1] - times[0], 1.0)     # seconds, floored at 1s

        # Per-segment time deltas; a zero delta would blow up rates, so nudge it.
        dt = np.diff(times)
        dt[dt == 0] = 0.5

        # Convert degree steps to metres. Longitude degrees shrink with latitude,
        # so scale them by cos(lat); latitude degrees are ~constant at 111 km.
        cos_lat = np.cos(np.radians(np.mean(lats)))
        d_lon_m = np.diff(lons) * 111320 * cos_lat
        d_lat_m = np.diff(lats) * 111000
        d_z_m   = np.diff(zs)

        seg_dist_2d = np.sqrt(d_lon_m**2 + d_lat_m**2)                 # ground distance
        seg_dist_3d = np.sqrt(d_lon_m**2 + d_lat_m**2 + d_z_m**2)      # true path length
        total_distance = np.sum(seg_dist_2d)                          # keep 2D as "ground distance"

        # Straight-line start-to-end displacement, and how directly the bird flew.
        net_dx = (lons[-1] - lons[0]) * 111320 * cos_lat
        net_dy = (lats[-1] - lats[0]) * 111000
        net_displacement = np.sqrt(net_dx**2 + net_dy**2)
        net_displacement_rate = net_displacement / (duration + eps)
        sinuosity = total_distance / (net_displacement + eps)         # 1 = perfectly straight

        # Aspect ratio of the track's bounding box: elongated vs. compact/erratic.
        if n_points >= 3:
            lon_range = np.ptp(lons) * 111320 * cos_lat
            lat_range = np.ptp(lats) * 111000
            track_aspect_ratio = max(lon_range, lat_range) / (min(lon_range, lat_range) + eps)
        else:
            track_aspect_ratio = 1.0

        # ── RCS features (dBsm scale — kept because tree splits like the raw scale) ──
        rcs_mean   = np.mean(rcs)
        rcs_std    = np.std(rcs)
        rcs_min    = np.min(rcs)
        rcs_max    = np.max(rcs)
        rcs_median = np.median(rcs)
        rcs_iqr    = np.percentile(rcs, 75) - np.percentile(rcs, 25)
        rcs_p10    = np.percentile(rcs, 10)
        rcs_p90    = np.percentile(rcs, 90)
        rcs_range_90_10 = rcs_p90 - rcs_p10
        rcs_skew_val = skew(rcs) if n_points >= 3 else 0.0
        rcs_kurt_val = kurtosis(rcs) if n_points >= 4 else 0.0
        rcs_cv     = rcs_std / (abs(rcs_mean) + eps)

        # Lag-1 autocorrelation: does the RCS vary smoothly or flicker point-to-point?
        if n_points >= 4:
            rcs_c = rcs - rcs_mean
            rcs_autocorr_1 = np.corrcoef(rcs_c[:-1], rcs_c[1:])[0, 1]
            if np.isnan(rcs_autocorr_1):
                rcs_autocorr_1 = 0.0
        else:
            rcs_autocorr_1 = 0.0

        # Linear trend of RCS over the track (slope of a first-order fit).
        if n_points >= 3:
            rcs_trend = np.polyfit(np.arange(n_points), rcs, 1)[0]
        else:
            rcs_trend = 0.0

        # How fast RCS changes between samples.
        rcs_rate = np.diff(rcs) / dt
        rcs_rate_mean = np.mean(np.abs(rcs_rate))
        rcs_rate_std  = np.std(rcs_rate)

        # ── RCS on a linear scale (m²) ──
        # Ratios and energies below must use physical area, so undo the dB (log) scale.
        rcs_linear = 10.0 ** (rcs / 10.0)
        rcs_linear_mean = np.mean(rcs_linear)
        rcs_linear_std  = np.std(rcs_linear)
        rcs_linear_cv   = rcs_linear_std / (rcs_linear_mean + eps)

        # ── Altitude features ──
        z_mean = np.mean(zs)
        z_std  = np.std(zs)
        z_min  = np.min(zs)
        z_max  = np.max(zs)
        z_range = z_max - z_min
        z_cv   = z_std / (abs(z_mean) + eps)
        z_p25  = np.percentile(zs, 25)
        z_p75  = np.percentile(zs, 75)
        z_iqr  = z_p75 - z_p25
        z_stable_frac = np.mean(np.abs(zs - z_mean) < 10)   # fraction within 10 m of mean
        frac_below_50m  = np.mean(zs < 50)
        frac_above_200m = np.mean(zs > 200)

        # Altitude trend over real time, plus how much it wanders off that trend.
        time_sec = times - times[0]
        if len(np.unique(time_sec)) >= 2:
            z_trend = np.polyfit(time_sec, zs, 1)[0]
        else:
            z_trend = 0.0
        if len(np.unique(time_sec)) >= 3:
            z_pred = np.poly1d(np.polyfit(time_sec, zs, 1))(time_sec)
            z_residual_std = np.std(zs - z_pred)
        else:
            z_residual_std = 0.0

        # ── Vertical velocity (rate of climb/descent, m/s) ──
        vz = d_z_m / dt
        vz_mean     = np.mean(vz) if len(vz) > 0 else 0.0
        vz_std      = np.std(vz)  if len(vz) > 0 else 0.0
        vz_abs_mean = np.mean(np.abs(vz)) if len(vz) > 0 else 0.0
        frac_climbing   = np.mean(vz > 0.5)  if len(vz) > 0 else 0.0
        frac_descending = np.mean(vz < -0.5) if len(vz) > 0 else 0.0
        frac_level      = np.mean(np.abs(vz) <= 0.5) if len(vz) > 0 else 0.0
        max_climb_rate  = np.max(vz)  if len(vz) > 0 else 0.0
        max_descent_rate = np.min(vz) if len(vz) > 0 else 0.0

        # How often the climb/descent flips sign — bobbing vs. steady flight.
        if len(vz) >= 4:
            signs = np.sign(vz)
            zero_crossings = np.sum(np.abs(np.diff(signs)) > 0)
            vz_oscillation_rate = zero_crossings / duration
        else:
            vz_oscillation_rate = 0.0

        # ── Speed features (3D — true airspeed through space, not ground speed) ──
        speeds = seg_dist_3d / dt
        speed_mean = np.mean(speeds) if len(speeds) > 0 else 0.0
        speed_std  = np.std(speeds)  if len(speeds) > 0 else 0.0
        speed_max  = np.max(speeds)  if len(speeds) > 0 else 0.0
        speed_min  = np.min(speeds)  if len(speeds) > 0 else 0.0
        speed_cv   = speed_std / (speed_mean + eps)

        if len(speeds) >= 3:
            speed_trend = np.polyfit(np.arange(len(speeds)), speeds, 1)[0]
        else:
            speed_trend = 0.0

        if len(speeds) >= 2:
            accel = np.diff(speeds) / dt[:-1] if len(dt) > 1 else np.array([0.0])
            accel_mean = np.mean(np.abs(accel))
            accel_std  = np.std(accel)
        else:
            accel_mean = accel_std = 0.0

        # ── Heading / turning (angles from metre-space steps, so they're undistorted) ──
        if n_points >= 3:
            angles = np.arctan2(d_lat_m, d_lon_m)
            angle_changes = np.diff(angles)
            # Wrap each change into [-pi, pi] and take magnitude, so a turn past
            # due-north doesn't register as a near-360° swing.
            angle_changes = np.abs((angle_changes + np.pi) % (2 * np.pi) - np.pi)

            heading_rate      = np.sum(angle_changes) / duration
            heading_change_cv = np.std(angle_changes) / (np.mean(angle_changes) + eps)
            cumulative_turn_angle = np.sum(angle_changes)
            turn_rate_mean = np.mean(angle_changes)
            turn_rate_std  = np.std(angle_changes)
            turn_rate_max  = np.max(angle_changes)
        else:
            heading_rate = heading_change_cv = cumulative_turn_angle = 0.0
            turn_rate_mean = turn_rate_std = turn_rate_max = 0.0

        # ── Cross-domain interactions (mix motion and RCS; use linear RCS) ──
        scintillation_index = rcs_linear_cv                                  # radar-return flicker
        kinetic_proxy       = (speed_mean ** 2) / (rcs_linear_mean + eps)    # speed vs. size
        turn_to_speed_ratio = turn_rate_max / (speed_max + eps)              # agility
        climb_efficiency    = vz_mean / (speed_mean + eps)                   # climb per unit speed
        rcs_speed_ratio     = rcs_linear_mean / (speed_mean + eps)
        formation_stability = speed_mean / (rcs_linear_std + eps)
        pseudo_energy       = rcs_linear_mean * (speed_mean ** 2)            # mass-like × speed²
        flight_volatility   = speed_std * z_std                             # jitter in speed and height
        z_speed_interaction = z_mean * speed_mean

        # ── Bio-temporal features (where the track falls in the day/night cycle) ──
        ts = pd.Timestamp(row['timestamp_start_radar_utc'])
        sunrise, sunset = compute_sun_times(ts)

        if sunrise is not None and sunset is not None:
            ts_utc = ts.tz_localize(UTC) if ts.tzinfo is None else ts.tz_convert(UTC)
            day_length_hours    = (sunset - sunrise).total_seconds() / 3600
            hours_after_sunset  = (ts_utc - sunset).total_seconds() / 3600
            hours_after_sunrise = (ts_utc - sunrise).total_seconds() / 3600
            is_night = int(ts_utc < sunrise or ts_utc > sunset)
            # Roughly the civil-twilight windows around dawn and dusk.
            is_twilight = int(
                (-1.5 <= hours_after_sunrise <= 0) or (0 <= hours_after_sunset <= 1.5)
            )
            # The first few hours after dusk, when nocturnal migrants take off.
            is_nocturnal_migration_window = int(1 <= hours_after_sunset <= 5)
        else:
            is_night = 0
            hours_after_sunset = hours_after_sunrise = day_length_hours = 0.0
            is_twilight = is_nocturnal_migration_window = 0

        # Assemble in exactly BROAD_COLS order (positions must line up).
        values = [
            # Geometry
            bearing, duration, n_points, total_distance,
            sinuosity, net_displacement, net_displacement_rate,
            track_aspect_ratio,
            # RCS dBsm
            rcs_mean, rcs_std, rcs_min, rcs_max,
            rcs_median, rcs_iqr, rcs_skew_val, rcs_kurt_val,
            rcs_p10, rcs_p90, rcs_range_90_10,
            rcs_autocorr_1, rcs_trend, rcs_rate_mean, rcs_rate_std,
            rcs_cv,
            # RCS linear
            rcs_linear_mean, rcs_linear_std, rcs_linear_cv,
            # Altitude
            z_mean, z_std, z_min, z_max, z_range,
            z_cv, z_trend, z_residual_std,
            z_p25, z_p75, z_iqr,
            z_stable_frac,
            frac_below_50m, frac_above_200m,
            # Vertical velocity
            vz_mean, vz_std, vz_abs_mean,
            frac_climbing, frac_descending, frac_level,
            max_climb_rate, max_descent_rate,
            vz_oscillation_rate,
            # Speed
            speed_mean, speed_std, speed_max, speed_min,
            speed_cv, speed_trend,
            accel_mean, accel_std,
            # Turning
            heading_rate, heading_change_cv,
            cumulative_turn_angle,
            turn_rate_mean, turn_rate_std, turn_rate_max,
            # Cross-domain
            scintillation_index, kinetic_proxy,
            turn_to_speed_ratio, climb_efficiency,
            rcs_speed_ratio, formation_stability,
            pseudo_energy, flight_volatility,
            z_speed_interaction,
            # Bio-temporal
            is_night,
            hours_after_sunset, hours_after_sunrise,
            day_length_hours,
            is_twilight,
            is_nocturnal_migration_window,
        ]

        return pd.Series(values)

    except Exception:
        # Any parsing/maths failure on a single track becomes an all-NaN row.
        return pd.Series([np.nan] * NUM_BROAD_FEATURES)


def add_features(df):
    """Attach every `BROAD_COLS` feature to `df` and return the enriched DataFrame.

    This is the function the model imports. It does not touch disk. Timestamps are
    parsed and the categorical `radar_bird_size` is cleaned so downstream code gets
    consistent dtypes. If the frame already carries feature columns from a previous
    run, they are dropped first so features are always recomputed, not duplicated.
    """
    df = df.copy()

    if 'timestamp_start_radar_utc' in df.columns:
        df['timestamp_start_radar_utc'] = pd.to_datetime(df['timestamp_start_radar_utc'])

    if 'radar_bird_size' in df.columns:
        df['radar_bird_size'] = df['radar_bird_size'].fillna("Unknown").astype(str)

    # Drop stale feature columns so a re-run overwrites rather than duplicates them.
    existing = [c for c in BROAD_COLS if c in df.columns]
    if existing:
        df = df.drop(columns=existing)

    df[BROAD_COLS] = df.apply(extract_broad_features, axis=1)
    return df


if __name__ == "__main__":
    # Standalone mode: compute the features once and cache them back into the CSVs.
    for file_name in ["train.csv", "test.csv"]:
        print(f"--- Processing {file_name} ---")
        try:
            df = pd.read_csv(file_name)
            print(f"  Extracting {NUM_BROAD_FEATURES} features for {len(df)} tracks...")
            df = add_features(df)
            df.to_csv(file_name, index=False)
            print(f"  ✅ Overwrote {file_name} ({len(df)} rows, {len(df.columns)} cols)")
        except FileNotFoundError:
            print(f"  ⚠️ {file_name} not found, skipping")