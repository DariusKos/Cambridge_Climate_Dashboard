#!/usr/bin/env python3
"""
Cambridge climate dashboard — single-file pipeline.

Reads three CSV files (paths set in CONFIG below) and writes one
self-contained interactive HTML file with tabbed sections.

Required Python packages:
    pip install pandas numpy scipy statsmodels
Optional (enables the Drought / SPEI tab):
    pip install climate_indices

Usage:
    1. Place this script in the same folder as your three input CSVs
       (or edit the CONFIG paths below).
    2. Run:  python3 build_dashboard.py
    3. Open the resulting cambridge_climate_dashboard.html in any browser,
       or commit it to a GitHub Pages site to share publicly.

Inputs expected (all CSVs):
    - midas_merged_daily_airtemp.csv     (MIDAS Open daily air temp, BG)
        columns: date, max_air_temp, min_air_temp
    - midas_merged_daily_groundtemp.csv  (MIDAS Open daily soil temp)
        columns: date, src_id, q10cm_soil_temp, q30cm_soil_temp,
                 q100cm_soil_temp  (other depths ignored)
    - NIAB_daily_rainfall_NOAA.csv       (NOAA GHCN-D daily precip)
        columns: DATE, PRCP   (millimetres)

Outputs:
    - cambridge_climate_dashboard.html in the current directory.
"""

from __future__ import annotations
from pathlib import Path
import json
import warnings
import sys

import numpy as np
import pandas as pd

try:
    import statsmodels.api as sm
    from scipy.stats import nbinom, poisson as sp_poisson
except ImportError as e:
    sys.exit("Please install dependencies first:\n    "
             "pip install pandas numpy scipy statsmodels\n"
             f"(missing: {e.name})")

# climate_indices is optional: if absent, the SPEI tab is simply omitted
# rather than failing the whole build.
try:
    from climate_indices import indices as ci_indices
    from climate_indices import compute as ci_compute
    from climate_indices import eto as ci_eto
    HAVE_CLIMATE_INDICES = True
    # Quieten climate_indices' verbose structlog/standard logging output
    import logging as _logging
    _logging.getLogger("climate_indices").setLevel(_logging.ERROR)
    for _noisy in ("climate_indices.eto", "climate_indices.compute",
                   "climate_indices.indices"):
        _logging.getLogger(_noisy).setLevel(_logging.ERROR)
except ImportError:
    HAVE_CLIMATE_INDICES = False

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# CONFIG — edit these if your CSVs live elsewhere
# ---------------------------------------------------------------------------
HERE = Path(__file__).parent
AIRTEMP_CSV  = HERE / "midas_merged_daily_airtemp.csv"
GROUND_CSV   = HERE / "midas_merged_daily_groundtemp.csv"
RAINFALL_CSV = HERE / "NIAB_daily_rainfall_NOAA.csv"
OUTPUT_HTML  = HERE / "cambridge_climate_dashboard.html"

BASELINE_START = 1960
BASELINE_END   = 1990  # inclusive — WMO climate-normal-ish reference window
PROJECT_TO     = 2050
MIN_OBS_PER_YEAR = 340  # exclude years with poor daily coverage

# SPEI configuration
STATION_LATITUDE = 52.2   # Cambridge Botanic Garden, degrees north (for PET)
SPEI_CALIB_START = 1961   # calibration period for the SPEI distribution fit
SPEI_CALIB_END   = 1990   # (the WMO 1961–1990 standard normal)
SPEI_SCALES      = [3, 12]  # months: SPEI-3 (seasonal), SPEI-12 (annual/hydrological)


# ===========================================================================
# DATA LOADING
# ===========================================================================

def load_air_temperature() -> pd.DataFrame:
    """Daily max/min air temperature. Adds mean column. Tags incomplete years.

    Handles the older MIDAS layout (pre ~1930) where max and min for the
    same date appear on separate rows (one row has max filled and min NaN,
    the other has min filled and max NaN). We collapse to one row per date
    by taking the first non-NaN value of each column.
    """
    df = pd.read_csv(AIRTEMP_CSV,
                     usecols=['date', 'max_air_temp', 'min_air_temp'],
                     low_memory=False)
    df['date'] = pd.to_datetime(df['date'])

    # Collapse duplicate-date rows: take the first non-NaN value per column.
    # For modern rows (one per date) this is a no-op.
    def first_nonnan(s):
        s = s.dropna()
        return s.iloc[0] if len(s) else np.nan
    df = (df.groupby('date', as_index=False)
            .agg(max_air_temp=('max_air_temp', first_nonnan),
                 min_air_temp=('min_air_temp', first_nonnan)))
    df['year']  = df['date'].dt.year
    df['doy']   = df['date'].dt.dayofyear
    df['tmean'] = (df['max_air_temp'] + df['min_air_temp']) / 2
    return df


def load_ground_temperature() -> dict:
    """Returns a dict of per-depth daily series for Cambridge Botanic Garden
    (1911–2019). Only the BG record is used here: it's the clean, consistent
    century-long series. NIAB ground temperature is at a different depth
    (10 cm vs 30/100 cm), at a different station, and does not overlap in
    time — so combining them would be misleading."""
    df = pd.read_csv(GROUND_CSV, low_memory=False)
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year

    out = {}
    bg = df[df['src_id'] == 454].copy()
    for depth_col, depth_label in [('q30cm_soil_temp',  'BG 30 cm'),
                                    ('q100cm_soil_temp', 'BG 100 cm')]:
        if depth_col in bg.columns and bg[depth_col].notna().any():
            sub = bg[['date', 'year', depth_col]].dropna(subset=[depth_col]).copy()
            sub = sub.rename(columns={depth_col: 'value'})
            out[depth_label] = sub
    return out


def load_rainfall() -> pd.DataFrame:
    """Daily rainfall in mm. Incomplete years are tagged downstream rather
    than dropped here."""
    df = pd.read_csv(RAINFALL_CSV)
    df['date'] = pd.to_datetime(df['DATE'], format='mixed', dayfirst=False)
    df = df[['date', 'PRCP']].rename(columns={'PRCP': 'rain'})
    df['year'] = df['date'].dt.year
    return df


# ===========================================================================
# REGRESSION HELPERS
# ===========================================================================

def fit_ols(x, y, project_to=PROJECT_TO):
    """OLS linear and quadratic; pick by AIC (quad must beat linear by >=2)."""
    valid = ~np.isnan(y)
    xv, yv = x[valid], y[valid]
    x_future = np.arange(int(x.min()), project_to + 1)

    Xl = sm.add_constant(xv)
    m_lin = sm.OLS(yv, Xl).fit()
    Xq = sm.add_constant(np.column_stack([xv, xv**2]))
    m_quad = sm.OLS(yv, Xq).fit()
    use_quad = (m_lin.aic - m_quad.aic) > 2.0
    m = m_quad if use_quad else m_lin

    if use_quad:
        Xf = sm.add_constant(np.column_stack([x_future, x_future**2]))
    else:
        Xf = sm.add_constant(x_future)
    p = m.get_prediction(Xf).summary_frame(alpha=0.05)

    recent = float(xv.max())
    if use_quad:
        decade_change = (m.params[1] + 2 * m.params[2] * recent) * 10
    else:
        decade_change = m.params[1] * 10

    return {
        'family': 'ols',
        'model': 'quadratic' if use_quad else 'linear',
        'aic_linear':    round(float(m_lin.aic), 1),
        'aic_quadratic': round(float(m_quad.aic), 1),
        'x_future': x_future.astype(int).tolist(),
        'mean': [round(float(v), 3) for v in p['mean'].values],
        'lo':   [round(float(v), 3) for v in p['obs_ci_lower'].values],
        'hi':   [round(float(v), 3) for v in p['obs_ci_upper'].values],
        'decade_change': round(float(decade_change), 3),
        'p_slope': round(float(m.pvalues[1]), 6),
    }


def fit_count(x, y, family='negbin', project_to=PROJECT_TO):
    """GLM fit for count data. family='negbin' or 'poisson'.
    Picks linear vs quadratic by AIC, conservative threshold 2.

    Falls back to a flat (constant) fit if the data is too sparse for GLM
    to converge — this happens when nearly all years are zero.
    """
    valid = ~np.isnan(y)
    xv, yv = x[valid], y[valid]
    x_future = np.arange(int(x.min()), project_to + 1)
    x_mean = xv.mean()

    # If too sparse, do a constant fit (mean) and skip GLM entirely.
    nonzero_count = int((yv > 0).sum())
    if len(yv) < 5 or nonzero_count < 3 or yv.sum() == 0:
        mu = float(yv.mean()) if len(yv) else 0.0
        mean_pred = np.full(len(x_future), mu)
        if family == 'negbin':
            # Use mild dispersion for PI even with low signal
            var = max(np.var(yv, ddof=1) if len(yv) > 1 else mu, mu + 0.5)
            alpha = max((var - mu) / max(mu**2, 0.01), 0.5) if mu > 0 else 1.0
            n_param = 1.0 / alpha
            p_param = n_param / (n_param + np.maximum(mean_pred, 0.01))
            lo = nbinom.ppf(0.025, n_param, p_param)
            hi = nbinom.ppf(0.975, n_param, p_param)
        else:
            lo = sp_poisson.ppf(0.025, np.maximum(mean_pred, 0.01))
            hi = sp_poisson.ppf(0.975, np.maximum(mean_pred, 0.01))
        return {
            'family': family,
            'model': 'constant (too sparse for trend)',
            'aic_linear':    None,
            'aic_quadratic': None,
            'x_future': x_future.astype(int).tolist(),
            'mean': [round(float(v), 3) for v in mean_pred],
            'lo':   [round(float(v), 3) for v in lo],
            'hi':   [round(float(v), 3) for v in hi],
            'decade_change': 0.0,
            'p_slope': 1.0,
        }

    Xl = sm.add_constant(xv)
    x_c = xv - x_mean
    Xq = sm.add_constant(np.column_stack([x_c, x_c**2]))

    if family == 'negbin':
        var = np.var(yv, ddof=1); mean = np.mean(yv)
        alpha = max((var - mean) / (mean**2), 0.01) if mean > 0 else 1.0
        fam = sm.families.NegativeBinomial(alpha=alpha)
    else:
        alpha = None
        fam = sm.families.Poisson()

    try:
        m_lin  = sm.GLM(yv, Xl, family=fam).fit()
    except (ValueError, Exception):
        m_lin = None
    try:
        m_quad = sm.GLM(yv, Xq, family=fam).fit()
    except (ValueError, Exception):
        m_quad = None

    # If GLM fails, fall through to Poisson; if THAT fails, constant.
    if m_lin is None and m_quad is None:
        if family != 'poisson':
            return fit_count(x, y, family='poisson', project_to=project_to)
        # truly hopeless — emit constant
        mu = float(yv.mean())
        mean_pred = np.full(len(x_future), mu)
        lo = sp_poisson.ppf(0.025, np.maximum(mean_pred, 0.01))
        hi = sp_poisson.ppf(0.975, np.maximum(mean_pred, 0.01))
        return {
            'family': family, 'model': 'constant (fit failed)',
            'aic_linear': None, 'aic_quadratic': None,
            'x_future': x_future.astype(int).tolist(),
            'mean': [round(float(v), 3) for v in mean_pred],
            'lo':   [round(float(v), 3) for v in lo],
            'hi':   [round(float(v), 3) for v in hi],
            'decade_change': 0.0, 'p_slope': 1.0,
        }
    if m_lin is None:
        use_quad = True; m = m_quad
    elif m_quad is None:
        use_quad = False; m = m_lin
    else:
        use_quad = (m_lin.aic - m_quad.aic) > 2.0
        m = m_quad if use_quad else m_lin

    if use_quad:
        x_c_f = x_future - x_mean
        Xf = sm.add_constant(np.column_stack([x_c_f, x_c_f**2]))
    else:
        Xf = sm.add_constant(x_future)
    mean_pred = m.predict(Xf)

    if family == 'negbin':
        n_param = 1.0 / alpha
        p_param = n_param / (n_param + np.maximum(mean_pred, 0.001))
        lo = nbinom.ppf(0.025, n_param, p_param)
        hi = nbinom.ppf(0.975, n_param, p_param)
    else:
        lo = sp_poisson.ppf(0.025, np.maximum(mean_pred, 0.001))
        hi = sp_poisson.ppf(0.975, np.maximum(mean_pred, 0.001))

    recent = float(xv.max())
    if use_quad:
        x_c_now = recent - x_mean
        mu_now = np.exp(m.params[0] + m.params[1]*x_c_now + m.params[2]*x_c_now**2)
        x_c_10 = (recent + 10) - x_mean
        mu_10  = np.exp(m.params[0] + m.params[1]*x_c_10 + m.params[2]*x_c_10**2)
    else:
        mu_now = np.exp(m.params[0] + m.params[1] * recent)
        mu_10  = np.exp(m.params[0] + m.params[1] * (recent + 10))
    decade_change = float(mu_10 - mu_now)

    return {
        'family': family,
        'model': 'quadratic' if use_quad else 'linear',
        'aic_linear':    round(float(m_lin.aic), 1) if m_lin else None,
        'aic_quadratic': round(float(m_quad.aic), 1) if m_quad else None,
        'x_future': x_future.astype(int).tolist(),
        'mean': [round(float(v), 3) for v in mean_pred],
        'lo':   [round(float(v), 3) for v in lo],
        'hi':   [round(float(v), 3) for v in hi],
        'decade_change': round(decade_change, 3),
        'p_slope': round(float(m.pvalues[1]), 6),
    }


def basic_stats(years, y, baseline_start=BASELINE_START,
                baseline_end=BASELINE_END, dp=1, valid_mask=None):
    """Compute summary stats. If valid_mask is provided, use it to restrict
    which years count for baseline, rolling, and high/low — incomplete years
    are excluded from these summaries."""
    y = np.asarray(y, dtype=float)
    years = np.asarray(years)
    if valid_mask is None:
        valid_mask = np.ones(len(y), dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    # NaN out invalid for the summary computations
    y_valid = np.array(y, dtype=float, copy=True)
    y_valid[~valid_mask] = np.nan

    in_baseline = (years >= baseline_start) & (years <= baseline_end) & valid_mask
    baseline_y = y[in_baseline]
    baseline_mean = (round(float(np.nanmean(baseline_y)), dp)
                     if len(baseline_y) and (~np.isnan(baseline_y)).any() else 0)

    # 5-year trailing mean: rolling mean over the 5-year window, requiring at
    # least 1 valid year (incomplete years are NaN'd out in y_valid above).
    # This way, gaps in coverage don't kill the rolling line.
    s = pd.Series(y_valid)
    rolling = s.rolling(window=5, min_periods=1).mean()
    # But we should only START the rolling line once we have at least one
    # year of data — to avoid a leading run of NaN-from-no-data-yet looking
    # different from a NaN-from-incomplete-data later. (rolling already
    # handles that since the first 4 entries with min_periods=1 are means of
    # 1, 2, 3, 4 entries respectively.)

    if (~np.isnan(y_valid)).any():
        high_idx = int(np.nanargmax(y_valid))
        low_idx  = int(np.nanargmin(y_valid))
        high = {'year': int(years[high_idx]),
                'val': round(float(y_valid[high_idx]), dp), 'label': 'Highest'}
        low  = {'year': int(years[low_idx]),
                'val': round(float(y_valid[low_idx]), dp),  'label': 'Lowest'}
    else:
        high = low = {'year': 0, 'val': 0, 'label': 'n/a'}

    return {
        'mean_val': round(float(np.nanmean(y_valid)), dp),
        'baseline_mean': baseline_mean,
        'high': high, 'low': low,
        'rolling5': [None if np.isnan(v) else round(float(v), dp)
                     for v in rolling.values],
    }


def fit_with_mask(fitter, years_int, y, valid_mask):
    """Fit a model on only the valid years. Wraps fit_ols / fit_count so that
    incomplete years are excluded from the regression but still appear in the
    chart."""
    valid_mask = np.asarray(valid_mask, dtype=bool)
    yv = np.array(y, dtype=float, copy=True)  # writable copy
    yv[~valid_mask] = np.nan  # so fitter's own ~isnan filter takes them out
    x = np.array(years_int, dtype=float, copy=True)
    return fitter(x, yv)


# ===========================================================================
# SECTION 1 — annual overview (temperature + rainfall + ground temp)
# ===========================================================================

def build_annual_overview(air, rain, ground):
    # Count daily obs per year — using max OR min available (we want a year to
    # be "well-observed" if it has enough days of either variable).
    air['has_any'] = air['max_air_temp'].notna() | air['min_air_temp'].notna()
    air_counts  = air.groupby('year')['has_any'].sum()
    rain_counts = rain.groupby('year').size()

    # Annual means: compute the mean of tmax and tmin INDEPENDENTLY (each
    # using all its available days), then derive tmean = (mean_max + mean_min)/2.
    # This makes tmean robust to days where one of max/min is missing — a
    # year with 360 days of max and 350 days of min still gives a meaningful
    # annual mean, whereas computing (max+min)/2 *per day* would NaN out any
    # day where either was missing.
    annual_air = (air.groupby('year')
                     .agg(tmax=('max_air_temp','mean'),
                          tmin=('min_air_temp','mean'),
                          n_max=('max_air_temp', 'count'),
                          n_min=('min_air_temp', 'count'))
                     .reset_index())
    annual_air['tmean'] = (annual_air['tmax'] + annual_air['tmin']) / 2
    # Per-metric "incomplete" flags:
    #   tmax incomplete  if n_max < threshold
    #   tmin incomplete  if n_min < threshold
    #   tmean incomplete if EITHER side is incomplete (need both well-sampled
    #                    for the annual mean of (max+min)/2 to be reliable)
    annual_air['incomplete_tmax']  = annual_air['n_max'] < MIN_OBS_PER_YEAR
    annual_air['incomplete_tmin']  = annual_air['n_min'] < MIN_OBS_PER_YEAR
    annual_air['incomplete_tmean'] = (annual_air['incomplete_tmax']
                                      | annual_air['incomplete_tmin'])

    annual_rain = (rain.groupby('year')
                       .agg(total=('rain','sum'))
                       .reset_index())
    annual_rain['n_obs'] = annual_rain['year'].map(rain_counts).fillna(0).astype(int)
    annual_rain['incomplete'] = annual_rain['n_obs'] < MIN_OBS_PER_YEAR

    metrics = {}

    # Air temperature metrics — each uses its own incomplete flag and obs count
    for key, label, col, n_col, inc_col, desc in [
        ('air_tmean', 'Air: annual mean', 'tmean', None, 'incomplete_tmean',
         'Annual mean temperature, computed as (annual mean T<sub>max</sub> + '
         'annual mean T<sub>min</sub>) / 2. A year is treated as incomplete '
         'for this metric if either T<sub>max</sub> or T<sub>min</sub> has '
         'too few daily observations — both sides need to be well-distributed '
         'across the year for the annual mean to be reliable.'),
        ('air_tmax', 'Air: avg daily max', 'tmax', 'n_max', 'incomplete_tmax',
         'Annual mean of the daily maximum temperature.'),
        ('air_tmin', 'Air: avg daily min', 'tmin', 'n_min', 'incomplete_tmin',
         'Annual mean of the daily minimum temperature.'),
    ]:
        vals = annual_air[col].astype(float).values
        yrs  = annual_air['year'].values
        incomplete = annual_air[inc_col].values
        if n_col is not None:
            n_obs = annual_air[n_col].astype(int).values
        else:
            # For tmean, report min(n_max, n_min) as effective obs count
            n_obs = annual_air[['n_max','n_min']].min(axis=1).astype(int).values
        valid_mask = ~incomplete
        metrics[key] = {
            'label': label, 'description': desc,
            'unit': '°C', 'decimals': 1,
            'color_bar': '#d97559', 'color_bar_edge': '#9c4528',
            'group': 'Air temperature',
            'years': yrs.astype(int).tolist(),
            'values': [round(float(v), 2) for v in vals],
            'incomplete': [bool(x) for x in incomplete],
            'n_obs': [int(x) for x in n_obs],
            'min_obs': MIN_OBS_PER_YEAR,
            'fit': fit_with_mask(fit_ols, yrs, vals, valid_mask),
            **basic_stats(yrs, vals, dp=2, valid_mask=valid_mask),
        }

    # Annual rainfall total
    yrs = annual_rain['year'].values
    vals = annual_rain['total'].astype(float).values
    incomplete = annual_rain['incomplete'].values
    n_obs = annual_rain['n_obs'].values
    valid_mask = ~incomplete
    metrics['rain_total'] = {
        'label': 'Rainfall: annual total',
        'description': 'Sum of daily rainfall totals across the year (mm). '
                       'Years with too few daily observations are shown in '
                       'grey and excluded from the trend fit.',
        'unit': 'mm', 'decimals': 0,
        'color_bar': '#3a7ca5', 'color_bar_edge': '#1f5879',
        'group': 'Rainfall',
        'years': yrs.astype(int).tolist(),
        'values': [round(float(v), 1) for v in vals],
        'incomplete': [bool(x) for x in incomplete],
        'n_obs': [int(x) for x in n_obs],
        'min_obs': MIN_OBS_PER_YEAR,
        'fit': fit_with_mask(fit_ols, yrs, vals, valid_mask),
        **basic_stats(yrs, vals, dp=0, valid_mask=valid_mask),
    }

    # Seasonal rainfall (meteorological seasons: DJF/MAM/JJA/SON).
    # December is assigned to the FOLLOWING year's winter — a Dec 2020 day
    # counts toward winter 2021 (Dec 2020 + Jan 2021 + Feb 2021).
    rs = rain.copy()
    rs['month'] = rs['date'].dt.month
    def season_year_row(month, year):
        if month == 12:    return ('Winter', year + 1)
        if month <= 2:     return ('Winter', year)
        if month <= 5:     return ('Spring', year)
        if month <= 8:     return ('Summer', year)
        return ('Autumn', year)
    seas = rs.apply(lambda r: season_year_row(r['month'], r['year']),
                    axis=1, result_type='expand')
    rs['season'] = seas[0]
    rs['season_year'] = seas[1]
    seasonal = (rs.groupby(['season', 'season_year'])
                  .agg(total=('rain', 'sum'),
                       n_obs=('rain', 'count'))
                  .reset_index())
    # A season needs ~90 days of data; use the same proportion as for the
    # annual threshold (340/365 ≈ 0.93)  → ~84 days minimum out of 90-92.
    SEASON_MIN_OBS = int(MIN_OBS_PER_YEAR * 90 / 365)

    seasonal_defs = [
        ('rain_winter', 'Rainfall: winter (DJF)',
         '#2c5d8c', '#143955',
         'Total rainfall during the meteorological winter (December of the '
         'previous calendar year through February). Cambridge\'s wettest '
         'season has trended upward by ~6 mm/decade.'),
        ('rain_spring', 'Rainfall: spring (MAM)',
         '#5b9b3f', '#2e5a1f',
         'Total rainfall during March, April and May. Year-to-year '
         'variability dominates this season — no clear long-term trend.'),
        ('rain_summer', 'Rainfall: summer (JJA)',
         '#e09f3e', '#8e5e1e',
         'Total rainfall during June, July and August. Despite climate '
         'expectations of drier UK summers, the historical Cambridge trend '
         'is essentially flat.'),
        ('rain_autumn', 'Rainfall: autumn (SON)',
         '#9e4e2b', '#5c2a14',
         'Total rainfall during September, October and November. Cambridge\'s '
         'wettest individual months tend to fall in this season.'),
    ]
    for key, label, color, edge, desc in seasonal_defs:
        season_name = label.split('(')[1].rstrip(')')
        # Map back from the visible season name to the value column
        season_label = {'DJF': 'Winter', 'MAM': 'Spring',
                        'JJA': 'Summer', 'SON': 'Autumn'}[season_name]
        sub = (seasonal[seasonal['season'] == season_label]
                       .sort_values('season_year')
                       .copy())
        # Mark seasons as incomplete if fewer than ~84 days of data
        sub['incomplete'] = sub['n_obs'] < SEASON_MIN_OBS
        yrs_s    = sub['season_year'].values
        vals_s   = sub['total'].astype(float).values
        inc_s    = sub['incomplete'].values
        nobs_s   = sub['n_obs'].astype(int).values
        mask_s   = ~inc_s
        metrics[key] = {
            'label': label,
            'description': desc,
            'unit': 'mm', 'decimals': 0,
            'color_bar': color, 'color_bar_edge': edge,
            'group': 'Rainfall',
            'years': yrs_s.astype(int).tolist(),
            'values': [round(float(v), 1) for v in vals_s],
            'incomplete': [bool(x) for x in inc_s],
            'n_obs': [int(x) for x in nobs_s],
            'min_obs': SEASON_MIN_OBS,
            'fit': fit_with_mask(fit_ols, yrs_s, vals_s, mask_s),
            **basic_stats(yrs_s, vals_s, dp=0, valid_mask=mask_s),
        }

    # Ground temperature (BG only) — apply the same coverage threshold
    for label, df_g in ground.items():
        annual_g = (df_g.groupby('year')
                       .agg(value=('value','mean'),
                            n_obs=('value','count'))
                       .reset_index())
        annual_g['incomplete'] = annual_g['n_obs'] < MIN_OBS_PER_YEAR
        yrs = annual_g['year'].values
        vals = annual_g['value'].astype(float).values
        incomplete = annual_g['incomplete'].values
        n_obs = annual_g['n_obs'].values
        valid_mask = ~incomplete
        key = 'ground_' + label.lower().replace(' ', '').replace('cm', 'cm')
        colors = ('#7e1e9c', '#4a1058') if '30' in label else ('#5e3c8e', '#3d2462')
        metrics[key] = {
            'label': f'Ground: {label}',
            'description': (f'Annual mean soil temperature at Cambridge '
                            f'Botanic Garden, {label.replace("BG ", "")} depth '
                            f'(1911–2019). Years with sparse daily coverage '
                            f'are shown in grey. Deeper soil is heavily '
                            f'damped: 100 cm has a much smaller seasonal '
                            f'swing than 30 cm and lags behind it by weeks.'),
            'unit': '°C', 'decimals': 1,
            'color_bar': colors[0], 'color_bar_edge': colors[1],
            'group': 'Ground temperature',
            'years': yrs.astype(int).tolist(),
            'values': [round(float(v), 2) for v in vals],
            'incomplete': [bool(x) for x in incomplete],
            'n_obs': [int(x) for x in n_obs],
            'min_obs': MIN_OBS_PER_YEAR,
            'fit': fit_with_mask(fit_ols, yrs, vals, valid_mask),
            **basic_stats(yrs, vals, dp=2, valid_mask=valid_mask),
        }

    groups = {
        'Air temperature':    ['air_tmean', 'air_tmax', 'air_tmin'],
        'Ground temperature': [k for k in metrics if k.startswith('ground_')],
        'Rainfall':           ['rain_total', 'rain_winter', 'rain_spring',
                                'rain_summer', 'rain_autumn'],
    }
    return {'metrics': metrics, 'groups': groups}


# ===========================================================================
# SECTION 2 — extreme rainfall (from daily rain)
# ===========================================================================

def longest_run(b: pd.Series) -> int:
    if b.empty: return 0
    g = (b != b.shift()).cumsum()
    runs = b.groupby(g).sum()
    return int(runs.max()) if len(runs) else 0


def build_extreme_rainfall(rain: pd.DataFrame):
    df = rain.copy()
    df['is_dry'] = df['rain'] < 1.0
    df['is_wet'] = df['rain'] >= 1.0
    df['is_10']  = df['rain'] >= 10
    df['is_20']  = df['rain'] >= 20
    df['is_25']  = df['rain'] >= 25
    p95 = df.loc[df['is_wet'], 'rain'].quantile(0.95)
    df['is_very_wet'] = df['rain'] >= p95
    df['very_wet_amount'] = df['rain'].where(df['is_very_wet'], 0)

    agg = df.groupby('year').agg(
        total=('rain', 'sum'),
        max_day=('rain', 'max'),
        days_1mm=('is_wet', 'sum'),
        days_10mm=('is_10', 'sum'),
        days_20mm=('is_20', 'sum'),
        days_25mm=('is_25', 'sum'),
        very_wet_days=('is_very_wet', 'sum'),
        very_wet_total=('very_wet_amount', 'sum'),
        n_obs=('rain', 'count'),
    ).reset_index()
    agg['dry_spell'] = df.groupby('year')['is_dry'].apply(longest_run).values
    agg['wet_spell'] = df.groupby('year')['is_wet'].apply(longest_run).values
    agg['very_wet_fraction'] = (agg['very_wet_total'] / agg['total'] * 100).round(2)
    agg['incomplete'] = agg['n_obs'] < MIN_OBS_PER_YEAR

    # --- Dry / drought-style extremes ---
    # Longest run of consecutive "marginal" weeks (<2 mm in the week).
    dfw = df.copy()
    dfw['week'] = ((dfw['date'].dt.dayofyear - 1) // 7 + 1).clip(upper=52)
    weekly = dfw.groupby(['year', 'week'])['rain'].sum().reset_index()
    weekly['marginal'] = weekly['rain'] < 2.0
    marginal_run = (weekly.sort_values(['year', 'week'])
                          .groupby('year')['marginal']
                          .apply(longest_run))
    agg['marginal_week_run'] = agg['year'].map(marginal_run).fillna(0).astype(int)

    # Count of "dry months" (<20 mm in the calendar month).
    dfm = df.copy()
    dfm['month'] = dfm['date'].dt.month
    monthly = dfm.groupby(['year', 'month'])['rain'].sum().reset_index()
    monthly['dry'] = monthly['rain'] < 20.0
    dry_months = monthly.groupby('year')['dry'].sum()
    agg['dry_months'] = agg['year'].map(dry_months).fillna(0).astype(int)

    yrs = agg['year'].values
    incomplete = agg['incomplete'].values
    n_obs = agg['n_obs'].values
    valid_mask = ~incomplete

    metric_defs = [
        ('days_10mm', 'Heavy rain days (≥10 mm)',
         'Days per year with ≥10 mm rainfall — the Met Office\'s common '
         '"heavy" threshold.', 'days', '#3a7ca5', 0),
        ('days_20mm', 'Very heavy days (≥20 mm)',
         'Days with ≥20 mm rainfall.', 'days', '#27538a', 0),
        ('days_25mm', 'Extreme days (≥25 mm)',
         'Days with ≥25 mm rainfall — about a fifth of an average month '
         'in one day.', 'days', '#1a3c66', 0),
        ('max_day', 'Wettest day of the year',
         'Single largest daily total each year.', 'mm', '#5e3c8e', 1),
        ('very_wet_fraction', f'Share from very wet days (≥{p95:.1f} mm)',
         f'% of annual rain from days with ≥{p95:.1f} mm (the 95th '
         f'percentile of wet days). Climate science predicts this should '
         f'rise with warming.', '%', '#9a3c5e', 1),
        ('dry_spell', 'Longest dry spell',
         'Longest consecutive run of days with <1 mm rain.',
         'days', '#d97559', 0),
        ('marginal_week_run', 'Longest marginal-rain stretch',
         'Longest run of consecutive weeks each receiving under 2 mm of rain '
         '— a proxy for prolonged near-drought conditions. The flip side of '
         'the "heavier bursts" pattern would be longer such stretches, but '
         'the historical trend here is essentially flat.',
         'weeks', '#cc8b3c', 0),
        ('dry_months', 'Dry months (<20 mm)',
         'Number of calendar months per year receiving under 20 mm of rain '
         '(less than half the ~46 mm monthly average). A drought indicator; '
         'rare and without a clear long-term trend in this record.',
         'months', '#b5762e', 0),
        ('wet_spell', 'Longest wet spell',
         'Longest consecutive run of days with ≥1 mm rain.',
         'days', '#4a6fa5', 0),
        ('days_1mm', 'Rainy days (≥1 mm)',
         'Total wet days per year.', 'days', '#3a7ca5', 0),
        ('total', 'Annual rainfall total',
         'Total mm summed over the year.', 'mm', '#3a7ca5', 0),
    ]

    metrics = {}
    for key, label, desc, unit, color, dp in metric_defs:
        vals = agg[key].astype(float).values
        # Use Negative Binomial for the count-style dry metrics (bounded ≥ 0),
        # OLS for everything continuous, matching the rest of the dashboard.
        if key in ('marginal_week_run', 'dry_months'):
            fitter = lambda x, y: fit_count(x, y, family='negbin')
            fit = fit_with_mask(fitter, yrs, vals, valid_mask)
        else:
            fit = fit_with_mask(fit_ols, yrs, vals, valid_mask)
        metrics[key] = {
            'label': label, 'description': desc,
            'unit': unit, 'decimals': dp,
            'color_bar': color, 'color_bar_edge': '#222',
            'group': 'Rainfall extremes',
            'years': yrs.astype(int).tolist(),
            'values': [round(float(v), dp) for v in vals],
            'incomplete': [bool(x) for x in incomplete],
            'n_obs': [int(x) for x in n_obs],
            'min_obs': MIN_OBS_PER_YEAR,
            'fit': fit,
            **basic_stats(yrs, vals, dp=dp, valid_mask=valid_mask),
        }
    return {'metrics': metrics,
            'groups': {'Rainfall extremes': [k for k,*_ in metric_defs]}}


# ===========================================================================
# SECTION 3 — extreme temperature (from daily air temp)
# ===========================================================================

def count_heatwave_days(b: pd.Series, min_run=3) -> int:
    if b.empty: return 0
    g = (b != b.shift()).cumsum()
    s = b.groupby(g).agg(['sum', 'size'])
    w = s[(s['sum'] == s['size']) & (s['sum'] >= min_run)]
    return int(w['sum'].sum())


def build_ground_extreme_temperature(ground: dict):
    """Extreme-soil-temperature metrics per BG depth (30 cm and 100 cm).

    Metrics chosen for relevance:
      - cold soil days (≤2 °C) — proxy for unworkable/cold soil, important
        for early-season planting and the start of biological activity
      - true ground frost days (≤0 °C) — rare at 30 cm, essentially never at
        100 cm, but a strong indicator of past harsh winters
      - hot soil days (≥18 °C, ≥20 °C) — soil heat stress on crop roots
      - growing-season length (days where the soil temp ≥6 °C — the common
        UK agronomic threshold for cool-season grass growth)
      - annual max and min soil temp
    """
    all_metrics = {}
    groups = {'30 cm depth': [], '100 cm depth': []}

    for label, df_g in ground.items():
        depth = '30 cm' if '30' in label else '100 cm'
        prefix = depth.replace(' ', '')
        df = df_g.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df = df.sort_values('date').reset_index(drop=True)

        df['cold2']  = df['value'] <= 2.0
        df['frost']  = df['value'] <= 0.0
        df['hot18']  = df['value'] >= 18.0
        df['hot20']  = df['value'] >= 20.0
        df['warm6']  = df['value'] >= 6.0   # growing-season threshold

        agg = df.groupby('year').agg(
            cold_days=('cold2', 'sum'),
            frost_days=('frost', 'sum'),
            hot_18_days=('hot18', 'sum'),
            hot_20_days=('hot20', 'sum'),
            growing_days=('warm6', 'sum'),
            max_temp=('value', 'max'),
            min_temp=('value', 'min'),
            n_obs=('value', 'count'),
        ).reset_index()
        agg['incomplete'] = agg['n_obs'] < MIN_OBS_PER_YEAR

        yrs = agg['year'].values
        valid_mask = ~agg['incomplete'].values
        n_obs = agg['n_obs'].astype(int).values
        incomplete = agg['incomplete'].values

        # Choose colors / families per metric
        if depth == '30 cm':
            cold_color, hot_color = '#5b9bd5', '#d97559'
        else:
            cold_color, hot_color = '#4a6fa5', '#a93226'

        depth_defs = [
            ('cold_days',    f'Cold soil days (≤2 °C) at {depth}',
             f'Days per year where the {depth} soil temperature is at or '
             f'below 2 °C — near-freezing, biologically inactive soil.',
             'days', cold_color, 0, 'negbin', 'cold'),
            ('frost_days',   f'Soil frost days (≤0 °C) at {depth}',
             f'Days per year where the {depth} soil temperature reaches '
             f'freezing. Rare even at 30 cm; essentially never at 100 cm.',
             'days', cold_color, 0, 'poisson', 'cold'),
            ('hot_18_days',  f'Warm soil days (≥18 °C) at {depth}',
             f'Days per year where the {depth} soil reaches 18 °C or above.',
             'days', hot_color, 0, 'negbin', 'hot'),
            ('hot_20_days',  f'Hot soil days (≥20 °C) at {depth}',
             f'Days per year where the {depth} soil reaches 20 °C or above. '
             f'Very rare at 100 cm.',
             'days', hot_color, 0, 'negbin', 'hot'),
            ('growing_days', f'Growing days (≥6 °C) at {depth}',
             f'Days per year where the {depth} soil is warm enough '
             f'(≥6 °C) for cool-season grass and many crops to grow.',
             'days', '#5b9b3f', 0, 'negbin', 'growing'),
            ('max_temp',     f'Hottest soil day at {depth}',
             f'Annual maximum {depth} soil temperature.',
             '°C', hot_color, 1, 'ols', 'hot'),
            ('min_temp',     f'Coldest soil day at {depth}',
             f'Annual minimum {depth} soil temperature.',
             '°C', cold_color, 1, 'ols', 'cold'),
        ]

        for short_key, label, desc, unit, color, dp, family, kind in depth_defs:
            key = f'ground_{prefix}_{short_key}'
            vals = agg[short_key].astype(float).values
            if family in ('negbin', 'poisson'):
                fitter = lambda x, y, fam=family: fit_count(x, y, family=fam)
                fit = fit_with_mask(fitter, yrs, vals, valid_mask)
            else:
                fit = fit_with_mask(fit_ols, yrs, vals, valid_mask)
            all_metrics[key] = {
                'label': label, 'description': desc,
                'unit': unit, 'decimals': dp,
                'color_bar': color, 'color_bar_edge': '#222',
                'group': depth,
                'years': yrs.astype(int).tolist(),
                'values': [None if np.isnan(v) else round(float(v), dp)
                           for v in vals],
                'incomplete': [bool(x) for x in incomplete],
                'n_obs': [int(x) for x in n_obs],
                'min_obs': MIN_OBS_PER_YEAR,
                'fit': fit,
                **basic_stats(yrs, vals, dp=dp, valid_mask=valid_mask),
            }
            groups[f'{depth} depth'].append(key)

    return {'metrics': all_metrics, 'groups': groups}


def build_extreme_temperature(air: pd.DataFrame):
    df = air.copy()
    df['hot_25'] = df['max_air_temp'] >= 25
    df['hot_28'] = df['max_air_temp'] >= 28
    df['hot_30'] = df['max_air_temp'] >= 30
    df['frost']      = df['min_air_temp'] <= 0
    df['icing']      = df['max_air_temp'] <= 0
    df['hard_frost'] = df['min_air_temp'] <= -5
    df['tropical_night'] = df['min_air_temp'] >= 20
    df = df.sort_values('date').reset_index(drop=True)

    agg = df.groupby('year').agg(
        hot_25_days=('hot_25', 'sum'),
        hot_28_days=('hot_28', 'sum'),
        hot_30_days=('hot_30', 'sum'),
        frost_days=('frost', 'sum'),
        icing_days=('icing', 'sum'),
        hard_frost_days=('hard_frost', 'sum'),
        tropical_nights=('tropical_night', 'sum'),
        tmax_record=('max_air_temp', 'max'),
        tmin_record=('min_air_temp', 'min'),
        n_obs=('max_air_temp', 'count'),
    ).reset_index()
    agg['incomplete'] = agg['n_obs'] < MIN_OBS_PER_YEAR

    def per_year(grp):
        # "Last spring frost": latest frost day between Jan 1 and June 30.
        # If no frost occurred in that window, this year has no spring frost
        # to report.
        spring = grp[grp['date'].dt.month <= 6]
        spring_frosts = spring.loc[spring['frost'], 'date']
        last_spring_frost = (spring_frosts.dt.dayofyear.max()
                             if not spring_frosts.empty else np.nan)
        return pd.Series({
            'heatwave_25_days': count_heatwave_days(grp['hot_25'], 3),
            'longest_hot_streak': longest_run(grp['hot_25']),
            'last_frost': last_spring_frost,
        })
    streaks = df.groupby('year').apply(per_year, include_groups=False).reset_index()
    agg = agg.merge(streaks, on='year')

    family_by_key = {
        'hot_25_days':'negbin', 'hot_28_days':'negbin', 'hot_30_days':'negbin',
        'heatwave_25_days':'negbin', 'longest_hot_streak':'negbin',
        'frost_days':'negbin', 'hard_frost_days':'negbin', 'icing_days':'negbin',
        'tropical_nights':'poisson',
        'tmax_record':'ols', 'tmin_record':'ols', 'last_frost':'ols',
    }

    metric_defs = [
        ('hot_25_days', 'Warm days (≥25 °C)',
         'Days where the daytime maximum reached at least 25 °C.',
         'days', '#e09f3e', 0, 'heat'),
        ('hot_28_days', 'Hot days (≥28 °C)',
         'Days where the daytime maximum reached at least 28 °C.',
         'days', '#d97559', 0, 'heat'),
        ('hot_30_days', 'Very hot days (≥30 °C)',
         'Days where the daytime maximum reached at least 30 °C.',
         'days', '#c1292e', 0, 'heat'),
        ('heatwave_25_days', 'Days in a heatwave',
         'Days within a streak of 3+ consecutive days with max ≥ 25 °C.',
         'days', '#a93226', 0, 'heat'),
        ('longest_hot_streak', 'Longest hot streak',
         'Longest consecutive run of days with max ≥ 25 °C.',
         'days', '#7a1518', 0, 'heat'),
        ('tmax_record', 'Hottest day of the year',
         'Single highest daily maximum each year.',
         '°C', '#a01818', 1, 'heat'),
        ('tropical_nights', 'Tropical nights (≥20 °C min)',
         'Nights where the minimum stayed ≥20 °C — extremely rare in the UK.',
         'nights', '#7e1e9c', 0, 'heat'),
        ('frost_days', 'Frost days (min ≤0 °C)',
         'Nights where the air temperature reached freezing.',
         'days', '#5b9bd5', 0, 'cold'),
        ('hard_frost_days', 'Hard frost days (min ≤−5 °C)',
         'Nights where the air temperature reached −5 °C or below.',
         'days', '#2c5d8c', 0, 'cold'),
        ('icing_days', 'Icing days (max ≤0 °C)',
         'Days where even the daytime maximum stayed at or below freezing.',
         'days', '#1f3b66', 0, 'cold'),
        ('tmin_record', 'Coldest night of the year',
         'Single lowest daily minimum each year.',
         '°C', '#1f3b66', 1, 'cold'),
        ('last_frost', 'Last spring frost',
         'Day-of-year of the latest frost (min ≤0 °C) occurring in the first '
         'half of the year (Jan–Jun). Earlier dates indicate warmer springs. '
         'Years with no spring frost at all show no bar.',
         'day of year', '#5b9bd5', 0, 'cold'),
    ]

    yrs = agg['year'].values
    incomplete = agg['incomplete'].values
    n_obs = agg['n_obs'].values
    valid_mask = ~incomplete
    metrics = {}
    for key, label, desc, unit, color, dp, group_kind in metric_defs:
        vals = agg[key].astype(float).values
        family = family_by_key[key]
        if family in ('negbin', 'poisson'):
            fitter = lambda x, y, fam=family: fit_count(x, y, family=fam)
            fit = fit_with_mask(fitter, yrs, vals, valid_mask)
        else:
            fit = fit_with_mask(fit_ols, yrs, vals, valid_mask)
        metrics[key] = {
            'label': label, 'description': desc,
            'unit': unit, 'decimals': dp,
            'color_bar': color, 'color_bar_edge': '#222',
            'group': 'Heat' if group_kind == 'heat' else 'Cold',
            'years': yrs.astype(int).tolist(),
            'values': [None if np.isnan(v) else round(float(v), dp) for v in vals],
            'incomplete': [bool(x) for x in incomplete],
            'n_obs': [int(x) for x in n_obs],
            'min_obs': MIN_OBS_PER_YEAR,
            'fit': fit,
            **basic_stats(yrs, vals, dp=dp, valid_mask=valid_mask),
        }
    return {'metrics': metrics,
            'groups': {'Heat':  [k for k,*_,g in metric_defs if g=='heat'],
                       'Cold':  [k for k,*_,g in metric_defs if g=='cold']}}


# ===========================================================================
# SECTION 4 — day-of-year lines (daily / weekly / pentad)
# ===========================================================================

def build_doy_plot(air: pd.DataFrame):
    air = air.copy()
    # Only use years with sufficient daily coverage — incomplete years would
    # produce partial lines that mislead in this view.
    counts = air.groupby('year').size()
    drop = set(counts[counts < MIN_OBS_PER_YEAR].index)
    air = air[~air['year'].isin(drop)].copy()
    air['week'] = ((air['doy'] - 1) // 7 + 1).clip(upper=52)
    years = sorted(air['year'].unique().tolist())

    # Daily per-year — ship a few smoothing variants so the user can toggle
    # in-browser without re-loading. We pre-compute raw, 7-day, and 14-day
    # centered rolling means.
    def smooth(arr, w):
        if w <= 1: return arr
        return pd.Series(arr).rolling(window=w, center=True,
                                       min_periods=1).mean().tolist()

    daily_data = {}
    for yr in years:
        g = air[air['year'] == yr].sort_values('doy')
        raw_max = [None if pd.isna(v) else round(float(v), 1)
                   for v in g['max_air_temp']]
        raw_min = [None if pd.isna(v) else round(float(v), 1)
                   for v in g['min_air_temp']]
        # tmean per day = (tmax + tmin) / 2 ; NaN if either side missing
        raw_mean = [
            None if (mx is None or mn is None) else round((mx + mn) / 2, 1)
            for mx, mn in zip(raw_max, raw_min)
        ]
        daily_data[int(yr)] = {
            'doy':       g['doy'].astype(int).tolist(),
            'tmax':      raw_max,
            'tmin':      raw_min,
            'tmean':     raw_mean,
            # smoothed copies — preserve NaNs by skipping them in rolling
            'tmax_s7':  [None if v is None else round(float(v), 2)
                         for v in smooth(raw_max, 7)],
            'tmin_s7':  [None if v is None else round(float(v), 2)
                         for v in smooth(raw_min, 7)],
            'tmean_s7': [None if v is None else round(float(v), 2)
                         for v in smooth(raw_mean, 7)],
            'tmax_s14': [None if v is None else round(float(v), 2)
                         for v in smooth(raw_max, 14)],
            'tmin_s14': [None if v is None else round(float(v), 2)
                         for v in smooth(raw_min, 14)],
            'tmean_s14':[None if v is None else round(float(v), 2)
                         for v in smooth(raw_mean, 14)],
        }
    # Weekly per-year
    air['tmean_daily'] = (air['max_air_temp'] + air['min_air_temp']) / 2
    weekly = (air.groupby(['year', 'week'])
                 .agg(tmax=('max_air_temp', 'mean'),
                      tmin=('min_air_temp', 'mean'),
                      tmean=('tmean_daily', 'mean'))
                 .reset_index())
    weekly_data = {}
    for yr in years:
        g = weekly[weekly['year'] == yr].sort_values('week')
        weekly_data[int(yr)] = {
            'week':  g['week'].astype(int).tolist(),
            'tmax':  [None if pd.isna(v) else round(float(v), 2) for v in g['tmax']],
            'tmin':  [None if pd.isna(v) else round(float(v), 2) for v in g['tmin']],
            'tmean': [None if pd.isna(v) else round(float(v), 2) for v in g['tmean']],
        }
    # Pentad daily climatology
    air['pentad_start'] = (air['year'] // 5) * 5
    pentad_starts = sorted(air['pentad_start'].unique().tolist())
    pentad_data = {}
    for ps in pentad_starts:
        g = (air[air['pentad_start'] == ps]
                .groupby('doy')
                .agg(tmax=('max_air_temp', 'mean'),
                     tmin=('min_air_temp', 'mean'),
                     tmean=('tmean_daily', 'mean'))
                .reset_index())
        g['tmax_s']  = g['tmax'].rolling(7, center=True, min_periods=1).mean()
        g['tmin_s']  = g['tmin'].rolling(7, center=True, min_periods=1).mean()
        g['tmean_s'] = g['tmean'].rolling(7, center=True, min_periods=1).mean()
        yrs_in = sorted(air[air['pentad_start'] == ps]['year'].unique().tolist())
        label = (f"{ps}–{yrs_in[-1]}" if len(yrs_in) < 5 else f"{ps}–{ps+4}")
        pentad_data[int(ps)] = {
            'doy':    g['doy'].astype(int).tolist(),
            'tmax':   [round(float(v), 2) for v in g['tmax_s']],
            'tmin':   [round(float(v), 2) for v in g['tmin_s']],
            'tmean':  [round(float(v), 2) for v in g['tmean_s']],
            'label':  label,
        }
    # Baseline climatologies
    base = air[(air['year'] >= BASELINE_START) & (air['year'] <= BASELINE_END)]
    cd_max  = base.groupby('doy')['max_air_temp'].agg(['mean','std']).reset_index()
    cd_min  = base.groupby('doy')['min_air_temp'].agg(['mean','std']).reset_index()
    cd_mean = base.groupby('doy')['tmean_daily'].agg(['mean','std']).reset_index()
    for d in (cd_max, cd_min, cd_mean):
        d['mean_s'] = d['mean'].rolling(7, center=True, min_periods=1).mean()
        d['std_s']  = d['std'].rolling(7, center=True, min_periods=1).mean()
    bw = base.groupby('week').agg(
        tmax_mean=('max_air_temp','mean'),  tmax_std=('max_air_temp','std'),
        tmin_mean=('min_air_temp','mean'),  tmin_std=('min_air_temp','std'),
        tmean_mean=('tmean_daily','mean'),  tmean_std=('tmean_daily','std'),
    ).reset_index()

    clim = {
        'daily': {
            'doy':        cd_max['doy'].astype(int).tolist(),
            'tmax_mean':  [round(float(v), 2) for v in cd_max['mean_s']],
            'tmax_lo':    [round(float(m-s), 2) for m,s in
                           zip(cd_max['mean_s'], cd_max['std_s'])],
            'tmax_hi':    [round(float(m+s), 2) for m,s in
                           zip(cd_max['mean_s'], cd_max['std_s'])],
            'tmin_mean':  [round(float(v), 2) for v in cd_min['mean_s']],
            'tmin_lo':    [round(float(m-s), 2) for m,s in
                           zip(cd_min['mean_s'], cd_min['std_s'])],
            'tmin_hi':    [round(float(m+s), 2) for m,s in
                           zip(cd_min['mean_s'], cd_min['std_s'])],
            'tmean_mean': [round(float(v), 2) for v in cd_mean['mean_s']],
            'tmean_lo':   [round(float(m-s), 2) for m,s in
                           zip(cd_mean['mean_s'], cd_mean['std_s'])],
            'tmean_hi':   [round(float(m+s), 2) for m,s in
                           zip(cd_mean['mean_s'], cd_mean['std_s'])],
        },
        'weekly': {
            'week':       bw['week'].astype(int).tolist(),
            'tmax_mean':  [round(float(v), 2) for v in bw['tmax_mean']],
            'tmax_lo':    [round(float(m-s), 2) for m,s in
                           zip(bw['tmax_mean'], bw['tmax_std'])],
            'tmax_hi':    [round(float(m+s), 2) for m,s in
                           zip(bw['tmax_mean'], bw['tmax_std'])],
            'tmin_mean':  [round(float(v), 2) for v in bw['tmin_mean']],
            'tmin_lo':    [round(float(m-s), 2) for m,s in
                           zip(bw['tmin_mean'], bw['tmin_std'])],
            'tmin_hi':    [round(float(m+s), 2) for m,s in
                           zip(bw['tmin_mean'], bw['tmin_std'])],
            'tmean_mean': [round(float(v), 2) for v in bw['tmean_mean']],
            'tmean_lo':   [round(float(m-s), 2) for m,s in
                           zip(bw['tmean_mean'], bw['tmean_std'])],
            'tmean_hi':   [round(float(m+s), 2) for m,s in
                           zip(bw['tmean_mean'], bw['tmean_std'])],
        },
    }
    return {'years': years, 'pentad_starts': pentad_starts,
            'daily': daily_data, 'weekly': weekly_data,
            'pentad': pentad_data, 'clim': clim,
            'baseline_start': BASELINE_START,
            'baseline_end': BASELINE_END}


# ===========================================================================
# SECTION 5 — heatmaps: weeks (rows) × years (cols), one cell per (week, year)
# ===========================================================================

def build_heatmaps(air: pd.DataFrame, rain: pd.DataFrame, ground: dict):
    air = air.copy()
    rain = rain.copy()
    air['week']  = ((air['doy'] - 1) // 7 + 1).clip(upper=52)
    rain['week'] = ((rain['date'].dt.dayofyear - 1) // 7 + 1).clip(upper=52)

    # Drop incomplete years from heatmaps too (otherwise sparse years
    # produce blank-ish columns with mostly NaN cells)
    air_counts  = air.groupby('year').size()
    rain_counts = rain.groupby('year').size()
    air  = air[~air['year'].isin(set(air_counts[air_counts < MIN_OBS_PER_YEAR].index))]
    rain = rain[~rain['year'].isin(set(rain_counts[rain_counts < MIN_OBS_PER_YEAR].index))]

    # Aggregate to (year, week) cells
    air_agg = (air.groupby(['year', 'week'])
                  .agg(tmax=('max_air_temp', 'mean'),
                       tmin=('min_air_temp', 'mean'),
                       tmean=('tmean', 'mean'))
                  .reset_index())
    rain_agg = (rain.groupby(['year', 'week'])
                    .agg(rain=('rain', 'sum'))
                    .reset_index())

    def pivot(df, value_col, years):
        full = df.pivot(index='week', columns='year', values=value_col)
        full = full.reindex(index=range(1, 53))
        full = full.reindex(columns=years)
        return full

    air_years  = sorted(air['year'].unique().tolist())
    rain_years = sorted(rain['year'].unique().tolist())

    def to_matrix(pivoted):
        z = []
        for row in pivoted.values:
            z.append([None if pd.isna(v) else round(float(v), 2) for v in row])
        return z

    out = {
        'air': {
            'years': air_years,
            'weeks': list(range(1, 53)),
            'tmean': to_matrix(pivot(air_agg, 'tmean', air_years)),
            'tmax':  to_matrix(pivot(air_agg, 'tmax',  air_years)),
            'tmin':  to_matrix(pivot(air_agg, 'tmin',  air_years)),
        },
        'rain': {
            'years': rain_years,
            'weeks': list(range(1, 53)),
            'rain': to_matrix(pivot(rain_agg, 'rain', rain_years)),
        }
    }

    # Ground temperature: one section per depth, separate heatmaps
    for label, df_g in ground.items():
        depth_key = '30cm' if '30' in label else '100cm'
        df = df_g.copy()
        df['date'] = pd.to_datetime(df['date'])
        df['year'] = df['date'].dt.year
        df['doy']  = df['date'].dt.dayofyear
        df['week'] = ((df['doy'] - 1) // 7 + 1).clip(upper=52)
        # Drop incomplete years
        cnt = df.groupby('year').size()
        df = df[~df['year'].isin(set(cnt[cnt < MIN_OBS_PER_YEAR].index))]
        g_agg = (df.groupby(['year', 'week'])
                   .agg(value=('value', 'mean'))
                   .reset_index())
        g_years = sorted(df['year'].unique().tolist())
        out[f'ground_{depth_key}'] = {
            'years': g_years,
            'weeks': list(range(1, 53)),
            'value': to_matrix(pivot(g_agg, 'value', g_years)),
            'label': f'Ground temperature ({label.replace("BG ", "")})',
        }

    return out


# ===========================================================================
# HTML
# ===========================================================================

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Cambridge Climate Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  :root { --primary: #16425b; --accent: #c1292e; }
  * { box-sizing: border-box; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Helvetica, Arial, sans-serif;
    margin: 0; padding: 0; background: #fafafa; color: #222;
  }
  .container { max-width: 1240px; margin: 0 auto; padding: 20px; }
  h1 { font-size: 1.5rem; margin: 0 0 4px; }
  .subtitle { color: #666; font-size: 0.95rem; margin-bottom: 16px; }
  /* Top-level tabs */
  .tabs {
    display: flex; gap: 2px; margin-bottom: 16px;
    border-bottom: 2px solid #ddd; flex-wrap: wrap;
  }
  .tab {
    background: transparent; border: none;
    padding: 10px 18px; font-size: 0.98rem; color: #666;
    cursor: pointer; border-bottom: 3px solid transparent;
    margin-bottom: -2px; transition: all 0.15s;
  }
  .tab:hover { color: var(--primary); }
  .tab.active {
    color: var(--primary); border-bottom-color: var(--primary);
    font-weight: 600;
  }
  .panel { display: none; }
  .panel.active { display: block; }
  .group-label {
    font-size: 0.76rem; color: #888; margin: 10px 0 4px;
    text-transform: uppercase; letter-spacing: 0.5px;
  }
  .toggle-row { display: flex; gap: 6px; align-items: center; flex-wrap: wrap; }
  .controls {
    display: flex; gap: 8px; align-items: center; flex-wrap: wrap;
    margin-bottom: 8px;
  }
  .controls .label { font-size: 0.85rem; color: #666; margin-right: 6px; }
  .btn {
    background: white; border: 1.5px solid #ddd;
    padding: 6px 12px; border-radius: 20px; cursor: pointer;
    font-size: 0.85rem; color: #444; transition: all 0.15s;
  }
  .btn:hover { background: #f0f0f0; border-color: #bbb; }
  .btn.active {
    background: var(--primary); color: white; border-color: var(--primary);
    font-weight: 500;
  }
  .metric-desc {
    font-size: 0.85rem; color: #555; padding: 10px 14px;
    background: white; border-radius: 6px; margin: 12px 0;
    border-left: 3px solid var(--primary);
  }
  .model-badge {
    display: inline-block; background: #eef3f6; color: var(--primary);
    padding: 2px 8px; border-radius: 10px; font-size: 0.75rem;
    margin-left: 8px; font-weight: 500;
  }
  .chart-block {
    background: white; border-radius: 8px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    padding: 8px; margin-bottom: 18px;
  }
  .chart-title {
    font-size: 1rem; font-weight: 600; color: var(--primary);
    margin: 0 0 6px 4px;
  }
  .stats { display: flex; gap: 12px; margin-top: 14px; flex-wrap: wrap; }
  .stat-card {
    background: white; padding: 10px 14px; border-radius: 6px;
    box-shadow: 0 1px 2px rgba(0,0,0,0.06); font-size: 0.88rem;
    min-width: 100px;
  }
  .stat-card .k { color: #888; font-size: 0.76rem; }
  .stat-card .v { margin-top: 2px; }
  .note {
    font-size: 0.82rem; color: #777; margin-top: 14px; line-height: 1.45;
    background: #fff; padding: 14px; border-radius: 6px;
    border-left: 3px solid #ccc;
  }
  .sig { color: #2a7a2a; font-weight: 500; }
  .verysig { color: #1e5e1e; font-weight: 600; }
  .nonsig { color: #999; }
  .range-row {
    display: flex; gap: 10px; align-items: center;
    font-size: 0.85rem; color: #666;
  }
  .range-row input[type=range] { accent-color: var(--primary); }
  .range-row label {
    display: inline-flex; align-items: center; gap: 6px;
  }
  .pin-status { font-size: 0.85rem; color: var(--accent); font-weight: 500; }
</style>
</head>
<body>
<div class="container">
  <h1>Cambridge Climate Dashboard</h1>
  <div class="subtitle">
    Met Office MIDAS (air &amp; ground temperature, Cambridge Botanic Garden / NIAB)
    and NOAA GHCN-D (rainfall, Cambridge NIAB). Reference period:
    1960–1990 (WMO standard climate normal).
  </div>

  <div class="tabs">
    <button class="tab active" data-tab="overview">Annual overview</button>
    <button class="tab" data-tab="temp">Temperature extremes</button>
    <button class="tab" data-tab="ground">Ground extremes</button>
    <button class="tab" data-tab="rain">Rainfall extremes</button>
    <button class="tab" data-tab="doy">Day-of-year lines</button>
    <button class="tab" data-tab="heatmap">Heatmaps</button>
    <button class="tab" data-tab="spei">Drought (SPEI)</button>
    <button class="tab" data-tab="lookup">Date lookup</button>
  </div>

  <!-- ===================== OVERVIEW ===================== -->
  <div class="panel active" id="panel-overview">
    <div id="overview-buttons"></div>
    <div class="metric-desc" id="overview-desc"></div>
    <div class="chart-block">
      <div id="chart-overview" style="width:100%;height:560px;"></div>
    </div>
    <div class="stats" id="stats-overview"></div>
    <div class="note">
      Annual aggregates for each metric. Linear and quadratic trends are
      compared by AIC; the quadratic is preferred only if it improves AIC by
      ≥ 2. The shaded grey band on the year axis marks the 1960–1990
      baseline period.
      <br><br>
      <b>Incomplete years.</b> Years with fewer than 350 daily observations
      are shown in striped grey rather than the normal colour. Their values
      are still drawn (so gaps in the record are visible) but they are
      excluded from the trend fit, the baseline mean, the 5-year rolling
      mean, and the highest/lowest stats. Hover any bar to see how many
      daily observations contributed to it.
      <br><br>
      <b>Ground temperature:</b> Cambridge Botanic Garden 30 cm and 100 cm
      depths (1911–2019). These are two separate series, not a combined
      record; they're shown side-by-side because different soil depths
      damp the seasonal cycle differently.
    </div>
  </div>

  <!-- ===================== RAIN ===================== -->
  <div class="panel" id="panel-rain">
    <div id="rain-buttons"></div>
    <div class="metric-desc" id="rain-desc"></div>
    <div class="chart-block">
      <div id="chart-rain" style="width:100%;height:560px;"></div>
    </div>
    <div class="stats" id="stats-rain"></div>
    <div class="note">
      Rainfall indices computed from daily NOAA GHCN-D values.
      Years with fewer than 350 daily observations are shown in striped grey
      and excluded from the trend fit.
    </div>
  </div>

  <!-- ===================== TEMP EXTREMES ===================== -->
  <div class="panel" id="panel-temp">
    <div class="group-label">Heat metrics</div>
    <div class="toggle-row" id="temp-hot-buttons"></div>
    <div class="group-label">Cold metrics</div>
    <div class="toggle-row" id="temp-cold-buttons"></div>
    <div class="metric-desc" id="temp-desc"></div>
    <div class="chart-block">
      <div id="chart-temp" style="width:100%;height:560px;"></div>
    </div>
    <div class="stats" id="stats-temp"></div>
    <div class="note">
      Counts (frost days, hot days, etc.) are modelled with Negative Binomial
      regression: a log link keeps projections non-negative, and overdispersion
      (variance much greater than mean here) is handled honestly.
      Years with fewer than 340 daily observations are shown in striped grey
      and excluded from the trend fit.
    </div>
  </div>

  <!-- ===================== GROUND EXTREMES ===================== -->
  <div class="panel" id="panel-ground">
    <div id="ground-buttons"></div>
    <div class="metric-desc" id="ground-desc"></div>
    <div class="chart-block">
      <div id="chart-ground" style="width:100%;height:560px;"></div>
    </div>
    <div class="stats" id="stats-ground"></div>
    <div class="note">
      Soil temperature extremes from Cambridge Botanic Garden, 1911–2019, at
      two depths. Shallow soil (30 cm) responds quickly to air temperature
      and frosts; deep soil (100 cm) is heavily damped and is the deepest
      indicator of long-term climate signals. Many metrics are very rare —
      e.g. true freezing at 30 cm only happened in extreme winters.
    </div>
  </div>

  <!-- ===================== DOY LINES ===================== -->
  <div class="panel" id="panel-doy">
    <div class="controls">
      <span class="label">Resolution:</span>
      <button class="btn active" data-res="daily">Daily</button>
      <button class="btn" data-res="weekly">Weekly average</button>
      <button class="btn" data-res="pentad">5-year averages</button>
      <span class="label" style="margin-left:18px;">Smoothing (daily only):</span>
      <button class="btn" data-smooth="0">None</button>
      <button class="btn active" data-smooth="7">7-day</button>
      <button class="btn" data-smooth="14">14-day</button>
    </div>
    <div class="controls">
      <span class="label">Show:</span>
      <button class="btn active" data-doy-view="all">All three</button>
      <button class="btn" data-doy-view="tmean">Daily mean only</button>
      <button class="btn" data-doy-view="tmax">Daily max only</button>
      <button class="btn" data-doy-view="tmin">Daily min only</button>
      <span class="label" style="margin-left:18px;">Lines:</span>
      <button class="btn active" data-lines="all">All</button>
      <button class="btn" data-lines="early">1960–1990</button>
      <button class="btn" data-lines="recent">2010+</button>
      <button class="btn" data-lines="extreme">Extreme years</button>
    </div>
    <div class="controls range-row">
      <label>Line opacity:
        <input type="range" id="opacity" min="0.1" max="1" step="0.05" value="0.35">
        <span id="op-val">0.35</span>
      </label>
      <span style="margin-left:18px;" class="pin-status" id="pin-status"></span>
      <button class="btn" id="unpin-btn" style="display:none;">Unpin</button>
    </div>
    <div class="controls" style="background:#f4f6f8;padding:10px 12px;border-radius:6px;">
      <span class="label">Compare two periods:</span>
      <label style="font-size:0.85rem">
        Period A:
        <input type="number" id="period-a-start" min="1911" max="2024" value="1960" style="width:60px">
        –
        <input type="number" id="period-a-end" min="1911" max="2024" value="1990" style="width:60px">
      </label>
      <label style="font-size:0.85rem;margin-left:8px">
        Period B:
        <input type="number" id="period-b-start" min="1911" max="2024" value="2000" style="width:60px">
        –
        <input type="number" id="period-b-end" min="1911" max="2024" value="2023" style="width:60px">
      </label>
      <button class="btn" id="compare-toggle">Show comparison</button>
    </div>

    <div class="chart-block" id="block-doy-tmean">
      <div class="chart-title">Daily mean temperature (°C)</div>
      <div id="chart-doy-tmean" style="width:100%;height:480px;"></div>
    </div>
    <div class="chart-block" id="block-doy-tmax">
      <div class="chart-title">Daily maximum temperature (°C)</div>
      <div id="chart-doy-tmax" style="width:100%;height:480px;"></div>
    </div>
    <div class="chart-block" id="block-doy-tmin">
      <div class="chart-title">Daily minimum temperature (°C)</div>
      <div id="chart-doy-tmin" style="width:100%;height:480px;"></div>
    </div>
    <div class="note">
      One line per year (or per 5-year window in pentad mode), coloured from
      blue (older) to red (recent). Click any line to pin it. Black line +
      grey envelope = 1960–1990 mean ±1σ for each day/week.
    </div>
  </div>

  <!-- ===================== HEATMAPS ===================== -->
  <div class="panel" id="panel-heatmap">
    <div class="controls">
      <span class="label">Variable:</span>
      <button class="btn active" data-hm="tmean">Air: mean weekly</button>
      <button class="btn" data-hm="tmax">Air: weekly max</button>
      <button class="btn" data-hm="tmin">Air: weekly min</button>
      <button class="btn" data-hm="rain">Rainfall</button>
      <button class="btn" data-hm="ground_30cm">Ground: 30 cm</button>
      <button class="btn" data-hm="ground_100cm">Ground: 100 cm</button>
    </div>
    <div class="controls">
      <span class="label">Colour scale:</span>
      <button class="btn active" data-hm-scale="absolute">Absolute</button>
      <button class="btn" data-hm-scale="anomaly">Anomaly vs. 1960–1990</button>
    </div>
    <div class="chart-block">
      <div class="chart-title" id="heatmap-title">Mean weekly air temperature (°C)</div>
      <div id="chart-heatmap" style="width:100%;height:560px;"></div>
    </div>
    <div class="note">
      Each cell is one (week, year) value: rows are weeks of the year (week 1
      at the top = early January, week 52 at the bottom = late December),
      columns are years left → right. <b>Absolute</b> uses the raw value; the
      colour bar covers the full data range. <b>Anomaly</b> subtracts the
      1960–1990 mean for that week, so warmer-than-baseline cells are red
      and cooler-than-baseline cells are blue — climate change shows up
      visually as a leftward-blue / rightward-red gradient.
      Incomplete years (&lt; 340 daily observations) are excluded.
    </div>
  </div>

  <!-- ===================== SPEI ===================== -->
  <div class="panel" id="panel-spei">
    <div class="controls">
      <span class="label">Timescale:</span>
      <button class="btn active" data-spei="spei_12">SPEI-12 (annual / hydrological)</button>
      <button class="btn" data-spei="spei_3">SPEI-3 (seasonal)</button>
      <span class="label" style="margin-left:18px;">View:</span>
      <button class="btn active" data-spei-view="monthly">Monthly</button>
      <button class="btn" data-spei-view="annual">Annual mean</button>
    </div>
    <div class="metric-desc" id="spei-desc"></div>
    <div class="chart-block">
      <div id="chart-spei" style="width:100%;height:520px;"></div>
    </div>
    <div class="note">
      <b>SPEI — the Standardised Precipitation-Evapotranspiration Index —</b>
      measures drought by comparing the climatic water balance
      (precipitation minus potential evapotranspiration, "PET") against the
      historical distribution for that location and time of year. It is
      expressed like a z-score: 0 is normal, negative is drier than usual,
      positive is wetter. Unlike a precipitation-only drought index, SPEI
      folds in the <b>evaporative demand from warming</b>, so a hotter year
      registers as drier even with the same rainfall.
      <br><br>
      PET is computed with the Hargreaves method from daily max/min air
      temperature. The index is fitted to a Pearson Type III distribution,
      calibrated over the WMO 1961–1990 normal period. <b>SPEI-12</b> reflects
      slow-building hydrological drought (reservoirs, groundwater);
      <b>SPEI-3</b> reflects shorter seasonal dry spells (soil moisture,
      crops). Bars are coloured on the standard drought scale: browns for
      dry, blue-greens for wet. Cambridge's well-known drought years —
      1921, 1976, 2011, 2022 — all show up as strongly negative.
      <br><br>
      <b>Why it matters here:</b> the simple precipitation-only dry metrics
      (in the Rainfall extremes tab) showed flat trends, but SPEI also
      captures rising evaporative demand. Comparing the two tells you whether
      Cambridge's drought risk is shifting because of warming even where
      rainfall alone looks unchanged.
    </div>
  </div>

  <!-- ===================== DATE LOOKUP ===================== -->
  <div class="panel" id="panel-lookup">
    <div class="controls" style="margin-bottom:18px;">
      <span class="label">Date:</span>
      <input type="date" id="lookup-date"
             style="font-size:1rem;padding:6px 10px;
                    border:1.5px solid #ddd;border-radius:6px;">
      <span class="label" style="margin-left:12px;color:#aaa;font-size:0.8rem;"
            id="lookup-range"></span>
    </div>
    <div id="lookup-result" style="min-height:200px;"></div>
    <div class="note">
      Type any date within the covered range and see the daily readings for
      that day, along with how the day compared to its day-of-year average
      from the 1960–1990 baseline period. Missing values are marked "n/a".
    </div>
  </div>
</div>

<script>
const BUNDLE = __DATA__;

// =====================================================================
// Helpers shared by tabs
// =====================================================================
function formatVal(v, m) {
  if (v == null) return 'n/a';
  if (m.unit === 'day of year') {
    const d = new Date(2001, 0, 1);
    d.setDate(d.getDate() + Math.round(v) - 1);
    const mo = ['Jan','Feb','Mar','Apr','May','Jun',
                'Jul','Aug','Sep','Oct','Nov','Dec'];
    return mo[d.getMonth()] + ' ' + d.getDate();
  }
  if (m.unit === '°C')  return v.toFixed(m.decimals) + ' °C';
  if (m.unit === '%')   return v.toFixed(m.decimals) + ' %';
  return v.toFixed(m.decimals) + ' ' + m.unit;
}
function familyName(f) {
  if (f === 'negbin')   return 'Negative Binomial';
  if (f === 'poisson')  return 'Poisson';
  return 'Gaussian';
}
function buildBarTraces(m, isDoy) {
  const f = m.fit;
  const dp = m.decimals;
  const splitIdx = f.x_future.indexOf(m.years[m.years.length - 1]);
  const histX = f.x_future.slice(0, splitIdx + 1);
  const futX  = f.x_future.slice(splitIdx);
  const fitH  = f.mean.slice(0, splitIdx + 1);
  const fitF  = f.mean.slice(splitIdx);
  const loF   = f.lo.slice(splitIdx);
  const hiF   = f.hi.slice(splitIdx);

  // Split observed series into complete vs. incomplete for separate-trace
  // colouring. Use null/NaN to "skip" the wrong-bucket bars.
  const incompleteFlags = m.incomplete || m.years.map(_ => false);
  const nObs   = m.n_obs || m.years.map(_ => null);
  const minObs = m.min_obs || 0;
  const completeY  = m.values.map((v, i) =>
    (v == null || incompleteFlags[i]) ? null : v);
  const incompleteY = m.values.map((v, i) =>
    (v == null || !incompleteFlags[i]) ? null : v);
  const hasIncomplete = incompleteFlags.some(Boolean);

  function makeHover(complete) {
    return m.years.map((yr, i) => {
      const v = m.values[i];
      if (v == null) return `<b>${yr}</b><br>${m.label}: no data`;
      if (complete && incompleteFlags[i]) return '';
      if (!complete && !incompleteFlags[i]) return '';
      const anom = v - m.baseline_mean;
      const obsNote = nObs[i] != null
        ? `<br>Daily obs: ${nObs[i]}` +
          (incompleteFlags[i]
            ? ` <span style="color:#a23a3a">(below ${minObs} threshold — excluded from fit)</span>`
            : '')
        : '';
      return `<b>${yr}</b><br>` +
             `${m.label}: <b>${formatVal(v, m)}</b><br>` +
             `vs. 1960–1990 baseline: ${anom >= 0 ? '+' : ''}` +
             `${anom.toFixed(dp)} ${isDoy ? 'days' : m.unit}` + obsNote;
    });
  }

  const rollHover = m.years.map((yr, i) => {
    const v = m.rolling5[i];
    if (v == null) return '';
    return `<b>${yr-4}–${yr}</b><br>` +
           `5-year mean: <b>${formatVal(v, m)}</b>`;
  });
  const modelLabel = `${familyName(f.family)} ${f.model}`;

  const traces = [
    { x: m.years, y: completeY, type: 'bar', name: 'Observed',
      marker: { color: m.color_bar, opacity: 0.75,
                line: { color: m.color_bar_edge, width: 0.3 } },
      hovertext: makeHover(true), hoverinfo: 'text',
      hoverlabel: { bgcolor: '#fff', bordercolor: m.color_bar,
                    font: { size: 13 } } }
  ];
  if (hasIncomplete) {
    traces.push({
      x: m.years, y: incompleteY, type: 'bar',
      name: `Incomplete year (< ${minObs} obs)`,
      marker: { color: '#bbbbbb', opacity: 0.65,
                line: { color: '#666', width: 0.4 },
                pattern: { shape: '/', fgcolor: '#666',
                           bgcolor: '#dddddd', size: 4 } },
      hovertext: makeHover(false), hoverinfo: 'text',
      hoverlabel: { bgcolor: '#fff', bordercolor: '#888',
                    font: { size: 13 } }
    });
  }
  traces.push(
    { x: m.years, y: m.rolling5, type: 'scatter', mode: 'lines',
      name: '5-year trailing mean',
      line: { color: '#222', width: 2.5 },
      hovertext: rollHover, hoverinfo: 'text',
      hoverlabel: { bgcolor: '#fff', bordercolor: '#222' },
      connectgaps: false },
    { x: [...futX, ...futX.slice().reverse()],
      y: [...hiF, ...loF.slice().reverse()],
      fill: 'toself', fillcolor: 'rgba(22,66,91,0.13)',
      line: { color: 'transparent' },
      name: '95% prediction interval', hoverinfo: 'skip' },
    { x: histX, y: fitH, type: 'scatter', mode: 'lines',
      name: `${modelLabel} fit`,
      line: { color: '#16425b', width: 2.5, dash: 'dash' },
      hovertemplate: `<b>%{x}</b><br>Fit: <b>%{y:.${dp}f}</b><extra></extra>`,
      hoverlabel: { bgcolor: '#fff', bordercolor: '#16425b' } },
    { x: futX, y: fitF, type: 'scatter', mode: 'lines',
      showlegend: false, line: { color: '#16425b', width: 2.5, dash: 'dash' },
      hovertemplate: `<b>%{x}</b><br>Projection: <b>%{y:.${dp}f}</b>` +
                     `<extra></extra>`,
      hoverlabel: { bgcolor: '#fff', bordercolor: '#16425b' } },
    { x: [f.x_future[f.x_future.length-1]], y: [f.mean[f.mean.length-1]],
      type: 'scatter', mode: 'markers',
      name: f.x_future[f.x_future.length-1] + ' projection',
      marker: { color: '#16425b', size: 12,
                line: { color: 'white', width: 2 } },
      hovertemplate: (() => {
        const projVal = f.mean[f.mean.length-1];
        const anom = projVal - m.baseline_mean;
        const anomStr = (anom >= 0 ? '+' : '') + anom.toFixed(dp);
        return `<b>${f.x_future[f.x_future.length-1]} (${modelLabel})</b>` +
          `<br>${m.label}: <b>${projVal.toFixed(dp)}</b>` +
          `<br>95% PI: ${f.lo[f.lo.length-1].toFixed(dp)}–` +
          `${f.hi[f.hi.length-1].toFixed(dp)}` +
          `<br>vs. 1960–1990 baseline: <b>${anomStr} ${isDoy ? 'days' : m.unit}</b>` +
          `<extra></extra>`;
      })(),
      hoverlabel: { bgcolor: '#fff', bordercolor: '#16425b' } }
  );
  return traces;
}
function buildBarLayout(m) {
  const isDoy = m.unit === 'day of year';
  let yaxis = { showgrid: true, gridcolor: '#eee' };
  if (isDoy) {
    yaxis.tickvals = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335];
    yaxis.ticktext = ['Jan','Feb','Mar','Apr','May','Jun',
                       'Jul','Aug','Sep','Oct','Nov','Dec'];
    yaxis.title = m.label;
  } else if (m.unit === '°C') {
    yaxis.title = `${m.label} (°C)`;
  } else {
    yaxis.title = `${m.label} (${m.unit})`;
    yaxis.rangemode = 'tozero';
  }
  const x_end = m.fit.x_future[m.fit.x_future.length-1];
  const x_obs_end = m.years[m.years.length-1];
  return {
    margin: { l: 60, r: 30, t: 20, b: 50 },
    barmode: 'overlay',
    xaxis: { title: 'Year', range: [m.years[0]-1, x_end+2],
             showgrid: true, gridcolor: '#eee' },
    yaxis: yaxis, hovermode: 'closest',
    legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(255,255,255,0.92)',
              bordercolor: '#ddd', borderwidth: 1 },
    shapes: [
      { type: 'line', x0: x_obs_end + 0.5, x1: x_obs_end + 0.5,
        y0: 0, y1: 1, yref: 'paper',
        line: { color: '#888', width: 1, dash: 'dot' } },
      { type: 'line', x0: m.years[0]-1, x1: x_end+2,
        y0: m.baseline_mean, y1: m.baseline_mean,
        line: { color: '#999', width: 1, dash: 'dot' } },
      { type: 'rect', xref: 'x', yref: 'paper',
        x0: 1960, x1: 1990, y0: 0, y1: 1,
        fillcolor: 'rgba(220,220,220,0.35)',
        line: { width: 0 }, layer: 'below' }
    ],
    annotations: [
      { x: x_obs_end + 0.5, y: 1.0, yref: 'paper',
        text: 'extrapolation →', showarrow: false, xanchor: 'left',
        font: { size: 11, color: '#888' }, xshift: 4, yshift: -4 },
      { x: x_end + 2, y: m.baseline_mean,
        text: `baseline (${formatVal(m.baseline_mean, m)})`,
        showarrow: false, xanchor: 'right', yanchor: 'bottom',
        font: { size: 10, color: '#666' } },
      { x: 1975, y: 1.0, yref: 'paper',
        text: 'baseline period',
        showarrow: false, xanchor: 'center', yanchor: 'top',
        font: { size: 10, color: '#888', style: 'italic' },
        yshift: -4 }
    ],
    paper_bgcolor: 'white', plot_bgcolor: 'white'
  };
}
function statsCards(m) {
  const f = m.fit;
  const dp = m.decimals;
  const isDoy = m.unit === 'day of year';
  let sigClass, sigText;
  if (f.p_slope < 0.001) { sigClass = 'verysig'; sigText = 'highly significant'; }
  else if (f.p_slope < 0.05) { sigClass = 'sig'; sigText = 'significant'; }
  else { sigClass = 'nonsig'; sigText = 'not significant'; }
  const r5 = m.rolling5.filter(v => v != null);
  const recentRolling = r5.length ? r5[r5.length - 1] : null;
  const diff = recentRolling != null ? recentRolling - m.baseline_mean : null;
  const modelDesc = `${familyName(f.family)} ${f.model}`;
  const x_end = f.x_future[f.x_future.length-1];
  return [
    ['Model', `<b>${modelDesc}</b>` +
              ` <span style="color:#888;font-size:0.76rem">` +
              `(lin AIC ${f.aic_linear ?? 'n/a'}, quad ${f.aic_quadratic ?? 'n/a'})</span>`],
    ['Records', `${m.years.length} years (${m.years[0]}–${m.years[m.years.length-1]})`],
    ['1960–1990 baseline', `<b>${formatVal(m.baseline_mean, m)}</b>`],
    ['Last 5-yr mean',
     recentRolling != null
       ? `<b>${formatVal(recentRolling, m)}</b> ` +
         `<span style="color:${diff>0?'#a23a3a':'#2a7a2a'}">` +
         `(${diff>0?'+':''}${diff.toFixed(dp)} vs. baseline)</span>`
       : 'n/a'],
    [`${m.high.label} year`,
     `<b>${m.high.year}</b> (${formatVal(m.high.val, m)})`],
    [`${m.low.label} year`,
     `<b>${m.low.year}</b> (${formatVal(m.low.val, m)})`],
    ['Trend at present',
     `<b>${f.decade_change >= 0 ? '+' : ''}${f.decade_change}` +
     ` ${isDoy?'days':m.unit}/dec</b> ` +
     `<span class="${sigClass}">(${sigText}` +
     (f.p_slope<0.001 ? ', p<0.001' : `, p=${f.p_slope.toFixed(3)}`) + ')</span>'],
    [`${x_end} projection`, (() => {
       const v = f.mean[f.mean.length-1];
       const anom = v - m.baseline_mean;
       const color = anom > 0 ? '#a23a3a' : '#2a7a2a';
       return `<b>${formatVal(v, m)}</b> ` +
         `<span style="color:${color}">(${anom>0?'+':''}${anom.toFixed(dp)} vs. baseline)</span>`;
    })()]
  ];
}
function renderCards(targetId, cards) {
  document.getElementById(targetId).innerHTML = cards.map(([k,v]) =>
    `<div class="stat-card"><div class="k">${k}</div>` +
    `<div class="v">${v}</div></div>`).join('');
}

// =====================================================================
// Generic "metric with bar+trend chart" tab renderer
// =====================================================================
function renderBarTab(bundle, chartId, statsId, descId, current) {
  const m = bundle.metrics[current];
  Plotly.react(chartId, buildBarTraces(m, m.unit === 'day of year'),
               buildBarLayout(m),
               { responsive: true, displaylogo: false });
  renderCards(statsId, statsCards(m));
  document.getElementById(descId).innerHTML = m.description +
    `<span class="model-badge">${familyName(m.fit.family)} ${m.fit.model}</span>`;
}

// =====================================================================
// OVERVIEW
// =====================================================================
const overviewBundle = BUNDLE.overview;
let currentOverview = Object.keys(overviewBundle.metrics)[0];
(function() {
  const container = document.getElementById('overview-buttons');
  for (const [group, keys] of Object.entries(overviewBundle.groups)) {
    if (!keys.length) continue;
    const label = document.createElement('div');
    label.className = 'group-label';
    label.textContent = group;
    container.appendChild(label);
    const row = document.createElement('div');
    row.className = 'toggle-row';
    keys.forEach(k => {
      const b = document.createElement('button');
      b.className = 'btn'; b.dataset.metric = k;
      b.textContent = overviewBundle.metrics[k].label;
      b.addEventListener('click', () => {
        currentOverview = k; renderOverview();
      });
      row.appendChild(b);
    });
    container.appendChild(row);
  }
})();
function renderOverview() {
  renderBarTab(overviewBundle, 'chart-overview', 'stats-overview',
               'overview-desc', currentOverview);
  document.querySelectorAll('#overview-buttons .btn').forEach(b =>
    b.classList.toggle('active', b.dataset.metric === currentOverview));
}

// =====================================================================
// RAINFALL EXTREMES
// =====================================================================
const rainBundle = BUNDLE.rainfall;
let currentRain = Object.keys(rainBundle.metrics)[0];
(function() {
  const row = document.createElement('div');
  row.className = 'toggle-row';
  Object.keys(rainBundle.metrics).forEach(k => {
    const b = document.createElement('button');
    b.className = 'btn'; b.dataset.metric = k;
    b.textContent = rainBundle.metrics[k].label;
    b.addEventListener('click', () => { currentRain = k; renderRain(); });
    row.appendChild(b);
  });
  document.getElementById('rain-buttons').appendChild(row);
})();
function renderRain() {
  renderBarTab(rainBundle, 'chart-rain', 'stats-rain',
               'rain-desc', currentRain);
  document.querySelectorAll('#rain-buttons .btn').forEach(b =>
    b.classList.toggle('active', b.dataset.metric === currentRain));
}

// =====================================================================
// TEMP EXTREMES
// =====================================================================
const tempBundle = BUNDLE.temperature;
let currentTemp = tempBundle.groups.Heat[0];
(function() {
  function makeRow(targetId, keys) {
    const row = document.getElementById(targetId);
    keys.forEach(k => {
      const b = document.createElement('button');
      b.className = 'btn'; b.dataset.metric = k;
      b.textContent = tempBundle.metrics[k].label;
      b.addEventListener('click', () => { currentTemp = k; renderTemp(); });
      row.appendChild(b);
    });
  }
  makeRow('temp-hot-buttons', tempBundle.groups.Heat);
  makeRow('temp-cold-buttons', tempBundle.groups.Cold);
})();
function renderTemp() {
  renderBarTab(tempBundle, 'chart-temp', 'stats-temp',
               'temp-desc', currentTemp);
  document.querySelectorAll('#panel-temp .btn[data-metric]').forEach(b =>
    b.classList.toggle('active', b.dataset.metric === currentTemp));
}

// =====================================================================
// GROUND EXTREMES
// =====================================================================
const groundBundle = BUNDLE.ground;
let currentGround = Object.keys(groundBundle.metrics)[0];
(function() {
  const container = document.getElementById('ground-buttons');
  for (const [group, keys] of Object.entries(groundBundle.groups)) {
    if (!keys.length) continue;
    const label = document.createElement('div');
    label.className = 'group-label';
    label.textContent = group;
    container.appendChild(label);
    const row = document.createElement('div');
    row.className = 'toggle-row';
    keys.forEach(k => {
      const b = document.createElement('button');
      b.className = 'btn'; b.dataset.metric = k;
      b.textContent = groundBundle.metrics[k].label
                        .replace(/\s+at\s+\d+\s+cm.*$/, '');
      b.addEventListener('click', () => {
        currentGround = k; renderGround();
      });
      row.appendChild(b);
    });
    container.appendChild(row);
  }
})();
function renderGround() {
  renderBarTab(groundBundle, 'chart-ground', 'stats-ground',
               'ground-desc', currentGround);
  document.querySelectorAll('#ground-buttons .btn').forEach(b =>
    b.classList.toggle('active', b.dataset.metric === currentGround));
}

// =====================================================================
// DOY LINES
// =====================================================================
const doyBundle = BUNDLE.doy;
let doyView = 'all', doyRes = 'daily', doyLines = 'all';
let doyOpacity = 0.35, doyPinned = null, doySmooth = 7;
let doyCompare = false;
let doyCompareA = [1960, 1990], doyCompareB = [2000, 2023];
const EXTREME_YEARS = [1963, 1976, 2003, 2018, 2022];

function lineColor(yr, allKeys) {
  const yMin = allKeys[0], yMax = allKeys[allKeys.length - 1];
  const t = (yr - yMin) / Math.max(yMax - yMin, 1);
  const hue = 240 - t * 230;
  return `hsl(${hue}, 65%, 50%)`;
}
function whichKeys() {
  if (doyRes === 'pentad') return doyBundle.pentad_starts;
  if (doyLines === 'early') return doyBundle.years.filter(y => y >= 1960 && y <= 1990);
  if (doyLines === 'recent') return doyBundle.years.filter(y => y >= 2010);
  if (doyLines === 'extreme') return doyBundle.years.filter(y => EXTREME_YEARS.includes(y));
  return doyBundle.years;
}
function getDoySeries(key, metric) {
  if (doyRes === 'daily') {
    // Apply smoothing if any
    const suffix = doySmooth === 7 ? '_s7'
                 : doySmooth === 14 ? '_s14'
                 : '';
    return { x: doyBundle.daily[key].doy,
             y: doyBundle.daily[key][metric + suffix] };
  }
  if (doyRes === 'weekly') return { x: doyBundle.weekly[key].week, y: doyBundle.weekly[key][metric] };
  return { x: doyBundle.pentad[key].doy, y: doyBundle.pentad[key][metric] };
}
function keyLabel(k) {
  return doyRes === 'pentad' ? doyBundle.pentad[k].label : String(k);
}
function getClim(metric) {
  if (doyRes === 'weekly') {
    return { x: doyBundle.clim.weekly.week,
             mean: doyBundle.clim.weekly[metric + '_mean'],
             lo:   doyBundle.clim.weekly[metric + '_lo'],
             hi:   doyBundle.clim.weekly[metric + '_hi'] };
  }
  return { x: doyBundle.clim.daily.doy,
           mean: doyBundle.clim.daily[metric + '_mean'],
           lo:   doyBundle.clim.daily[metric + '_lo'],
           hi:   doyBundle.clim.daily[metric + '_hi'] };
}
function buildPeriodSummary(metric, yrStart, yrEnd) {
  // For each doy/week in the resolution, average across the years in [yrStart, yrEnd].
  // Returns { x: [...], mean: [...], lo: [...], hi: [...] }  (lo/hi = mean ± SD)
  const isWeekly = doyRes === 'weekly';
  const isDaily  = doyRes === 'daily';
  // Pentad mode doesn't make sense for comparison - fall back to daily climatology
  // We use raw data (no smoothing) here to get accurate per-period means.
  const xkey  = isWeekly ? 'week' : 'doy';
  const xs    = isWeekly ? doyBundle.clim.weekly.week
                          : doyBundle.clim.daily.doy;
  const result = { x: xs, mean: [], lo: [], hi: [] };
  // Accumulate values per x position
  const byX = {};
  xs.forEach(x => byX[x] = []);
  for (let yr = yrStart; yr <= yrEnd; yr++) {
    let src;
    if (isWeekly && doyBundle.weekly[yr]) src = doyBundle.weekly[yr];
    else if (doyBundle.daily[yr])         src = doyBundle.daily[yr];
    else continue;
    const xa = src[xkey], ya = src[metric];
    if (!xa || !ya) continue;
    for (let i = 0; i < xa.length; i++) {
      if (ya[i] == null) continue;
      if (byX[xa[i]] !== undefined) byX[xa[i]].push(ya[i]);
    }
  }
  xs.forEach(x => {
    const arr = byX[x];
    if (!arr.length) { result.mean.push(null); result.lo.push(null); result.hi.push(null); return; }
    const m = arr.reduce((a,b) => a+b, 0) / arr.length;
    let v = 0;
    arr.forEach(a => v += (a - m) ** 2);
    const sd = Math.sqrt(v / Math.max(arr.length - 1, 1));
    result.mean.push(m);
    result.lo.push(m - sd);
    result.hi.push(m + sd);
  });
  // Smooth the mean line for visual clarity (7-day rolling)
  const w = 7;
  const smooth = arr => arr.map((v, i) => {
    if (v == null) return null;
    let s = 0, n = 0;
    for (let k = Math.max(0, i - Math.floor(w/2));
             k <= Math.min(arr.length-1, i + Math.floor(w/2)); k++) {
      if (arr[k] != null) { s += arr[k]; n++; }
    }
    return n ? s / n : null;
  });
  result.mean = smooth(result.mean);
  result.lo = smooth(result.lo);
  result.hi = smooth(result.hi);
  return result;
}

function buildDoyTraces(metric) {
  const allKeys = doyRes === 'pentad' ? doyBundle.pentad_starts : doyBundle.years;

  // Comparison mode: show only the two summary bands, baseline, and nothing else.
  if (doyCompare) {
    const sumA = buildPeriodSummary(metric, doyCompareA[0], doyCompareA[1]);
    const sumB = buildPeriodSummary(metric, doyCompareB[0], doyCompareB[1]);
    const labelA = `${doyCompareA[0]}–${doyCompareA[1]}`;
    const labelB = `${doyCompareB[0]}–${doyCompareB[1]}`;
    // Helper to make a labelled band+line for one period
    const makeBand = (sum, label, colorBand, colorLine) => {
      const validX = [], validHi = [], validLo = [], validMean = [];
      for (let i = 0; i < sum.x.length; i++) {
        if (sum.mean[i] != null) {
          validX.push(sum.x[i]);
          validHi.push(sum.hi[i]); validLo.push(sum.lo[i]);
          validMean.push(sum.mean[i]);
        }
      }
      return [
        { x: [...validX, ...validX.slice().reverse()],
          y: [...validHi, ...validLo.slice().reverse()],
          fill: 'toself', fillcolor: colorBand, line: { color: 'transparent' },
          name: `${label} ±1σ`, hoverinfo: 'skip', showlegend: true,
          customdata: [['__period__']] },
        { type: 'scatter', mode: 'lines',
          name: `${label} mean`,
          x: validX, y: validMean,
          line: { color: colorLine, width: 3 },
          hovertemplate: `<b>${label}</b><br>%{x}: <b>%{y:.1f} °C</b><extra></extra>`,
          hoverlabel: { bgcolor: '#fff', bordercolor: colorLine },
          showlegend: true,
          customdata: [['__period__']] }
      ];
    };
    const tracesA = makeBand(sumA, labelA, 'rgba(91,155,213,0.25)', '#1f4d7a');
    const tracesB = makeBand(sumB, labelB, 'rgba(217,117,89,0.25)', '#a83622');
    return [...tracesA, ...tracesB];
  }

  const keys = whichKeys();
  const traces = [];
  const clim = getClim(metric);
  traces.push({
    x: [...clim.x, ...clim.x.slice().reverse()],
    y: [...clim.hi, ...clim.lo.slice().reverse()],
    fill: 'toself', fillcolor: 'rgba(0,0,0,0.10)',
    line: { color: 'transparent' },
    name: '1960–1990 ±1σ', hoverinfo: 'skip', showlegend: false
  });
  keys.forEach(k => {
    if (k === doyPinned) return;
    const s = getDoySeries(k, metric);
    const color = lineColor(k, allKeys);
    const isExtreme = (doyLines === 'extreme' && doyRes !== 'pentad');
    const isPentad = doyRes === 'pentad';
    const opacity = isExtreme ? 0.9 : (isPentad ? 0.85 : doyOpacity);
    const width = (isExtreme || isPentad) ? 1.8 : 1;
    traces.push({
      type: 'scatter', mode: 'lines',
      name: keyLabel(k),
      x: s.x, y: s.y, line: { color, width },
      opacity,
      hovertemplate: `<b>${keyLabel(k)}</b><br>%{x}: <b>%{y:.1f} °C</b><extra></extra>`,
      hoverlabel: { bgcolor: '#fff', bordercolor: color },
      showlegend: false, connectgaps: false,
      customdata: [[k]]
    });
  });
  traces.push({
    type: 'scatter', mode: 'lines',
    name: '1960–1990 mean',
    x: clim.x, y: clim.mean,
    line: { color: '#000', width: 2.4 },
    hovertemplate: '%{x}<br>1960–1990 mean: <b>%{y:.1f} °C</b><extra></extra>',
    hoverlabel: { bgcolor: '#fff', bordercolor: '#000' },
    showlegend: false, customdata: [['__clim__']]
  });
  if (doyPinned !== null && keys.includes(doyPinned)) {
    const s = getDoySeries(doyPinned, metric);
    const color = lineColor(doyPinned, allKeys);
    traces.push({
      type: 'scatter', mode: 'lines',
      name: keyLabel(doyPinned) + ' (pinned)',
      x: s.x, y: s.y, line: { color, width: 3.5 }, opacity: 1,
      hovertemplate: `<b>${keyLabel(doyPinned)} ★</b><br>%{x}: <b>%{y:.1f} °C</b><extra></extra>`,
      hoverlabel: { bgcolor: '#fff', bordercolor: color,
                    font: { color: '#000' } },
      showlegend: false, customdata: [[doyPinned]]
    });
  }
  if (!(doyLines === 'extreme' && doyRes !== 'pentad')) {
    traces.push({
      type: 'scatter', mode: 'markers',
      x: [null], y: [null],
      marker: {
        color: allKeys,
        colorscale: [
          [0, 'hsl(240,65%,50%)'], [0.25, 'hsl(180,65%,50%)'],
          [0.5, 'hsl(120,65%,50%)'], [0.75, 'hsl(60,65%,50%)'],
          [1, 'hsl(10,65%,50%)']
        ],
        cmin: allKeys[0], cmax: allKeys[allKeys.length-1],
        showscale: true,
        colorbar: {
          title: { text: doyRes==='pentad' ? 'Pentad start' : 'Year',
                   side: 'top' },
          thickness: 12, len: 0.75, x: 1.02,
          // Tickvals: span the actual range, every 20 years (or 10 if pentad)
          tickvals: (() => {
            const lo = allKeys[0], hi = allKeys[allKeys.length-1];
            const step = doyRes === 'pentad' ? 10 : 20;
            // Round lo up to nearest multiple of step
            const first = Math.ceil(lo / step) * step;
            const vals = [];
            for (let v = first; v <= hi; v += step) vals.push(v);
            return vals;
          })()
        }
      },
      showlegend: false, hoverinfo: 'skip'
    });
  }
  return traces;
}
function buildDoyLayout(metric) {
  const allVals = [];
  doyBundle.years.forEach(yr => {
    doyBundle.daily[yr][metric].forEach(v => { if (v != null) allVals.push(v); });
  });
  const yMin = Math.floor(Math.min(...allVals)) - 1;
  const yMax = Math.ceil(Math.max(...allVals)) + 1;
  const isWeekly = doyRes === 'weekly';
  const xConfig = isWeekly
    ? { title: 'Week of year', range: [0.5, 52.5],
        tickvals: [1, 5, 9, 13, 18, 22, 26, 31, 35, 39, 44, 48, 52],
        ticktext: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug',
                   'Sep','Oct','Nov','Dec',''] }
    : { title: 'Day of year', range: [1, 366],
        tickvals: [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366],
        ticktext: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug',
                   'Sep','Oct','Nov','Dec',''] };
  return {
    margin: { l: 60, r: 80, t: 10, b: 60 },
    xaxis: { ...xConfig, showgrid: true, gridcolor: '#eee' },
    yaxis: { title: `${metric === 'tmax' ? 'Daily max' : 'Daily min'} (°C)`,
             showgrid: true, gridcolor: '#eee', range: [yMin, yMax],
             zeroline: true, zerolinecolor: '#bbb', zerolinewidth: 1 },
    hovermode: 'closest',
    paper_bgcolor: 'white', plot_bgcolor: 'white'
  };
}
function renderDoy() {
  const isPentad = doyRes === 'pentad';
  const isDaily = doyRes === 'daily';
  document.querySelectorAll('#panel-doy .btn[data-lines]').forEach(b => {
    b.disabled = isPentad || doyCompare;
    b.style.opacity = (isPentad || doyCompare) ? '0.45' : '';
    b.style.cursor = (isPentad || doyCompare) ? 'not-allowed' : 'pointer';
  });
  // Smoothing buttons only make sense in daily mode
  document.querySelectorAll('#panel-doy .btn[data-smooth]').forEach(b => {
    b.disabled = !isDaily;
    b.style.opacity = isDaily ? '' : '0.45';
    b.style.cursor = isDaily ? 'pointer' : 'not-allowed';
  });
  document.getElementById('opacity').disabled = isPentad || doyCompare;

  // Render the three temperature panels
  ['tmean', 'tmax', 'tmin'].forEach(metric => {
    Plotly.react(`chart-doy-${metric}`,
                 buildDoyTraces(metric),
                 buildDoyLayout(metric),
                 { responsive: true, displaylogo: false });
    const el = document.getElementById(`chart-doy-${metric}`);
    el.on('plotly_click', evt => {
      if (!evt || !evt.points || !evt.points.length) return;
      const key = evt.points[0].data.customdata?.[0]?.[0];
      if (key === undefined || key === '__clim__' || key === '__period__') return;
      doyPinned = (doyPinned === key) ? null : key;
      renderDoy();
    });
  });
  // Show/hide each panel based on doyView
  const showTmean = (doyView === 'all' || doyView === 'tmean');
  const showTmax  = (doyView === 'all' || doyView === 'tmax');
  const showTmin  = (doyView === 'all' || doyView === 'tmin');
  document.getElementById('block-doy-tmean').style.display = showTmean ? '' : 'none';
  document.getElementById('block-doy-tmax').style.display  = showTmax  ? '' : 'none';
  document.getElementById('block-doy-tmin').style.display  = showTmin  ? '' : 'none';

  document.querySelectorAll('#panel-doy .btn[data-doy-view]').forEach(b =>
    b.classList.toggle('active', b.dataset.doyView === doyView));
  document.querySelectorAll('#panel-doy .btn[data-lines]').forEach(b =>
    b.classList.toggle('active', b.dataset.lines === doyLines));
  document.querySelectorAll('#panel-doy .btn[data-res]').forEach(b =>
    b.classList.toggle('active', b.dataset.res === doyRes));
  document.querySelectorAll('#panel-doy .btn[data-smooth]').forEach(b =>
    b.classList.toggle('active', parseInt(b.dataset.smooth) === doySmooth));
  document.getElementById('compare-toggle').classList.toggle('active', doyCompare);
  document.getElementById('compare-toggle').textContent =
    doyCompare ? 'Hide comparison' : 'Show comparison';

  const pinStatus = document.getElementById('pin-status');
  const unpinBtn  = document.getElementById('unpin-btn');
  if (doyPinned !== null && !doyCompare) {
    pinStatus.textContent = `★ Pinned: ${
      doyRes === 'pentad' ? doyBundle.pentad[doyPinned].label : doyPinned}`;
    unpinBtn.style.display = '';
  } else {
    pinStatus.textContent = '';
    unpinBtn.style.display = 'none';
  }
}
document.querySelectorAll('#panel-doy .btn[data-doy-view]').forEach(b =>
  b.addEventListener('click', () => { doyView = b.dataset.doyView; renderDoy(); }));
document.querySelectorAll('#panel-doy .btn[data-lines]').forEach(b =>
  b.addEventListener('click', () => {
    if (doyRes === 'pentad') return;
    doyLines = b.dataset.lines; renderDoy();
  }));
document.querySelectorAll('#panel-doy .btn[data-res]').forEach(b =>
  b.addEventListener('click', () => {
    doyRes = b.dataset.res;
    doyPinned = null;
    renderDoy();
  }));
document.querySelectorAll('#panel-doy .btn[data-smooth]').forEach(b =>
  b.addEventListener('click', () => {
    if (doyRes !== 'daily') return;
    doySmooth = parseInt(b.dataset.smooth);
    renderDoy();
  }));
document.getElementById('unpin-btn').addEventListener('click', () => {
  doyPinned = null; renderDoy();
});
document.getElementById('opacity').addEventListener('input', e => {
  doyOpacity = parseFloat(e.target.value);
  document.getElementById('op-val').textContent = doyOpacity.toFixed(2);
  renderDoy();
});

// Period-comparison handlers
function updateComparePeriods() {
  doyCompareA = [
    parseInt(document.getElementById('period-a-start').value),
    parseInt(document.getElementById('period-a-end').value)
  ];
  doyCompareB = [
    parseInt(document.getElementById('period-b-start').value),
    parseInt(document.getElementById('period-b-end').value)
  ];
}
document.getElementById('compare-toggle').addEventListener('click', () => {
  doyCompare = !doyCompare;
  if (doyCompare) updateComparePeriods();
  renderDoy();
});
['period-a-start','period-a-end','period-b-start','period-b-end'].forEach(id =>
  document.getElementById(id).addEventListener('change', () => {
    updateComparePeriods();
    if (doyCompare) renderDoy();
  }));

// =====================================================================
// HEATMAPS
// =====================================================================
const heatmapBundle = BUNDLE.heatmaps;
let currentHm = 'tmean';
let currentHmScale = 'absolute';

const HM_CONFIG = {
  tmean: { label: 'Mean weekly air temperature (°C)', unit: '°C',
           src: 'air', key: 'tmean', colorscale: 'RdBu_r' },
  tmax:  { label: 'Weekly max air temperature (°C)', unit: '°C',
           src: 'air', key: 'tmax', colorscale: 'RdBu_r' },
  tmin:  { label: 'Weekly min air temperature (°C)', unit: '°C',
           src: 'air', key: 'tmin', colorscale: 'RdBu_r' },
  rain:  { label: 'Weekly rainfall total (mm)', unit: 'mm',
           src: 'rain', key: 'rain', colorscale: 'YlGnBu' },
  ground_30cm:  { label: 'Weekly ground temperature, 30 cm (°C)', unit: '°C',
           src: 'ground_30cm', key: 'value', colorscale: 'RdBu_r' },
  ground_100cm: { label: 'Weekly ground temperature, 100 cm (°C)', unit: '°C',
           src: 'ground_100cm', key: 'value', colorscale: 'RdBu_r' },
};

function computeHmBaseline(srcKey, matKey) {
  // For each of the 52 weeks, compute mean across baseline years 1960-1990
  const src = heatmapBundle[srcKey];
  const yrs = src.years;
  const mat = src[matKey];
  const baselineMeans = [];
  for (let w = 0; w < 52; w++) {
    let sum = 0, n = 0;
    for (let c = 0; c < yrs.length; c++) {
      const yr = yrs[c];
      if (yr < 1960 || yr > 1990) continue;
      const v = mat[w][c];
      if (v == null) continue;
      sum += v; n++;
    }
    baselineMeans.push(n ? sum / n : null);
  }
  return baselineMeans;
}

function buildHeatmapTrace() {
  const cfg = HM_CONFIG[currentHm];
  const src = heatmapBundle[cfg.src];
  const matRaw = src[cfg.key];
  const weeks = src.weeks;
  const years = src.years;

  let z, colorscale, zmid, zmin, zmax, cbarTitle;
  if (currentHmScale === 'anomaly') {
    const baseline = computeHmBaseline(cfg.src, cfg.key);
    z = matRaw.map((row, w) =>
      row.map(v => (v == null || baseline[w] == null)
              ? null : Math.round((v - baseline[w]) * 100) / 100));
    // Symmetric scale around zero
    let mx = 0;
    z.forEach(row => row.forEach(v => {
      if (v != null) mx = Math.max(mx, Math.abs(v));
    }));
    zmin = -mx; zmax = mx; zmid = 0;
    // For rain, use a diverging brown-blue scale instead of red-blue
    colorscale = (cfg.key === 'rain')
      ? [[0, '#8c510a'], [0.5, '#f5f5f5'], [1, '#01665e']]
      : 'RdBu_r';
    cbarTitle = `${cfg.label.replace(/\\s*\\([^)]+\\)/, '')} anomaly (${cfg.unit})`;
  } else {
    z = matRaw;
    colorscale = cfg.colorscale;
    cbarTitle = cfg.label;
    zmin = zmax = zmid = undefined;
  }

  const trace = {
    type: 'heatmap',
    x: years, y: weeks, z: z,
    colorscale,
    colorbar: { title: { text: cbarTitle, side: 'right' }, thickness: 14 },
    hovertemplate: 'Year %{x}, week %{y}<br>' +
                   '<b>%{z:.2f} ' +
                   (currentHmScale === 'anomaly' ? `${cfg.unit} vs. baseline`
                                                  : cfg.unit) +
                   '</b><extra></extra>',
    hoverongaps: false,
  };
  if (zmin !== undefined) {
    trace.zmin = zmin; trace.zmax = zmax; trace.zmid = zmid;
  }
  return [trace];
}

function buildHeatmapLayout() {
  return {
    margin: { l: 80, r: 30, t: 10, b: 50 },
    xaxis: { title: 'Year', showgrid: false, type: 'linear', dtick: 10 },
    yaxis: {
      title: 'Week of year', showgrid: false, autorange: 'reversed',
      // Approximate month labels at the start of each month's week
      tickvals: [1, 5, 9, 13, 18, 22, 26, 31, 35, 39, 44, 48],
      ticktext: ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug',
                 'Sep','Oct','Nov','Dec']
    },
    paper_bgcolor: 'white', plot_bgcolor: 'white',
  };
}

function renderHeatmap() {
  const cfg = HM_CONFIG[currentHm];
  document.getElementById('heatmap-title').textContent =
    cfg.label + (currentHmScale === 'anomaly' ? ' — anomaly vs. 1960–1990' : '');
  Plotly.react('chart-heatmap', buildHeatmapTrace(), buildHeatmapLayout(),
               { responsive: true, displaylogo: false });
  document.querySelectorAll('#panel-heatmap .btn[data-hm]').forEach(b =>
    b.classList.toggle('active', b.dataset.hm === currentHm));
  document.querySelectorAll('#panel-heatmap .btn[data-hm-scale]').forEach(b =>
    b.classList.toggle('active', b.dataset.hmScale === currentHmScale));
}

document.querySelectorAll('#panel-heatmap .btn[data-hm]').forEach(b =>
  b.addEventListener('click', () => { currentHm = b.dataset.hm; renderHeatmap(); }));
document.querySelectorAll('#panel-heatmap .btn[data-hm-scale]').forEach(b =>
  b.addEventListener('click', () => {
    currentHmScale = b.dataset.hmScale;
    renderHeatmap();
  }));

// =====================================================================
// DATE LOOKUP
// =====================================================================
const lookupBundle = BUNDLE.date_lookup;

function formatDateLong(dStr) {
  const [y,m,d] = dStr.split('-').map(Number);
  const dt = new Date(Date.UTC(y, m-1, d));
  const opts = { weekday:'long', year:'numeric', month:'long', day:'numeric' };
  return dt.toLocaleDateString('en-GB', opts);
}
function doyFromDate(dStr) {
  const [y,m,d] = dStr.split('-').map(Number);
  const dt = new Date(Date.UTC(y, m-1, d));
  const start = new Date(Date.UTC(y, 0, 1));
  return Math.floor((dt - start) / 86400000) + 1;
}
function anomColor(diff) {
  if (diff == null) return '#999';
  if (diff > 0) return '#a23a3a';
  if (diff < 0) return '#2a7a2a';
  return '#666';
}
function renderLookup() {
  const dateInput = document.getElementById('lookup-date');
  const dateStr = dateInput.value;
  const out = document.getElementById('lookup-result');
  if (!dateStr) {
    out.innerHTML = '<div style="color:#888;font-size:0.95rem;padding:20px;">' +
      'Pick a date above to see its readings.</div>';
    return;
  }
  const day = lookupBundle.data[dateStr];
  if (!day) {
    out.innerHTML =
      `<div style="color:#a23a3a;font-size:0.95rem;padding:20px;">` +
      `No data available for ${formatDateLong(dateStr)}.</div>`;
    return;
  }
  const doy = doyFromDate(dateStr);
  const baseline = (metric) => lookupBundle.baselines[metric]?.[doy] ?? null;

  function row(label, value, unit, metric, dp=1) {
    if (value == null) {
      return `<tr><td style="padding:6px 12px;color:#888;">${label}</td>` +
             `<td colspan="3" style="padding:6px 12px;color:#aaa;">no reading</td></tr>`;
    }
    const b = baseline(metric);
    const anom = b != null ? value - b : null;
    const anomCell = (anom == null)
      ? '<td colspan="2" style="padding:6px 12px;color:#aaa;">no baseline</td>'
      : `<td style="padding:6px 12px;color:#777;">${b.toFixed(dp)} ${unit}</td>` +
        `<td style="padding:6px 12px;color:${anomColor(anom)};font-weight:500;">` +
        `${anom >= 0 ? '+' : ''}${anom.toFixed(dp)} ${unit}</td>`;
    return `<tr>
      <td style="padding:6px 12px;">${label}</td>
      <td style="padding:6px 12px;font-weight:600;font-size:1.05rem;">${value.toFixed(dp)} ${unit}</td>
      ${anomCell}
    </tr>`;
  }

  out.innerHTML = `
    <div style="background:white;border-radius:8px;padding:18px 20px;
                box-shadow:0 1px 3px rgba(0,0,0,0.08);">
      <div style="font-size:1.05rem;font-weight:600;color:#16425b;margin-bottom:10px;">
        ${formatDateLong(dateStr)}
      </div>
      <table style="width:100%;border-collapse:collapse;font-size:0.92rem;">
        <thead>
          <tr style="border-bottom:1px solid #eee;color:#888;font-size:0.78rem;text-transform:uppercase;">
            <th style="padding:6px 12px;text-align:left;">Measurement</th>
            <th style="padding:6px 12px;text-align:left;">Reading</th>
            <th style="padding:6px 12px;text-align:left;">1960–1990 day-of-year avg</th>
            <th style="padding:6px 12px;text-align:left;">Anomaly</th>
          </tr>
        </thead>
        <tbody>
          ${row('Max air temperature', day.tmax, '°C', 'tmax')}
          ${row('Min air temperature', day.tmin, '°C', 'tmin')}
          ${row('Mean air temperature',
                (day.tmax != null && day.tmin != null)
                  ? Math.round(((day.tmax + day.tmin)/2)*10)/10 : null,
                '°C', null)}
          ${row('Rainfall',           day.rain, 'mm', 'rain')}
          ${row('Ground temp (30 cm)', day.g30, '°C', 'g30')}
          ${row('Ground temp (100 cm)', day.g100, '°C', 'g100')}
        </tbody>
      </table>
    </div>
  `;
}
// Initialize the date picker
(function() {
  const di = document.getElementById('lookup-date');
  if (lookupBundle.date_min) {
    di.min = lookupBundle.date_min;
    di.max = lookupBundle.date_max;
    di.value = lookupBundle.date_max; // start at most recent date
    document.getElementById('lookup-range').textContent =
      `Available range: ${lookupBundle.date_min} to ${lookupBundle.date_max}`;
  }
  di.addEventListener('change', renderLookup);
  di.addEventListener('input', renderLookup);
})();

// =====================================================================
// SPEI (drought index)
// =====================================================================
const speiBundle = BUNDLE.spei;
let currentSpei = 'spei_12';
let speiView = 'monthly';

// Standard SPEI drought-category colour scale (brown = dry, teal = wet).
function speiColor(v) {
  if (v == null) return '#cccccc';
  if (v <= -2.0) return '#8c510a';   // extremely dry
  if (v <= -1.5) return '#bf812d';   // severely dry
  if (v <= -1.0) return '#dfc27d';   // moderately dry
  if (v <   1.0) return '#d9d9b8';   // near normal (muted)
  if (v <   1.5) return '#80cdc1';   // moderately wet
  if (v <   2.0) return '#35978f';   // severely wet
  return '#01665e';                  // extremely wet
}

function setupSpeiTab() {
  if (!speiBundle) {
    // climate_indices wasn't installed at build time
    const panel = document.getElementById('panel-spei');
    if (panel) {
      panel.innerHTML =
        '<div class="note">The SPEI tab is unavailable because the ' +
        '<code>climate_indices</code> package was not installed when this ' +
        'dashboard was built. Install it with <code>pip install ' +
        'climate_indices</code> and re-run the build script to enable it.</div>';
    }
    // Also hide the tab button
    const tabBtn = document.querySelector('.tab[data-tab="spei"]');
    if (tabBtn) tabBtn.style.display = 'none';
    return false;
  }
  return true;
}

function renderSpei() {
  if (!speiBundle) return;
  const s = speiBundle.series[currentSpei];
  const scale = s.scale;
  let traces, layout;

  if (speiView === 'annual') {
    const xs = s.annual_years;
    const ys = s.annual_values;
    traces = [{
      type: 'bar', x: xs, y: ys,
      marker: { color: ys.map(speiColor),
                line: { color: '#666', width: 0.3 } },
      hovertemplate: '<b>%{x}</b><br>Annual mean SPEI-' + scale +
                     ': <b>%{y:.2f}</b><extra></extra>',
      hoverlabel: { bgcolor: '#fff' },
    }];
    layout = speiLayout(`Annual mean SPEI-${scale}`, xs[0]-1, xs[xs.length-1]+1);
  } else {
    // Monthly: x is an index, with year tick labels
    const months = s.months;       // 'YYYY-MM'
    const ys = s.values;
    const xs = months.map(m => {
      const [y, mo] = m.split('-').map(Number);
      return y + (mo - 0.5) / 12;   // fractional year for a continuous axis
    });
    traces = [{
      type: 'bar', x: xs, y: ys,
      marker: { color: ys.map(speiColor) },
      width: 1/12 * 0.95,
      hovertext: months.map((m, i) =>
        ys[i] == null ? `${m}: no data`
                      : `<b>${m}</b><br>SPEI-${scale}: <b>${ys[i].toFixed(2)}</b>`),
      hoverinfo: 'text',
      hoverlabel: { bgcolor: '#fff' },
    }];
    layout = speiLayout(`Monthly SPEI-${scale}`,
                        speiBundle.start_year - 1, speiBundle.end_year + 1);
  }
  Plotly.react('chart-spei', traces, layout,
               { responsive: true, displaylogo: false });

  // Description + category band shading
  document.getElementById('spei-desc').innerHTML =
    (currentSpei === 'spei_12'
      ? 'SPEI-12: each point summarises the climatic water balance over the ' +
        'preceding 12 months — a measure of slow, accumulated (hydrological) ' +
        'drought or surplus.'
      : 'SPEI-3: each point summarises the preceding 3 months — a measure of ' +
        'shorter, seasonal dry or wet spells.') +
    `<span class="model-badge">Pearson III · Hargreaves PET</span>`;

  document.querySelectorAll('#panel-spei .btn[data-spei]').forEach(b =>
    b.classList.toggle('active', b.dataset.spei === currentSpei));
  document.querySelectorAll('#panel-spei .btn[data-spei-view]').forEach(b =>
    b.classList.toggle('active', b.dataset.speiView === speiView));
}

function speiLayout(title, xmin, xmax) {
  // Horizontal reference bands for the drought categories
  const band = (y0, y1, color) => ({
    type: 'rect', xref: 'paper', x0: 0, x1: 1, yref: 'y', y0, y1,
    fillcolor: color, opacity: 0.12, line: { width: 0 }, layer: 'below'
  });
  return {
    margin: { l: 55, r: 20, t: 10, b: 45 },
    xaxis: { title: 'Year', range: [xmin, xmax],
             showgrid: true, gridcolor: '#eee' },
    yaxis: { title: title, range: [-3.2, 3.2], zeroline: true,
             zerolinecolor: '#888', zerolinewidth: 1,
             showgrid: true, gridcolor: '#eee' },
    shapes: [
      band(-3.2, -2.0, '#8c510a'),
      band(-2.0, -1.0, '#bf812d'),
      band( 1.0,  2.0, '#35978f'),
      band( 2.0,  3.2, '#01665e'),
    ],
    annotations: [
      { x: 1, xref: 'paper', xanchor: 'right', y: -2.5, yref: 'y',
        text: 'drought', showarrow: false,
        font: { size: 10, color: '#8c510a' } },
      { x: 1, xref: 'paper', xanchor: 'right', y: 2.5, yref: 'y',
        text: 'very wet', showarrow: false,
        font: { size: 10, color: '#01665e' } },
    ],
    hovermode: 'closest', bargap: 0,
    paper_bgcolor: 'white', plot_bgcolor: 'white',
  };
}

document.querySelectorAll('#panel-spei .btn[data-spei]').forEach(b =>
  b.addEventListener('click', () => { currentSpei = b.dataset.spei; renderSpei(); }));
document.querySelectorAll('#panel-spei .btn[data-spei-view]').forEach(b =>
  b.addEventListener('click', () => { speiView = b.dataset.speiView; renderSpei(); }));

const speiAvailable = setupSpeiTab();

// =====================================================================
// Top-level tabs
// =====================================================================
const PANELS = ['overview', 'rain', 'temp', 'ground', 'doy', 'heatmap',
                'spei', 'lookup'];
const RENDERERS = {
  overview: renderOverview,
  rain: renderRain,
  temp: renderTemp,
  ground: renderGround,
  doy: renderDoy,
  heatmap: renderHeatmap,
  spei: renderSpei,
  lookup: renderLookup,
};
document.querySelectorAll('.tab').forEach(t =>
  t.addEventListener('click', () => {
    const target = t.dataset.tab;
    document.querySelectorAll('.tab').forEach(x =>
      x.classList.toggle('active', x === t));
    PANELS.forEach(p =>
      document.getElementById('panel-' + p).classList.toggle('active', p === target));
    RENDERERS[target]();
    // Plotly needs a resize nudge after the panel becomes visible
    setTimeout(() => {
      ['chart-overview','chart-rain','chart-temp','chart-ground',
       'chart-doy-tmean','chart-doy-tmax','chart-doy-tmin',
       'chart-heatmap','chart-spei'].forEach(id => {
        const el = document.getElementById(id);
        if (el && el.offsetParent !== null) Plotly.Plots.resize(el);
      });
    }, 50);
  }));

// Initial render
renderOverview();
renderRain();
renderTemp();
renderGround();
renderDoy();
renderHeatmap();
if (speiAvailable) renderSpei();
renderLookup();
</script>
</body>
</html>
"""


# ===========================================================================
# SECTION 7 — SPEI (Standardised Precipitation-Evapotranspiration Index)
# ===========================================================================

def build_spei(air: pd.DataFrame, rain: pd.DataFrame):
    """Compute SPEI at the configured timescales using the climate_indices
    package, with Hargreaves PET derived from daily tmax/tmin.

    Returns a dict describing each SPEI timescale as a monthly series, plus
    annual means and category counts, ready for charting. Returns None if
    climate_indices is unavailable.
    """
    if not HAVE_CLIMATE_INDICES:
        return None

    LAT = STATION_LATITUDE

    # --- Build a gap-free daily series over the common period ---
    a = air[['date', 'max_air_temp', 'min_air_temp']].copy()
    r = rain[['date', 'rain']].copy()
    start_year = max(a['date'].min().year, r['date'].min().year)
    end_year   = min(a['date'].max().year, r['date'].max().year)

    full_idx = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq='D')
    a = a.set_index('date').reindex(full_idx)
    r = r.set_index('date').reindex(full_idx)

    # Fix the handful of tmin>tmax data errors by swapping
    swap = a['min_air_temp'] > a['max_air_temp']
    a.loc[swap, ['min_air_temp', 'max_air_temp']] = \
        a.loc[swap, ['max_air_temp', 'min_air_temp']].values
    # Interpolate temperature gaps (PET is a smooth function of temperature;
    # short gaps interpolate well). Limit keeps us from inventing long stretches.
    a['max_air_temp'] = a['max_air_temp'].interpolate(limit=15).bfill().ffill()
    a['min_air_temp'] = a['min_air_temp'].interpolate(limit=15).bfill().ffill()
    a['tmean'] = (a['max_air_temp'] + a['min_air_temp']) / 2

    # --- Daily PET via Hargreaves, then monthly totals ---
    pet_daily = ci_eto.eto_hargreaves(
        a['min_air_temp'].values, a['max_air_temp'].values,
        a['tmean'].values, LAT)
    a['pet'] = pet_daily
    a['ym'] = a.index.to_period('M')
    r['ym'] = r.index.to_period('M')
    monthly_pet  = a.groupby('ym')['pet'].sum()
    # Require at least 20 valid days for a monthly rain total to be trustworthy
    monthly_rain = r.groupby('ym')['rain'].sum(min_count=20)

    months = pd.period_range(f"{start_year}-01", f"{end_year}-12", freq='M')
    P   = pd.Series(monthly_rain, dtype=float).reindex(months)
    PET = pd.Series(monthly_pet,  dtype=float).reindex(months).values.astype(np.float64)

    # Fill any missing monthly rainfall with that calendar month's mean so the
    # index routine sees a complete series (a handful of months over 110+ years).
    n_filled = int(P.isna().sum())
    P = P.groupby(P.index.month).transform(lambda s: s.fillna(s.mean()))
    P = P.values.astype(np.float64)

    month_labels = [str(m) for m in months]          # e.g. '1911-01'
    year_of_month = np.array([m.year for m in months])

    series = {}
    for scale in SPEI_SCALES:
        spei = ci_indices.spei(
            precips_mm=P, pet_mm=PET, scale=scale,
            distribution=ci_indices.Distribution.pearson,
            periodicity=ci_compute.Periodicity.monthly,
            data_start_year=start_year,
            calibration_year_initial=SPEI_CALIB_START,
            calibration_year_final=SPEI_CALIB_END,
        )
        spei = np.asarray(spei, dtype=float)
        vals = [None if (v is None or np.isnan(v)) else round(float(v), 2)
                for v in spei]
        # Annual mean SPEI (mean of the monthly values within each calendar year)
        ann = {}
        for yr in sorted(set(year_of_month.tolist())):
            mask = year_of_month == yr
            yvals = spei[mask]
            yvals = yvals[~np.isnan(yvals)]
            if len(yvals):
                ann[int(yr)] = round(float(np.mean(yvals)), 2)
        series[f'spei_{scale}'] = {
            'scale': scale,
            'months': month_labels,
            'values': vals,
            'annual_years': list(ann.keys()),
            'annual_values': list(ann.values()),
        }

    return {
        'series': series,
        'start_year': start_year,
        'end_year': end_year,
        'latitude': LAT,
        'calib_start': SPEI_CALIB_START,
        'calib_end': SPEI_CALIB_END,
        'n_filled_months': n_filled,
        'scales': SPEI_SCALES,
    }


def build_date_lookup(air, rain, ground):
    """Build a per-date lookup table for the "look up a date" tab.
    Returns a dict { 'YYYY-MM-DD': {tmax, tmin, rain, g30, g100} }."""
    air = air.copy()
    rain = rain.copy()
    # Use date string as key
    air['key']  = air['date'].dt.strftime('%Y-%m-%d')
    rain['key'] = rain['date'].dt.strftime('%Y-%m-%d')

    lookup = {}
    for _, r in air[['key', 'max_air_temp', 'min_air_temp']].iterrows():
        d = lookup.setdefault(r['key'], {})
        if pd.notna(r['max_air_temp']): d['tmax'] = round(float(r['max_air_temp']), 1)
        if pd.notna(r['min_air_temp']): d['tmin'] = round(float(r['min_air_temp']), 1)
    for _, r in rain[['key', 'rain']].iterrows():
        d = lookup.setdefault(r['key'], {})
        if pd.notna(r['rain']): d['rain'] = round(float(r['rain']), 1)
    # Ground temp - two depths from BG
    for label, df_g in ground.items():
        col = 'g30' if '30' in label else 'g100'
        gd = df_g.copy()
        gd['key'] = gd['date'].dt.strftime('%Y-%m-%d')
        for _, r in gd[['key', 'value']].iterrows():
            d = lookup.setdefault(r['key'], {})
            if pd.notna(r['value']): d[col] = round(float(r['value']), 1)

    # Also compute the day-of-year baseline (1960-1990) for each metric so we
    # can show anomaly information in the lookup
    all_dates = pd.to_datetime(list(lookup.keys()))
    doy_index = pd.DataFrame({'date': all_dates,
                              'key': [d.strftime('%Y-%m-%d') for d in all_dates],
                              'doy': all_dates.dayofyear,
                              'year': all_dates.year})
    # Build baselines from the actual lookup data
    baselines = {}
    for metric in ['tmax', 'tmin', 'rain', 'g30', 'g100']:
        vals_by_doy = {}
        for _, row in doy_index.iterrows():
            if row['year'] < BASELINE_START or row['year'] > BASELINE_END:
                continue
            v = lookup.get(row['key'], {}).get(metric)
            if v is None: continue
            vals_by_doy.setdefault(int(row['doy']), []).append(v)
        baselines[metric] = {d: round(sum(vs)/len(vs), 2)
                             for d, vs in vals_by_doy.items() if vs}

    # Sort the lookup so keys are in chronological order (helps JS)
    sorted_lookup = {k: lookup[k] for k in sorted(lookup.keys())}

    return {
        'data': sorted_lookup,
        'baselines': baselines,
        'date_min': min(sorted_lookup.keys()) if sorted_lookup else None,
        'date_max': max(sorted_lookup.keys()) if sorted_lookup else None,
    }


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    # Friendly check first
    for name, p in [('airtemp', AIRTEMP_CSV),
                    ('ground temperature', GROUND_CSV),
                    ('rainfall', RAINFALL_CSV)]:
        if not p.exists():
            sys.exit(f"ERROR: {name} CSV not found at {p}\n"
                     f"Edit the CONFIG paths at the top of this script "
                     f"or place the files alongside it.")

    print("Loading air temperature ...")
    air = load_air_temperature()
    air_counts = air.groupby('year').size()
    n_complete_air = (air_counts >= MIN_OBS_PER_YEAR).sum()
    n_incomplete_air = (air_counts < MIN_OBS_PER_YEAR).sum()
    print(f"  {len(air):,} daily rows across {air['year'].nunique()} years "
          f"({air['year'].min()}–{air['year'].max()})")
    print(f"  {n_complete_air} complete, {n_incomplete_air} incomplete "
          f"(<{MIN_OBS_PER_YEAR} obs)")

    print("Loading rainfall ...")
    rain = load_rainfall()
    rain_counts = rain.groupby('year').size()
    n_complete_r = (rain_counts >= MIN_OBS_PER_YEAR).sum()
    n_incomplete_r = (rain_counts < MIN_OBS_PER_YEAR).sum()
    print(f"  {len(rain):,} daily rows across {rain['year'].nunique()} years "
          f"({rain['year'].min()}–{rain['year'].max()})")
    print(f"  {n_complete_r} complete, {n_incomplete_r} incomplete "
          f"(<{MIN_OBS_PER_YEAR} obs)")

    print("Loading ground temperature ...")
    ground = load_ground_temperature()
    for label, df in ground.items():
        cnt = df.groupby('year').size()
        n_c = (cnt >= MIN_OBS_PER_YEAR).sum()
        n_i = (cnt < MIN_OBS_PER_YEAR).sum()
        print(f"  {label}: {len(df):,} rows, "
              f"{df['year'].min()}–{df['year'].max()} "
              f"({n_c} complete, {n_i} incomplete)")

    print("Building annual overview ...")
    overview = build_annual_overview(air, rain, ground)

    print("Building rainfall extremes ...")
    rainfall_extremes = build_extreme_rainfall(rain)

    print("Building temperature extremes ...")
    temperature_extremes = build_extreme_temperature(air)

    print("Building day-of-year section ...")
    doy = build_doy_plot(air)

    print("Building heatmaps ...")
    heatmaps = build_heatmaps(air, rain, ground)

    print("Building ground-temperature extremes ...")
    ground_extremes = build_ground_extreme_temperature(ground)

    print("Building date lookup ...")
    date_lookup = build_date_lookup(air, rain, ground)

    if HAVE_CLIMATE_INDICES:
        print("Building SPEI ...")
        spei = build_spei(air, rain)
    else:
        print("Skipping SPEI (climate_indices not installed — "
              "run 'pip install climate_indices' to enable the SPEI tab).")
        spei = None

    bundle = {
        'overview':    overview,
        'rainfall':    rainfall_extremes,
        'temperature': temperature_extremes,
        'doy':         doy,
        'heatmaps':    heatmaps,
        'ground':      ground_extremes,
        'date_lookup': date_lookup,
        'spei':        spei,
    }

    html = HTML_TEMPLATE.replace('__DATA__', json.dumps(bundle, default=str))
    OUTPUT_HTML.write_text(html, encoding='utf-8')
    print(f"\n✅ Wrote {OUTPUT_HTML.resolve()}")
    print(f"   Size: {OUTPUT_HTML.stat().st_size / 1024:.0f} KB")
    print("   Open it in any browser to view the dashboard.")


if __name__ == '__main__':
    main()
