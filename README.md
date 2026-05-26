# Cambridge Climate Dashboard
An interactive, single-page dashboard exploring over a century of weather observations from **Cambridge, UK** — air temperature, soil temperature, rainfall, and a temperature-aware drought index.  It is built from a single Python script that reads three CSV files and writes one self-contained HTML file. 
If you just want to look at the Dashboard, download the **"cambridge_climate_dashboard.html"** file and open in any browser.

This data analysis has been vibe-coded with Claude.ai.

## What's in it

The dashboard has eight tabs:

| Tab | What it shows |
|-----|---------------|
| **Annual overview** | Annual air temperature (mean/max/min), soil temperature (30 cm & 100 cm), total rainfall, and seasonal rainfall (winter/spring/summer/autumn). Each with linear or quadratic trend and a projection to 2050. |
| **Temperature extremes** | 12 indices: warm/hot/very-hot days, heatwave streaks, frost/hard-frost/icing days, tropical nights, last spring frost, and annual record highs/lows. |
| **Ground extremes** | Soil-temperature extremes at 30 cm and 100 cm: cold/frost/warm/hot-soil days, growing-season length, annual soil max/min. |
| **Rainfall extremes** | Heavy-rain-day counts, wettest day, share of rain from very wet days, dry/wet spells, and drought-style metrics (longest marginal-rain stretch, dry months). |
| **Day-of-year lines** | One line per year (or 5-year average) coloured old→recent, for daily mean/max/min. Includes smoothing, click-to-pin, and a two-period comparison tool. |
| **Heatmaps** | Week-of-year × year grids for air temperature, rainfall, and soil temperature, in absolute or anomaly-vs-baseline colours. |
| **Drought (SPEI)** | The Standardised Precipitation-Evapotranspiration Index at 3- and 12-month timescales, which folds warming-driven evaporative demand into the drought picture. |
| **Date lookup** | Type any date and see that day's readings versus the 1960–1990 day-of-year average. |

## Data sources

- **Air & soil temperature**: Met Office MIDAS Open (via CEDA), Cambridge Botanic Garden, station 00454.
- **Rainfall**: NOAA GHCN-Daily, Cambridge NIAB.

The reference period for anomalies and the SPEI calibration is the WMO standard climate normal, **1961–1990** (labelled 1960–1990 in the UI as the 1960 observation is included where available).

## Rebuilding the dashboard yourself

```bash
# 1. Install dependencies
pip install pandas numpy scipy statsmodels
pip install climate_indices      # optional — enables the Drought (SPEI) tab

# 2. Put the three CSVs next to build_dashboard.py (filenames as below)
#    - midas_merged_daily_airtemp.csv
#    - midas_merged_daily_groundtemp.csv
#    - NIAB_daily_rainfall_NOAA.csv

# 3. Run it
python3 build_dashboard.py

# 4. Open the generated cambridge_climate_dashboard.html
```

If `climate_indices` is not installed, the script still runs — it simply omits the Drought (SPEI) tab.

## Methodology notes

- **Trends**: continuous metrics use ordinary least squares; count metrics
  (days per year) use Negative Binomial regression with a log link so
  projections stay non-negative. Linear vs. quadratic is chosen by AIC, only
  preferring quadratic when it improves AIC by at least 2.
- **Incomplete years**: years with fewer than 340 daily observations are shown
  in striped grey and excluded from trend fits.
- **SPEI**: potential evapotranspiration is estimated with the Hargreaves
  method (from daily max/min temperature and latitude). The water balance
  (precipitation − PET) is fitted to a Pearson Type III distribution. Because
  Hargreaves PET is temperature-based, the SPEI here reflects evaporative
  demand as estimated from temperature, not a full energy-balance calculation.

These are statistical descriptions and extrapolations of the historical record,
**not** climate-model projections.

## Licence / attribution

Remember to credit the underlying data providers (Met Office / CEDA and NOAA) according to their respective licences when sharing.
