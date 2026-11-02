"""Read-only full CSV audit; writes findings to dataset_audit_results.json."""
from collections import Counter, defaultdict
from pathlib import Path
import json
import re
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
dictionary = pd.read_csv(ROOT / 'flights_sample_3m_features_dictionary.csv')
columns = list(pd.read_csv(ROOT / 'flights_sample_3m.csv', nrows=0).columns)
counts = Counter()
missing = Counter()
dtypes = defaultdict(set)
values = defaultdict(Counter)
ranges = {}
year_stats = defaultdict(Counter)
groups = defaultdict(Counter)
hashes, key_hashes, daily, eligible_delays = [], [], [], []
cancellation_by_year = Counter()
airlines = set()
origins, destinations, routes = set(), set(), set()
previous_date = None
times = ['CRS_DEP_TIME', 'DEP_TIME', 'WHEELS_OFF', 'WHEELS_ON', 'CRS_ARR_TIME', 'ARR_TIME']
causes = [c for c in columns if c.startswith('DELAY_DUE_')]
for df in pd.read_csv(ROOT / 'flights_sample_3m.csv', chunksize=250_000):
    counts['rows'] += len(df)
    missing.update({c: int(n) for c, n in df.isna().sum().items()})
    for c in columns:
        dtypes[c].add(str(df[c].dtype))
    dates = pd.to_datetime(df.FL_DATE, format='%Y-%m-%d', errors='coerce')
    counts['invalid_dates'] += int(dates.isna().sum())
    counts['date_order_inversions'] += int((dates.diff().dt.total_seconds() < 0).sum())
    if previous_date is not None and dates.iloc[0] < previous_date:
        counts['date_order_inversions'] += 1
    previous_date = dates.iloc[-1]
    cancelled = df.CANCELLED.eq(1)
    diverted = df.DIVERTED.eq(1)
    eligible = ~cancelled & df.DEP_DELAY.notna()
    delayed = eligible & df.DEP_DELAY.ge(15)
    eligible_delays.append(df.loc[eligible, 'DEP_DELAY'].to_numpy())
    cancellation_by_year.update({f'{int(y)}:{code}': int(n) for (y, code), n in df.loc[cancelled].groupby([dates.dt.year, 'CANCELLATION_CODE']).size().items()})
    counts['CRS_ELAPSED_TIME_under_10'] += int(df.CRS_ELAPSED_TIME.lt(10).sum())
    counts.update({'cancelled': int(cancelled.sum()), 'diverted': int(diverted.sum()),
                   'delay_eligible': int(eligible.sum()), 'delayed_15m': int(delayed.sum()),
                   'eligible_diverted': int((eligible & diverted).sum()),
                   'noncancelled_missing_dep_delay': int((~cancelled & df.DEP_DELAY.isna()).sum()),
                   'cancelled_with_dep_delay': int((cancelled & df.DEP_DELAY.notna()).sum()),
                   'negative_dep_delay': int(df.DEP_DELAY.lt(0).sum()),
                   'noncancelled_with_cancellation_code': int((~cancelled & df.CANCELLATION_CODE.notna()).sum()),
                   'cancelled_without_cancellation_code': int((cancelled & df.CANCELLATION_CODE.isna()).sum()),
                   'causes_partial_missing': int((df[causes].isna().any(axis=1) & ~df[causes].isna().all(axis=1)).sum()),
                   'arr_delay_ge15_causes_missing': int((df.ARR_DELAY.ge(15) & df[causes].isna().all(axis=1)).sum()),
                   'arr_delay_lt15_causes_present': int((df.ARR_DELAY.lt(15) & df[causes].notna().any(axis=1)).sum())})
    for label, mask in [('cancelled', cancelled), ('diverted_not_cancelled', diverted & ~cancelled), ('normal', ~cancelled & ~diverted)]:
        groups[label]['rows'] += int(mask.sum())
        for c in ['DEP_DELAY', 'ARR_DELAY', 'DEP_TIME', 'AIR_TIME', 'CRS_ELAPSED_TIME']:
            groups[label][c + '_missing'] += int(df.loc[mask, c].isna().sum())
    for year in dates.dt.year.dropna().unique():
        m = dates.dt.year.eq(year)
        year_stats[int(year)].update({'rows': int(m.sum()), 'cancelled': int((m & cancelled).sum()),
                                     'delay_eligible': int((m & eligible).sum()), 'delayed': int((m & delayed).sum())})
    for c in ['CANCELLED', 'DIVERTED', 'CANCELLATION_CODE']:
        values[c].update({str(k): int(v) for k, v in df[c].value_counts(dropna=False).items()})
    for c in df.select_dtypes(include='number').columns:
        lo, hi = df[c].min(), df[c].max()
        if pd.notna(lo):
            old = ranges.get(c, [float(lo), float(hi)])
            ranges[c] = [min(old[0], float(lo)), max(old[1], float(hi))]
    for c in times:
        s = df[c].dropna()
        counts[c + '_2400'] += int(s.eq(2400).sum())
        counts[c + '_invalid_hhmm_except_2400'] += int(((s < 0) | (s > 2400) | ((s % 100) >= 60) | (s % 1 != 0)).sum())
    for c in ['DISTANCE', 'CRS_ELAPSED_TIME', 'ELAPSED_TIME', 'AIR_TIME']:
        counts[c + '_nonpositive'] += int(df[c].le(0).sum())
    hashes.append(pd.util.hash_pandas_object(df, index=False).to_numpy())
    key = ['FL_DATE', 'AIRLINE_CODE', 'FL_NUMBER', 'ORIGIN', 'DEST', 'CRS_DEP_TIME']
    key_hashes.append(pd.util.hash_pandas_object(df[key], index=False).to_numpy())
    airlines.update(df[['AIRLINE_CODE', 'DOT_CODE', 'AIRLINE', 'AIRLINE_DOT']].itertuples(index=False, name=None))
    origins.update(df.ORIGIN.dropna().unique())
    destinations.update(df.DEST.dropna().unique())
    routes.update(zip(df.ORIGIN, df.DEST))
    d = pd.DataFrame({'date': dates, 'origin': df.ORIGIN, 'rows': 1, 'eligible': eligible.astype(int), 'delayed': delayed.astype(int)})
    daily.append(d.groupby(['date', 'origin'], observed=True)[['rows', 'eligible', 'delayed']].sum())
    print(f"Scanned {counts['rows']:,} rows", flush=True)

airport_day = pd.concat(daily).groupby(level=[0, 1]).sum()
date_counts = airport_day.groupby(level=0)['rows'].sum()
airport_summary = airport_day.groupby(level=1).agg(rows=('rows', 'sum'), days=('rows', 'size'), eligible_days=('eligible', lambda x: int(x.gt(0).sum())))
airport_summary['coverage'] = airport_summary.days / len(date_counts)
airport_summary['mean_rows_per_observed_day'] = airport_summary.rows / airport_summary.days
for name, blocks in [('full_row_hash_duplicate_candidates', hashes), ('flight_key_hash_duplicate_candidates', key_hashes)]:
    arr = np.concatenate(blocks)
    counts[name] = int(len(arr) - len(np.unique(arr)))
plan = (ROOT / 'flight_disruption_research_plan.md').read_text(encoding='utf-8')
variable_block = re.search(r'Typical variables.*?```text\s*(.*?)```', plan, re.S).group(1).split()
result = {
    'counts': dict(counts), 'columns': columns,
    'dictionary_columns': dictionary['Updated Header'].tolist(),
    'dataset_not_in_dictionary': sorted(set(columns) - set(dictionary['Updated Header'])),
    'dictionary_not_in_dataset': sorted(set(dictionary['Updated Header']) - set(columns)),
    'plan_columns_not_in_dataset': sorted(set(variable_block) - set(columns)),
    'observed_dtypes': {c: sorted(v) for c, v in dtypes.items()},
    'dictionary_dtype_mismatches': {r['Updated Header']: {'declared': r['Data Type'], 'observed': sorted(dtypes[r['Updated Header']])} for _, r in dictionary.iterrows() if r['Data Type'] not in dtypes[r['Updated Header']]},
    'missing': dict(missing), 'ranges': ranges, 'values': dict(values), 'groups': dict(groups),
    'years': dict(year_stats),
    'cancellation_by_year_code': dict(cancellation_by_year),
    'noncancelled_dep_delay_quantiles': pd.Series(np.concatenate(eligible_delays)).quantile([0, .01, .5, .9, .95, .99, .999, 1]).to_dict(),
    'dates': {'min': str(date_counts.index.min().date()), 'max': str(date_counts.index.max().date()),
              'observed_days': len(date_counts), 'missing_calendar_days': len(pd.date_range(date_counts.index.min(), date_counts.index.max()).difference(date_counts.index)),
              'daily_sample_rows_quantiles': date_counts.quantile([0, .25, .5, .75, 1]).to_dict()},
    'airline_mappings': sorted(airlines), 'origin_count': len(origins), 'destination_count': len(destinations), 'route_count': len(routes),
    'airport_day': {'observed_pairs': len(airport_day), 'rows_quantiles': airport_day.rows.quantile([0, .25, .5, .75, .9, 1]).to_dict(),
                    'pairs_under_10_rows': int(airport_day.rows.lt(10).sum()),
                    'pairs_zero_eligible': int(airport_day.eligible.eq(0).sum()),
                    'airports_full_calendar': int(airport_summary.days.eq(len(date_counts)).sum()),
                    'top20': airport_summary.sort_values('rows', ascending=False).head(20).reset_index().to_dict('records')}
}
(ROOT / 'dataset_audit_results.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps({k: result[k] for k in ['counts', 'dataset_not_in_dictionary', 'dictionary_dtype_mismatches', 'years', 'dates', 'airport_day']}, indent=2))
