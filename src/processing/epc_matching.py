"""
Fuzzy matching of Land Registry transactions to EPC records.
Implements Methodology Section 2.1.


Match criteria:
  1. Same postcode (exact match on normalised postcode)
  2. Address string similarity > 0.8 (fuzzywuzzy token_sort_ratio)
  3. EPC lodgement date within 2 years of transaction date
  4. Best match = highest similarity, then closest date
"""


import logging
import pandas as pd
import numpy as np
from fuzzywuzzy import fuzz


log = logging.getLogger(__name__)


SIMILARITY_THRESHOLD = 80   # fuzzywuzzy score 0-100
DATE_WINDOW_DAYS = 730      # 2 years




def build_lr_address(row):
    """Combine Land Registry address fields into a single string."""
    parts = [str(row.get('saon', '')), str(row.get('paon', '')),
             str(row.get('street', '')), str(row.get('locality', ''))]
    return ' '.join(p for p in parts if p and p != 'nan').upper().strip()




def match_epc_to_transactions(transactions, epc, sample_frac=None):
    """
    For each transaction, find the best matching EPC record.
    Returns the transactions DataFrame with EPC columns joined.
    """
    log.info('Starting EPC-to-LandRegistry fuzzy matching...')


    # Build address column for Land Registry
    transactions['lr_address'] = transactions.apply(build_lr_address, axis=1)
    transactions['postcode_nospace'] = (
        transactions['postcode'].str.upper().str.replace(' ', '', regex=False)
    )


    if sample_frac:
        transactions = transactions.sample(frac=sample_frac, random_state=42)
        log.info(f'  Sampled {len(transactions):,} transactions for matching')


    # Group EPC records by postcode for efficient lookup
    epc_grouped = epc.groupby('postcode_nospace')


    epc_cols = ['floor_area', 'num_rooms', 'energy_rating',
                'energy_efficiency_score', 'construction_age_band']


    results = []
    matched = 0
    total = len(transactions)


    for idx, (_, tx) in enumerate(transactions.iterrows()):
        if idx % 100_000 == 0 and idx > 0:
            log.info(f'  Matching progress: {idx:,}/{total:,} '
                     f'({matched:,} matched, {matched/idx*100:.1f}%)')


        pc = tx['postcode_nospace']
        if pc not in epc_grouped.groups:
            continue


        candidates = epc_grouped.get_group(pc)


        # Date window filter
        tx_date = tx['date_of_transfer']
        date_mask = (
            (candidates['lodgement_date'] >= tx_date - pd.Timedelta(days=DATE_WINDOW_DAYS))
            & (candidates['lodgement_date'] <= tx_date + pd.Timedelta(days=DATE_WINDOW_DAYS))
        )
        candidates = candidates[date_mask]
        if candidates.empty:
            continue


        # Compute similarity scores
        tx_addr = tx['lr_address']
        candidates = candidates.copy()
        candidates['sim_score'] = candidates['epc_address'].apply(
            lambda a: fuzz.token_sort_ratio(tx_addr, a)
        )


        # Filter by threshold
        candidates = candidates[candidates['sim_score'] >= SIMILARITY_THRESHOLD]
        if candidates.empty:
            continue


        # Best match: highest similarity, then closest date
        candidates['date_diff'] = abs(
            (candidates['lodgement_date'] - tx_date).dt.days
        )
        best = candidates.sort_values(
            ['sim_score', 'date_diff'], ascending=[False, True]
        ).iloc[0]


        match_data = {col: best[col] for col in epc_cols if col in best.index}
        match_data['_tx_idx'] = tx.name
        results.append(match_data)
        matched += 1


    log.info(f'  Matching complete: {matched:,}/{total:,} matched '
             f'({matched/total*100:.1f}%)')


    if not results:
        log.warning('  No matches found!')
        return transactions


    match_df = pd.DataFrame(results).set_index('_tx_idx')
    merged = transactions.join(match_df, how='inner')


    log.info(f'  Final matched dataset: {len(merged):,} rows')
    return merged
