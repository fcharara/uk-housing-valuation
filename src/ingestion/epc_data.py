"""
Downloads and processes EPC (Energy Performance Certificate) data.
Source: https://epc.opendatacommunities.org/


EPC data must be manually downloaded (requires free registration).
Place all ZIP/CSV files in data/raw/epc/


Key variables extracted: total_floor_area, number_habitable_rooms,
    current_energy_rating, construction_age_band, lodgement_date,
    address1, address2, postcode
"""


import sys, logging, glob
import pandas as pd
from pathlib import Path


sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import RAW_DIR, PROC_DIR, PARQUET_COMPRESS, CHUNK_SIZE


log = logging.getLogger(__name__)


EPC_RAW_DIR  = RAW_DIR / 'epc'
EPC_PROC_DIR = PROC_DIR / 'epc'
EPC_PROC_DIR.mkdir(parents=True, exist_ok=True)


EPC_COLS = [
    'LMK_KEY', 'ADDRESS1', 'ADDRESS2', 'ADDRESS3', 'POSTCODE',
    'LODGEMENT_DATE', 'CURRENT_ENERGY_RATING', 'CURRENT_ENERGY_EFFICIENCY',
    'TOTAL_FLOOR_AREA', 'NUMBER_HABITABLE_ROOMS', 'CONSTRUCTION_AGE_BAND',
    'PROPERTY_TYPE', 'BUILT_FORM', 'TENURE',
]




def process_epc(overwrite=False):
    """Read all EPC CSVs and produce a cleaned, deduplicated parquet."""
    out_path = EPC_PROC_DIR / 'epc_cleaned.parquet'
    if out_path.exists() and not overwrite:
        log.info('EPC data already processed.')
        return pd.read_parquet(out_path)


    csv_files = sorted(glob.glob(str(EPC_RAW_DIR / '**' / 'certificates.csv'),
                                  recursive=True))
    if not csv_files:
        csv_files = sorted(glob.glob(str(EPC_RAW_DIR / '*.csv')))
    if not csv_files:
        raise FileNotFoundError(
            'No EPC CSV files found. Download from '
            'https://epc.opendatacommunities.org/ and place in data/raw/epc/'
        )


    log.info(f'Processing {len(csv_files)} EPC files...')
    chunks = []


    for fpath in csv_files:
        log.info(f'  Reading {fpath}...')
        try:
            for chunk in pd.read_csv(
                fpath, usecols=lambda c: c.upper() in [x.upper() for x in EPC_COLS],
                dtype=str, chunksize=CHUNK_SIZE, encoding='latin-1',
                on_bad_lines='skip',
            ):
                chunk.columns = chunk.columns.str.upper()
                # Normalise postcode
                chunk['POSTCODE'] = chunk['POSTCODE'].str.upper().str.strip()
                chunk['POSTCODE_NOSPACE'] = chunk['POSTCODE'].str.replace(' ', '', regex=False)


                # Parse lodgement date
                chunk['LODGEMENT_DATE'] = pd.to_datetime(
                    chunk['LODGEMENT_DATE'], errors='coerce'
                )


                # Numeric conversions
                chunk['TOTAL_FLOOR_AREA'] = pd.to_numeric(
                    chunk['TOTAL_FLOOR_AREA'], errors='coerce'
                )
                chunk['NUMBER_HABITABLE_ROOMS'] = pd.to_numeric(
                    chunk['NUMBER_HABITABLE_ROOMS'], errors='coerce'
                )
                chunk['CURRENT_ENERGY_EFFICIENCY'] = pd.to_numeric(
                    chunk['CURRENT_ENERGY_EFFICIENCY'], errors='coerce'
                )


                # Build combined address for fuzzy matching
                chunk['EPC_ADDRESS'] = (
                    chunk['ADDRESS1'].fillna('')
                    + ' ' + chunk['ADDRESS2'].fillna('')
                    + ' ' + chunk['ADDRESS3'].fillna('')
                ).str.strip().str.upper()


                # Drop rows missing critical fields
                chunk.dropna(
                    subset=['POSTCODE', 'LODGEMENT_DATE', 'TOTAL_FLOOR_AREA'],
                    inplace=True
                )


                # Floor area sanity filter
                chunk = chunk[chunk['TOTAL_FLOOR_AREA'].between(10, 1000)]


                chunks.append(chunk)
        except Exception as e:
            log.warning(f'  Failed on {fpath}: {e}')


    df = pd.concat(chunks, ignore_index=True)


    # Deduplicate: keep latest lodgement per address+postcode
    df.sort_values('LODGEMENT_DATE', ascending=False, inplace=True)
    df.drop_duplicates(subset=['POSTCODE_NOSPACE', 'EPC_ADDRESS'], keep='first', inplace=True)


    # Rename to lowercase for consistency
    df.columns = df.columns.str.lower()
    df.rename(columns={
        'total_floor_area': 'floor_area',
        'number_habitable_rooms': 'num_rooms',
        'current_energy_rating': 'energy_rating',
        'current_energy_efficiency': 'energy_efficiency_score',
    }, inplace=True)


    log.info(f'EPC processed: {len(df):,} unique records')
    df.to_parquet(out_path, compression=PARQUET_COMPRESS, index=False)
    return df




def run(overwrite=False):
    log.info('=== EPC Data: Starting ===')
    process_epc(overwrite)
    log.info('=== EPC Data: Done ===')




if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    run()
