#
#   Written by:  Mark W Kiehl
#   Company: Mechatronic Solutions LLC
#   http://mechatronicsolutionsllc.com/
#   http://www.savvysolutions.info/savvycodesolutions/

# MIT License
#
# Copyright (c) 2024 Mechatronic Solutions LLC


# configure logging
"""
import os.path
from pathlib import Path
import logging
logging.basicConfig(filename=Path(Path.cwd()).joinpath(os.path.basename(__file__).split('.')[0]+".log"), encoding='utf-8', level=logging.DEBUG, format='%(asctime)s %(message)s')
logging.info("Script start..")
"""

# Define the script version in terms of Semantic Versioning (SemVer)
# when Git or other versioning systems are not employed.
__version__ = "0.0.0"
from pathlib import Path
print(f"'{Path(__file__).stem}.py'\tv{__version__}\n")


"""
Local Data Storage System


Useful SQL Queries:
SELECT * FROM 'headlines' INNER JOIN 'source_float_int' ON headlines.source=source_float_int.source;

SELECT * FROM 'headlines' WHERE source is NULL;
UPDATE 'headlines' SET source='AP' WHERE source is NULL;
SELECT * FROM 'headlines' ORDER BY datetime_utc;
CREATE UNIQUE INDEX idx_dt_utc ON 'headlines' (datetime_utc);

SELECT * FROM 'source_float_int' WHERE source is NULL;
UPDATE 'source_float_int' SET source='AP' WHERE source is NULL;
SELECT * FROM 'source_float_int' ORDER BY id;
UPDATE 'source_float_int' SET float=3.45 WHERE id=2;
UPDATE 'source_float_int' SET int=8 WHERE id=2;
CREATE UNIQUE INDEX idx_id ON 'source_float_int' (id);



pip install pandas
pip install pyarrow
pip install duckdb
pip install odfpy
"""

# Create a new column from the index and reset it.
def move_df_idx_to_first_col(df=None, index_name_if_none="id"):
    """
    Returns a DataFrame where the index has been moved to the first column, 
    and the index has been reset.  The new column will have the name of the
    index.  If the index name is None, then it will be assigned the function
    argument value 'index_name_if_none'.

    df = move_df_idx_to_first_col(df)

    """
    cols = list(df.columns)
    if df.index.name is None: 
        df.index.name = index_name_if_none
    df[df.index.name] = df.index
    cols.insert(0, df.index.name)
    df = df.loc[:, cols]
    df.reset_index(drop=True, inplace=True)

    return df



# Create two demo csv & Parquet files from Pandas DataFrames
def create_demo_csv_parquet_files(verbose=False):
    """
    Returns the path/file to the Parquet files and CSV files created.

    path_files_parquet, path_files_csv = create_demo_csv_parquet_files()
    """

    from pathlib import Path

    import pandas as pd
    import numpy as np
    # pip install pyarrow
    #from pyarrow import parquet

    path_files_parquet = []
    path_files_csv = []


    # Create the first DataFrame / Parquet file 'source_float_int.parquet'.

    df = pd.DataFrame({'source': ['CNN', 'ABC', np.nan, 'USNWR'], 'float': [1.23, 2.34, float('Nan'), 4.45], 'int': [10, 9, None, 7]})
    if verbose: print(df)
    """
    id source  float   int
    0   0    CNN   1.23  10.0
    1   1    ABC   2.34   9.0
    2   2   None    NaN   NaN
    3   3  USNWR   4.45   7.0
    """

    # Save the DataFrame to a Parquet file
    path_file = Path(Path.cwd()).joinpath('data/source_float_int.parquet')
    if path_file.is_file(): path_file.unlink()
    path_files_parquet.append(path_file)
    print(f"Saving DataFrame to {path_file}")
    # Note: Snappy, LZ4, and zstd compression algorithms are faster to decompress than gzip.
    df.to_parquet(path=path_file, compression='snappy')     # snappy is the default compression
    if not path_file.is_file(): print(f"File not found {path_file}")

    # Save the DataFrame to a CSV file
    path_file = Path(Path.cwd()).joinpath('data/source_float_int.csv')
    if path_file.is_file(): path_file.unlink()
    path_files_csv.append(path_file)
    print(f"Saving DataFrame to {path_file}")
    df.to_csv(path_file, index=True, sep=";")
    del df



    # Create the 2nd DataFrame / Parquet file 'headlines.parquet'.

    unix_datetimes = [1702055700, 1702054318, 1702054250, 1702054080, 1702050060, 1702048116, 1702046440, 1702044611]
    # Convert to Numpy array as data type datetime64[s] (compatible with Pandas)
    datetime64 = np.array([unix_datetimes]).astype('datetime64[s]')
    #print(datetime64[0])    # ['2023-12-08T17:15:00' '2023-12-08T16:51:58' '2023-12-08T16:50:50' '2023-12-08T16:48:00' '2023-12-08T15:41:00' '2023-12-08T15:08:36'  '2023-12-08T14:40:40' '2023-12-08T14:10:11']
    headlines = ['Microsoft-OpenAI Partnership Draws Scrutiny From U.K. Regulator', 'US FTC examining Microsoft investment in OpenAI - Bloomberg News', "Microsoft, OpenAI Partnership Draws Scrutiny of UK's Competition Watchdog", 'Microsoft may face UK, FTC probes into OpenAI partnership after taking board seat', 'Does Intel Stock Have Its Mojo Back? We Talk to the Bulls.', 'Microsoft, OpenAI Are Facing a Potential Antitrust Probe in UK', 'UK antitrust regulators eyeing Microsoft-OpenAI partnership', "Microsoft's role in OpenAI on EU antitrust regulators' radar"]
    sources = ['CNN','ABC','USNWR',np.nan,'NBC','CBS','Fox','AP']
    # Create the Pandas dataframe
    df = pd.DataFrame({'datetime_utc': datetime64[0],
                    'headlines': headlines,
                    'source': sources}
                    )    
    # Set the index
    df.set_index('datetime_utc', inplace=True) # Create index from column 'unix_datetimes'
    if verbose: print(df)
    """
            datetime_utc                                          headlines source
    0 2023-12-08 17:15:00  Microsoft-OpenAI Partnership Draws Scrutiny Fr...    CNN
    1 2023-12-08 16:51:58  US FTC examining Microsoft investment in OpenA...    ABC
    2 2023-12-08 16:50:50  Microsoft, OpenAI Partnership Draws Scrutiny o...  USNWR
    3 2023-12-08 16:48:00  Microsoft may face UK, FTC probes into OpenAI ...   None
    4 2023-12-08 15:41:00  Does Intel Stock Have Its Mojo Back? We Talk t...    NBC
    5 2023-12-08 15:08:36  Microsoft, OpenAI Are Facing a Potential Antit...    CBS
    6 2023-12-08 14:40:40  UK antitrust regulators eyeing Microsoft-OpenA...    Fox
    7 2023-12-08 14:10:11  Microsoft's role in OpenAI on EU antitrust reg...     AP
    """

    # Save the DataFrame to a Parquet file
    path_file = Path(Path.cwd()).joinpath('data/headlines.parquet')
    if path_file.is_file(): path_file.unlink()
    path_files_parquet.append(path_file)
    print(f"Saving DataFrame to {path_file}")
    # Note: Snappy, LZ4, and zstd compression algorithms are faster to decompress than gzip.
    df.to_parquet(path=path_file, compression='snappy')     # snappy is the default compression
    if not path_file.is_file(): print(f"File not found {path_file}")

    # Save the DataFrame to a CSV file
    path_file = Path(Path.cwd()).joinpath('data/headlines.csv')
    if path_file.is_file(): path_file.unlink()
    path_files_csv.append(path_file)
    print(f"Saving DataFrame to {path_file}")
    df.to_csv(path_file, index=True, sep=";")
    del df

    return path_files_parquet, path_files_csv


def create_duckdb_db_from_parquet_files(path_files=None, path_file_db=None, verbose=True):
    """
    Reads the Parquet files specified by parquet_filenames and builds a DuckDB database with a table for each Parquet file.
    Returns the table names and the DuckDB path/file. 

    table_names, path_file_db = create_duckdb_db_from_parquet_files(verbose=False)
    """

    from pathlib import Path

    import pandas as pd
    # pip install pyarrow
    #from pyarrow import parquet

    # pip install duckdb
    import duckdb

    if not isinstance(path_files, list): raise Exception("path_files not a list")
    for path_file in path_files:
        if not isinstance(path_file, Path): raise Exception(f"path_file is not a pathlib Path()  {path_file}")
        if not path_file.is_file(): raise Exception(f"File not found  {path_file}")

    if not isinstance(path_file_db, Path): raise Exception(f"path_file_db is not a pathlib Path()  {path_file_db}")
    if path_file_db.is_file(): path_file_db.unlink()


    for path_file in path_files:
        df = pd.read_parquet(path=path_file)  
        if verbose: print(f"Read file {path_file.name} with {df.shape[0]} rows x {df.shape[1]} cols")

        # Create a DuckDB connection to a local database file
        with duckdb.connect(database=path_file_db) as conn:  

            # Must move the DataFrame index to the first column in order to include it in the table created from it.
            df = move_df_idx_to_first_col(df)
            print(df)

            # Write the DataFrame to a table in the DuckDB file
            sql = f"CREATE TABLE IF NOT EXISTS '{path_file.stem.lower()}' AS SELECT * FROM df;"
            if verbose: print(f"sql: {sql}")

            conn.execute(sql) 
            # OR
            #conn.sql(sql)

    del df

    table_names = get_db_table_names(path_file_db, verbose)
    return table_names

    """
    source_float_int
    ┌───────┬─────────┬────────┬────────┐
    │  id   │   cat   │ float  │  int   │
    │ int64 │ varchar │ double │ double │
    ├───────┼─────────┼────────┼────────┤
    │     0 │ A       │   1.23 │   10.0 │
    │     1 │ B       │   2.34 │    9.0 │
    │     2 │ NULL    │   NULL │   NULL │
    │     3 │ D       │   4.45 │    7.0 │
    └───────┴─────────┴────────┴────────┘

    headlines
    ┌─────────────────────┬───────────────────────────────────────────────────────────────────────────────────┐
    │    datetime_utc     │                                     headlines                                     │
    │    timestamp_ms     │                                      varchar                                      │
    ├─────────────────────┼───────────────────────────────────────────────────────────────────────────────────┤
    │ 2023-12-08 17:15:00 │ Microsoft-OpenAI Partnership Draws Scrutiny From U.K. Regulator                   │
    ..
    │ 2023-12-08 14:10:11 │ Microsoft's role in OpenAI on EU antitrust regulators' radar                      │
    └─────────────────────┴───────────────────────────────────────────────────────────────────────────────────┘
    """



def get_db_table_names(path_file_db=None, verbose=False):
    """
    Returns a list of the DuckDB db table names from path_file_db.
    """
    from pathlib import Path

    # pip install duckdb
    import duckdb

    if not isinstance(path_file_db, Path): raise Exception(f"path_file_db is not a pathlib Path()  {path_file_db}")
    if not path_file_db.is_file(): raise Exception(f"File not found {path_file_db}")


    table_names = []
    if verbose: print(f"DuckDB tables:")
    with duckdb.connect(database=path_file_db) as conn:  
        tables = conn.execute("SHOW TABLES").fetchall()
        for table in tables:
            table_names.append(table[0])
            if verbose: 
                print(f"\n{table[0]}")      # table is a tuple
                print(conn.sql(f"SELECT * FROM '{table[0]}'").show())

    return table_names


def query_db_tables(path_file_db=None, verbose=True):
    """
    """
    from pathlib import Path

    # pip install duckdb
    import duckdb

    if not isinstance(path_file_db, Path): raise Exception(f"path_file_db is not a pathlib Path()  {path_file_db}")
    if not path_file_db.is_file(): raise Exception(f"File not found {path_file_db}")

    def get_table_column_names(conn=None, table_name=None):
        tbl_col_names = []
        sql = f"DESCRIBE SELECT * from '{table_name}';"
        rows = conn.sql(sql).fetchall()
        for row in rows:
            tbl_col_names.append(row[0])
        return tbl_col_names

    def sql_rows_generator(conn=None):
        while True:
            row = conn.fetchone()
            if row is None: break
            yield row


    with duckdb.connect(database=path_file_db) as conn:  
        # Query table 'source_float_int'

        # Get the table column names
        #tbl_col_names = get_table_column_names(conn, 'source_float_int')
        #print(f"tbl_col_names: {tbl_col_names}")        # ['id', 'source', 'float', 'int']

        """
        sql = "SELECT * FROM 'source_float_int' ORDER BY id;"
        print(f"\nsql: {sql}")


        # Fetch the rows using a generator and process them one at a time. 
        conn.execute(sql)
        for row in sql_rows_generator(conn):
            print(row)

        
        # Get the rows from a SQL query one at at time.
        conn.execute(sql)
        while True:
            row = conn.fetchone()
            if row is None: break
            print(row) 

            
        # Get all of the rows from a SQL query at once. 
        #rows = conn.execute(sql).fetchall()
        rows = conn.sql(sql).fetchall()
        print(f"{len(rows)} rows for sql: {sql}")
        for row in rows:
            print(row)

        """

        # Perform a SQL join on the two tables between the column 'source'.
        """
        sql = f"SELECT * FROM 'source_float_int' INNER JOIN 'headlines' ON source_float_int.source=headlines.source;"
        conn.execute(sql)
        for row in sql_rows_generator(conn):
            print(row)
        (0, 'CNN', 1.23, 10.0, datetime.datetime(2023, 12, 8, 17, 15), 'Microsoft-OpenAI Partnership Draws Scrutiny From U.K. Regulator', 'CNN')
        (1, 'ABC', 2.34, 9.0, datetime.datetime(2023, 12, 8, 16, 51, 58), 'US FTC examining Microsoft investment in OpenAI - Bloomberg News', 'ABC')
        (3, 'USNWR', 4.45, 7.0, datetime.datetime(2023, 12, 8, 16, 50, 50), "Microsoft, OpenAI Partnership Draws Scrutiny of UK's Competition Watchdog", 'USNWR')
        """

        sql = f"SELECT * FROM 'headlines' INNER JOIN 'source_float_int' ON headlines.source=source_float_int.source;"
        conn.execute(sql)
        for row in sql_rows_generator(conn):
            print(row)
        """
        (datetime.datetime(2023, 12, 8, 17, 15), 'Microsoft-OpenAI Partnership Draws Scrutiny From U.K. Regulator', 'CNN', 0, 'CNN', 1.23, 10.0)
        (datetime.datetime(2023, 12, 8, 16, 51, 58), 'US FTC examining Microsoft investment in OpenAI - Bloomberg News', 'ABC', 1, 'ABC', 2.34, 9.0)
        (datetime.datetime(2023, 12, 8, 16, 50, 50), "Microsoft, OpenAI Partnership Draws Scrutiny of UK's Competition Watchdog", 'USNWR', 3, 'USNWR', 4.45, 7.0)
        """



def create_metadata(path_file_metadata=None, verbose=True):
    """
    Returns the JSON string for the data system metadata created and written to path_file_metadata.
    """
    import json
    from pathlib import Path

    json_str = '''
        {"author": "Mark W Kiehl", 
        "company": "Mechatronic Solutions LLC",
        "company_url": "http://mechatronicsolutionsllc.com/",
        "license": "MIT",
        "copyright": "Copyright (C) 2024 Mechatronic Solutions LLC",
        "description": "A data storage system for 2D, 3D, geospatial, geometry, high-dimensional vectors, and more!",
        "files": [
            {
            "headlines.csv": "News headlines with datatime and source.",
            "source_float_int.csv": "2D table of strings, float, and integer values."
            }
        ]
        }
    '''

    # Convert string to a dictionary
    json_str = json.loads(json_str)
    #print(f"type(json_str): {type(json_str)}")  # <class 'dict'>

    with open(path_file_metadata, 'w') as f:
        json.dump(json_str, f, indent=2)
    
    return json_str


def get_metadata(path_file_metadata=None, verbose=False):
    """
    Returns the JSON string for the data system metadata saved in path_file_metadata.
    """
    import json
    from pathlib import Path

    if not isinstance(path_file_metadata, Path): raise Exception(f"path_file_metadata is not a pathlib Path()  {path_file_metadata}")
    if not path_file_metadata.is_file(): raise Exception(f"File not found {path_file_metadata}")

    with open(path_file_metadata, 'r') as f:
        json_str = json.load(f)

    return json_str



# Write df to a .ods (spreadsheet) file 
def write_df_to_ods(df=None, path_file=None, verbose=False):
    """
    Writes the DataFrame df to path_file using Pandas pd.ExcelWriter().
    Overwrites the file path_file if it already exists.

    Requires odswriter  (pip install odswriter)

    """
    # https://pandas.pydata.org/docs/reference/api/pandas.ExcelWriter.html

    from pathlib import Path

    # pip install odswriter
    #import odswriter

    # pip install odfpy
    #https://pypi.org/project/odfpy/

    import pandas as pd
    # pip install pyarrow
    from pyarrow import parquet

    if not isinstance(df, pd.DataFrame): raise Exception("df is not a Pandas DataFrame")
    if not isinstance(path_file, Path): raise Exception("path_file is not a pathlib Path()")

    print(f"\nWriting df to {path_file}")
    
    # For .ods file, engine: odswriter: odf.opendocument.OpenDocumentSpreadsheet(**engine_kwargs)
    #with pd.ExcelWriter(path=path_file, engine="odswriter") as writer:
    with pd.ExcelWriter(path=path_file, engine="odf") as writer:
        df.to_excel(writer)



if __name__ == '__main__':
    import time
    t_start_sec = time.perf_counter()

    # Define the path/file for the DuckDB database file.
    from pathlib import Path
    path_file_db = Path(Path.cwd()).joinpath(f"data/local_data_storage_sys.duckdb")
    path_file_metadata = Path(Path.cwd()).joinpath(f"data/local_data_storage_sys.json")


    # Delete the files
    """
    if path_file_db.is_file(): path_file_db.unlink()
    if path_file_metadata.is_file(): path_file_metadata.unlink()
    """


    # Create the CSV & Parquet files, the DuckDB database path_file_db, and the metadata (only need to run these once)
    if not path_file_db.is_file():
        # Create the demo CSV & Parquet files
        path_files_parquet, path_files_csv = create_demo_csv_parquet_files()
        # Create the DuckDB db file from the Parquet files
        table_names = create_duckdb_db_from_parquet_files(path_files_parquet, path_file_db, verbose=True)
        print(f"table_names: {table_names}")
    if not path_file_metadata.is_file():
        # Create the metadata file for the data storage system
        json_str = create_metadata(path_file_metadata)
        print(f"json_str: {json_str}")


    # Accessing the data:

    # Get the metadata for the local_data_storage_sys project
    metadata = get_metadata(path_file_metadata)
    print(f"\n\nget_metadata():")
    print(metadata)

    # Read the DuckDB database file path_file_db.
    table_names = get_db_table_names(path_file_db, False)
    print(f"\ntable_names: {table_names}")

    # Query the db tables
    print(f"\nquery_db_tables()")
    query_db_tables(path_file_db)


    # Report the script execution time
    print(f"\nElapsed time {round(time.perf_counter()-t_start_sec,1)} sec")

    # ---------------------------------------------------------------------------
    #+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-
