from typing import Union, Dict
import sqlite3
import duckdb
import pandas as pd

def create_table_if_not_exists(conn: Union[sqlite3.Connection, duckdb.DuckDBPyConnection]) -> None:
    """
    Creates a KeyValue table with a 'dict_name' column for both SQLite and DuckDB.
    
    Args:
        conn: Either a SQLite or DuckDB connection
    """
    is_duckdb = isinstance(conn, duckdb.DuckDBPyConnection)
    
    if is_duckdb:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS KeyValue (
                dict_name VARCHAR,
                key VARCHAR,
                value VARCHAR,
                PRIMARY KEY (dict_name, key)
            )
        """)
    else:
        with conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS KeyValue (
                    dict_name TEXT,
                    key TEXT,
                    value TEXT,
                    PRIMARY KEY (dict_name, key)
                )
            """)

def store_dict_in_db(conn: Union[sqlite3.Connection, duckdb.DuckDBPyConnection], 
                     dict_name: str, 
                     data_dict: Dict[str, str]) -> None:
    """
    Stores the given data_dict under the name 'dict_name'.
    
    Args:
        conn: Either a SQLite or DuckDB connection
        dict_name: Name of the dictionary to store
        data_dict: Dictionary to store
    """
    create_table_if_not_exists(conn)
    is_duckdb = isinstance(conn, duckdb.DuckDBPyConnection)
    
    # Convert the dictionary into a list of tuples
    dict_items = [(dict_name, str(k), str(v)) for k, v in data_dict.items()]
    
    if is_duckdb:
        # DuckDB uses different syntax for upsert
        for item in dict_items:
            conn.execute("""
                INSERT OR REPLACE INTO KeyValue (dict_name, key, value) 
                VALUES (?, ?, ?)
            """, item)
    else:
        # SQLite version
        with conn:
            conn.executemany(
                "INSERT OR REPLACE INTO KeyValue (dict_name, key, value) VALUES (?, ?, ?)",
                dict_items
            )

def fetch_dict_from_db(conn: Union[sqlite3.Connection, duckdb.DuckDBPyConnection], 
                      dict_name: str) -> Dict[str, str]:
    """
    Fetches all key-value pairs for the specified 'dict_name'.
    
    Args:
        conn: Either a SQLite or DuckDB connection
        dict_name: Name of the dictionary to fetch
        
    Returns:
        Dict[str, str]: The fetched dictionary
    """
    create_table_if_not_exists(conn)
    is_duckdb = isinstance(conn, duckdb.DuckDBPyConnection)
    
    if is_duckdb:
        # DuckDB version
        result = conn.execute("""
            SELECT key, value
            FROM KeyValue
            WHERE dict_name = ?
        """, [dict_name]).fetchall()
        return {k: v for (k, v) in result}
    else:
        # SQLite version
        cursor = conn.execute("""
            SELECT key, value
            FROM KeyValue
            WHERE dict_name = ?
        """, (dict_name,))
        rows = cursor.fetchall()
        return {k: v for (k, v) in rows}

def store_df_in_db(conn: Union[sqlite3.Connection, duckdb.DuckDBPyConnection],
                   df: pd.DataFrame,
                   table_name: str,
                   if_exists: str = 'replace',
                   index: bool = False) -> None:
    """
    Store a pandas DataFrame in either SQLite or DuckDB.
    
    Args:
        conn: Either a SQLite or DuckDB connection
        df: Pandas DataFrame to store
        table_name: Name of the table to create/update
        if_exists: How to behave if table exists ('fail', 'replace', or 'append')
        index: Whether to store the DataFrame index as a column
    """
    is_duckdb = isinstance(conn, duckdb.DuckDBPyConnection)
    
    if is_duckdb:
        # DuckDB version
        if if_exists == 'replace':
            conn.execute(f"DROP TABLE IF EXISTS {table_name}")
        conn.register('temp_df', df)
        conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM temp_df")
        conn.unregister('temp_df')
    else:
        # SQLite version
        df.to_sql(
            name=table_name,
            con=conn,
            if_exists=if_exists,
            index=index
        )

def read_df_from_db(conn: Union[sqlite3.Connection, duckdb.DuckDBPyConnection],
                    table_name: str) -> pd.DataFrame:
    """
    Read a table from either SQLite or DuckDB into a pandas DataFrame.
    
    Args:
        conn: Either a SQLite or DuckDB connection
        table_name: Name of the table to read
        
    Returns:
        pd.DataFrame: The table contents as a DataFrame
    """
    is_duckdb = isinstance(conn, duckdb.DuckDBPyConnection)
    
    if is_duckdb:
        # DuckDB version
        return conn.execute(f"SELECT * FROM {table_name}").df()
    else:
        # SQLite version
        return pd.read_sql(f"SELECT * FROM {table_name}", conn)

def check_table_exists(conn: Union[sqlite3.Connection, duckdb.DuckDBPyConnection], 
                      table_name: str) -> bool:
    """
    Check if a table exists in the database.
    
    Args:
        conn: Either a SQLite or DuckDB connection
        table_name: Name of the table to check
        
    Returns:
        bool: True if table exists, False otherwise
    """
    is_duckdb = isinstance(conn, duckdb.DuckDBPyConnection)
    
    if is_duckdb:
        result = conn.execute(f"""
            SELECT name 
            FROM sqlite_master 
            WHERE type='table' AND name='{table_name}'
        """).fetchone()
    else:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?;", 
            (table_name,)
        )
        result = cursor.fetchone()
        
    return result is not None

def append_df_to_table(conn: Union[sqlite3.Connection, duckdb.DuckDBPyConnection],
                      df: pd.DataFrame,
                      table_name: str) -> None:
    """
    Append DataFrame to existing table or create new one if it doesn't exist.
    
    Args:
        conn: Either a SQLite or DuckDB connection
        df: DataFrame to store
        table_name: Name of the table
    """
    if check_table_exists(conn, table_name):
        # Read existing data
        df_existing = read_df_from_db(conn, table_name)
        # Combine with new data
        df_combined = pd.concat([df_existing, df], ignore_index=True)
        # Store combined data
        store_df_in_db(conn, df_combined, table_name, if_exists='replace', index=False)
        print(f"Appended data to existing table '{table_name}'.")
    else:
        # Create new table
        store_df_in_db(conn, df, table_name, if_exists='fail', index=False)
        print(f"Created new table '{table_name}' with data.")

if __name__ == "__main__":
    # Example usage with both SQLite and DuckDB
    
    # SQLite example
    sqlite_conn = sqlite3.connect("example.db")
    store_dict_in_db(sqlite_conn, "sqlite_dict", {"type": "SQLite", "version": "3"})
    result_sqlite = fetch_dict_from_db(sqlite_conn, "sqlite_dict")
    print("SQLite result:", result_sqlite)
    sqlite_conn.close()
    
    # DuckDB example
    duck_conn = duckdb.connect("example.duckdb")
    store_dict_in_db(duck_conn, "duck_dict", {"type": "DuckDB", "version": "0.8"})
    result_duck = fetch_dict_from_db(duck_conn, "duck_dict")
    print("DuckDB result:", result_duck)
    duck_conn.close()
