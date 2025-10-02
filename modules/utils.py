import re
import pandas as pd
import logging
from datetime import datetime
import pandas_gbq
from google.cloud.bigquery_storage import BigQueryReadClient

from config.client_initialiser import load_env_and_initialize_clients

_, bigquery_client, _ = load_env_and_initialize_clients()
bqstorage_client = BigQueryReadClient()

def standardize_column_names(df):
    df1 = df.copy()
    # Convert to lowercase and replace spaces with underscores
    new_columns = df1.columns.str.lower().str.replace(' ', '_')

    # Remove special characters, except underscores
    new_columns = new_columns.map(lambda x: re.sub(r'[^a-zA-Z0-9_]', '', x))

    # Convert to list for mutable operations
    new_columns = new_columns.tolist()

    # Append a suffix to duplicates
    counts = {}
    for i, col in enumerate(new_columns):
        if col in counts:
            counts[col] += 1
            new_columns[i] = f"{col}_{counts[col]}"
        else:
            counts[col] = 1

    # Assign the new column names back to the DataFrame
    df1.columns = new_columns
    return df1


def upload_to_bigquery_table(df,project_id, dataset, tableName, if_exists):
    full_table_name = f'{project_id}.{dataset}.{tableName}' # For logging
    logging.info(f"Attempting to upload DataFrame to BigQuery table: {full_table_name}. Rows: {df.shape[0]}, If exists: {if_exists}")
    if df.shape[0] > 0:
        df_auto_table = standardize_column_names(df)
        df_auto_table1 = df_auto_table.copy(deep=True)
        df_auto_table1 = df_auto_table1.astype(str)
        insertDate = datetime.now()
        df_auto_table1['insertdate'] = insertDate

            # Standardize dates
        df_auto_table1['insertdate'] = pd.to_datetime(df_auto_table1['insertdate'])
        df_auto_table1['insertdate'] = df_auto_table1['insertdate'].apply(lambda x: x.tz_localize(None) if x.tzinfo else x)

        tableName = f'{dataset}.{tableName}'

        # Upload the enriched dataframe to BigQuery
        pandas_gbq.to_gbq(df_auto_table1, destination_table=tableName,
                            project_id=project_id,
                            if_exists=if_exists)

        logging.info(f"Successfully uploaded data to BigQuery table: {tableName}")
    else:
        logging.warning(f"DataFrame is empty. Skipping upload to BigQuery table: {tableName}")


def create_df_from_bq(sql):
    """
    Executes a SQL query on BigQuery and returns the result as a Pandas DataFrame,
    using the BigQuery Storage API for potentially faster downloads.
    """
    # Use the BigQuery Storage API to download results more quickly.
    # The bqstorage_client argument enables the BigQuery Storage API.
    df = bigquery_client.query(sql).to_dataframe(bqstorage_client=bqstorage_client)
    return df