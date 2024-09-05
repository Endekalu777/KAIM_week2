import pandas as pd
from dotenv import load_dotenv
import os
import psycopg2

def load_environment_variables():
    """Load environment variables from a .env file."""
    load_dotenv()
    return {
        'dbname': os.getenv('DB_NAME'),
        'user': os.getenv('DB_USER'),
        'password': os.getenv('DB_PASSWORD'),
        'host': os.getenv('DB_HOST'),
        'port': os.getenv('DB_PORT')
    }

def fetch_data_from_postgres(query, table_name):
    """Fetch data from PostgreSQL and return it as a DataFrame."""
    # Load environment variables
    conn_params = load_environment_variables()

    try:
        # Create a connection to the PostgreSQL database
        with psycopg2.connect(**conn_params) as conn:
            print("Connection successful!")
            
            # Define the full query
            full_query = f"SELECT * FROM {table_name}"
            
            # Execute the query and read the result into a DataFrame
            df = pd.read_sql_query(full_query, conn)
            print("Data imported successfully!")
            return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
