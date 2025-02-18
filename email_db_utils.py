import pandas as pd
from sqlalchemy import create_engine
import datetime as dt
import yaml
from datetime import datetime, timedelta

class DataProcessor:
    def __init__(self):
        self.engine = self.get_atom_db_engine()
        self.table_mapping = self.get_table_map()

    def get_atom_db_engine(self):
        # Load database configuration from YAML file
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        url = config['database']['url']
        engine = create_engine(url, pool_size=1, max_overflow=1,
                               connect_args={"application_name": "macro_commentary"})
        return engine

    def query_email_table(self, schema: str, table_name: str, start_date: str, end_date: str) -> pd.DataFrame:
        table_name = self.table_mapping.get(table_name, table_name)

        sd_str = pd.to_datetime(start_date).strftime('%Y-%m-%d')
        ed_str = pd.to_datetime(end_date).strftime('%Y-%m-%d')

        query = f'''SELECT *
        FROM {schema}.{table_name}
        WHERE 1=1
        AND received_date_time BETWEEN '{sd_str}'::date AND '{ed_str}'::date
        ORDER BY received_date_time
        '''

        return pd.read_sql(query, con=self.engine)

    def last_thursday_and_previous_friday(self):
        # Get the current date
        current_date = dt.datetime.now()

        # Calculate the difference to the previous Thursday and Friday
        days_until_thursday = (current_date.weekday() - 3) % 7
        days_until_friday = 7 + (current_date.weekday() - 4) % 7

        # Calculate the dates for the last Thursday and the previous Friday
        last_thursday = current_date - timedelta(days=days_until_thursday)
        previous_friday = current_date - timedelta(days=days_until_friday)

        return last_thursday, previous_friday

    def get_table_map(self):
        # Load table mapping from YAML file
        with open('config.yaml', 'r') as file:
            config = yaml.safe_load(file)

        return config['table_map']
