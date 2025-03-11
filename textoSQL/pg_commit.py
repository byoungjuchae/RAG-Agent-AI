import psycopg2 as pg2
import pandas as pd




csv_data = pd.read_csv('./NVDA.csv')

conn = pg2.connect(host='postgres',
                   dbname ='postgres',
                   user='postgres',
                   password='postgres',
                   port=5432)

cur = conn.cursor()

cur.execute("""CREATE TABLE IF NOT EXISTS NVDA 
            ( date DATE,
                high FLOAT,
                low FLOAT,
                close FLOAT,
                volume BIGINT
                
                )"""
            )

conn.commit()

for index, row in csv_data.iterrows():
    cur.execute(
        "INSERT INTO nvda (date, high, low, close, volume) VALUES (%s, %s, %s, %s, %s)",
        (row['Date'], row['High'], row['Low'], row['Close'], row['Volume'])
    )

conn.commit()
cur.close()
conn.close()