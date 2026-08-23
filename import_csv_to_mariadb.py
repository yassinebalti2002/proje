"""
import_csv_to_mariadb.py
=========================
Importe data/dataset_real_full.csv (donnees reelles capteurs IFM) dans la table
full_data de MariaDB, au format exact attendu par realtime_mariadb.py.

Chaque ligne CSV (sensor_id, timestamp, temperature, vibration_x, vibration_y,
vibration_z) est eclatee en 3 lignes full_data (gph=temperature/vibration_x/
vibration_y), regroupees par MeasDetails.Id - exactement comme le fait la
gateway IFM reelle.

Usage :
    python import_csv_to_mariadb.py --limit 3000
"""

import argparse
import csv
import json
import os
import sys

import mysql.connector

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/dataset_real_full.csv")
    parser.add_argument("--host", default=os.getenv("MYSQL_HOST", "localhost"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MYSQL_PORT", "3306")))
    parser.add_argument("--user", default=os.getenv("MYSQL_USER", "root"))
    parser.add_argument("--password", default=os.getenv("MYSQL_PASSWORD", ""))
    parser.add_argument("--database", default=os.getenv("MYSQL_DATABASE", "ai_cp"))
    parser.add_argument("--limit", type=int, default=3000,
                         help="Nombre de mesures (lignes CSV) a importer, en partant de la fin (les plus recentes)")
    args = parser.parse_args()

    conn = mysql.connector.connect(
        host=args.host, port=args.port, user=args.user,
        password=args.password, database=args.database, autocommit=False,
    )
    cur = conn.cursor()

    print("Creation de la table full_data (si absente)...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS full_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            SensorNodeId VARCHAR(64),
            timestamp DATETIME,
            gph VARCHAR(32),
            data LONGTEXT,
            type VARCHAR(16)
        )
    """)
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sensor ON full_data (SensorNodeId)")
    conn.commit()

    print(f"Lecture de {args.csv}...")
    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    total = len(rows)
    rows = rows[-args.limit:] if args.limit else rows
    print(f"{total:,} lignes au total dans le CSV — import des {len(rows):,} dernieres mesures")

    insert_sql = """
        INSERT INTO full_data (SensorNodeId, timestamp, gph, data, type)
        VALUES (%s, %s, %s, %s, 'res')
    """

    batch = []
    meas_id = 0
    for row in rows:
        meas_id += 1
        sid = row["sensor_id"]
        ts = row["timestamp"]
        temp = float(row["temperature"])
        vx = float(row["vibration_x"])
        vy = float(row["vibration_y"])
        vz = float(row["vibration_z"])

        d_temp = json.dumps({
            "SensorNodeId": sid,
            "Temperature": temp,
            "Vibration": {"RMS": {"Z": vz}},
            "MeasDetails": {"Id": meas_id},
        })
        d_x = json.dumps({
            "SensorNodeId": sid,
            "Vibration": {"RMS": {"X": vx}},
            "MeasDetails": {"Id": meas_id},
        })
        d_y = json.dumps({
            "SensorNodeId": sid,
            "Vibration": {"RMS": {"Y": vy}},
            "MeasDetails": {"Id": meas_id},
        })

        batch.append((sid, ts, "temperature", d_temp))
        batch.append((sid, ts, "vibration_x", d_x))
        batch.append((sid, ts, "vibration_y", d_y))

        if len(batch) >= 3000:
            cur.executemany(insert_sql, batch)
            conn.commit()
            print(f"  {meas_id:,}/{len(rows):,} mesures inserees...", end="\r")
            batch = []

    if batch:
        cur.executemany(insert_sql, batch)
        conn.commit()

    print(f"\nImport termine : {meas_id:,} mesures ({meas_id*3:,} lignes full_data)")

    cur.execute("SELECT COUNT(*) FROM full_data")
    print(f"Total lignes dans full_data : {cur.fetchone()[0]:,}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
