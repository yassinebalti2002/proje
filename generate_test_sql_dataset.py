"""
generate_test_sql_dataset.py
=============================
Génère un dump SQL synthétique au format `ai_cp.full_data`, pour tester
le pipeline d'upload (/pipeline, pipeline_upload.html) sans utiliser le
vrai dump de production (658 MB).

Respecte exactement le format attendu par generate_dataset_from_sql.py :
  (id, 'SensorNodeId', 'timestamp', 'gph', '{json}', 'res', NULL)
avec JSON :
  - ligne Z : {"SensorNodeId":..,"MeasDetails":{"Id":..},"Temperature":..,
               "Vibration":{"RMS":{"Z":..}}}
  - ligne X : idem avec Vibration.RMS.X
  - ligne Y : idem avec Vibration.RMS.Y
  - ligne "acceleration" occasionnelle (~1/150) : Vibration.{A-P2P,A-Z2P,
    Crest,A-RMS}.Y + "mesure":{x,y,z,temperature}

Usage :
  python generate_test_sql_dataset.py --target-mb 180 --out data/ai_cp_test_sample.sql
"""

import argparse
import json
import random
from pathlib import Path
from datetime import datetime, timedelta

SENSORS = [
    "07da47b8", "0ff416d2", "2c6254af", "3a782f1b", "4b5e4b32",
    "53cb61b2", "68c11f06", "6e0c1740", "718fd2af", "8f7f2f7e",
    "91d92804", "99695e98", "a6a46be1", "aa7b02a1", "b2acdf45",
    "bc59bf5f", "d9508e77", "eb084747", "ed6fa322", "f48c25f9",
]

ROWS_PER_INSERT = 300


def gen_header() -> str:
    return (
        "-- Dump SQL synthétique — jeu de test pour /v1/pipeline/upload\n"
        "-- Format identique à ai_cp.full_data (mêmes clés JSON), valeurs simulées.\n"
        "-- NE CONTIENT AUCUNE DONNÉE RÉELLE.\n\n"
        "SET NAMES utf8mb4;\n"
        "SET FOREIGN_KEY_CHECKS=0;\n\n"
        "DROP TABLE IF EXISTS `full_data`;\n"
        "CREATE TABLE `full_data` (\n"
        "  `id` int(11) NOT NULL AUTO_INCREMENT,\n"
        "  `SensorNodeId` varchar(64) DEFAULT NULL,\n"
        "  `timestamp` datetime DEFAULT NULL,\n"
        "  `gph` varchar(64) DEFAULT NULL,\n"
        "  `data` longtext DEFAULT NULL,\n"
        "  `type` varchar(20) DEFAULT NULL,\n"
        "  `created_at` datetime DEFAULT NULL,\n"
        "  PRIMARY KEY (`id`)\n"
        ") ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;\n\n"
        "LOCK TABLES `full_data` WRITE;\n"
    )


def gen_footer() -> str:
    return "UNLOCK TABLES;\nSET FOREIGN_KEY_CHECKS=1;\n"


def esc(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def sql_row(row_id: int, sid: str, ts: str, gph: str, data: dict) -> str:
    payload = esc(json.dumps(data, separators=(",", ":")))
    return f"({row_id},'{sid}','{ts}','{gph}','{payload}','res',NULL)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-mb", type=int, default=180,
                     help="Taille cible du fichier .sql en MB (max conseillé: 200)")
    ap.add_argument("--out", default="data/ai_cp_test_sample.sql")
    args = ap.parse_args()

    target_bytes = args.target_mb * 1_000_000
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)

    # État par capteur : timestamp courant, température (marche aléatoire), MeasDetails.Id cyclique
    state = {
        sid: {
            "ts": datetime(2026, 1, 1) + timedelta(seconds=rng.randint(0, 3600)),
            "temp": rng.uniform(20, 35),
            "mid": rng.randint(0, 50),
        }
        for sid in SENSORS
    }

    written = 0
    row_id = 1
    buf = []

    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(gen_header())
        written += f.tell()

        pending_values = []

        def flush_insert():
            nonlocal pending_values
            if not pending_values:
                return
            stmt = "INSERT INTO `full_data` VALUES " + ",".join(pending_values) + ";\n"
            f.write(stmt)
            pending_values = []

        while written < target_bytes:
            for sid in SENSORS:
                st = state[sid]
                st["ts"] += timedelta(seconds=rng.randint(15, 45))
                ts_str = st["ts"].strftime("%Y-%m-%d %H:%M:%S")
                st["mid"] = (st["mid"] + 1) % 1000

                # Dérive lente de la température (marche aléatoire bornée)
                st["temp"] += rng.uniform(-0.3, 0.3)
                st["temp"] = max(14.0, min(54.0, st["temp"]))
                temp = round(st["temp"], 2)

                vib_z = round(max(0, rng.gauss(250, 150)), 1)
                vib_x = round(max(0, rng.gauss(300, 320)), 1)
                vib_y = round(max(0, rng.gauss(3, 4)), 1)

                mid = st["mid"]

                # Ligne Z (Temperature + Vibration.RMS.Z)
                pending_values.append(sql_row(
                    row_id, sid, ts_str, "temperature",
                    {"SensorNodeId": sid, "MeasDetails": {"Id": mid},
                     "Temperature": temp, "Vibration": {"RMS": {"Z": vib_z}}},
                ))
                row_id += 1

                # Ligne X
                pending_values.append(sql_row(
                    row_id, sid, ts_str, "vibration_x",
                    {"SensorNodeId": sid, "MeasDetails": {"Id": mid},
                     "Vibration": {"RMS": {"X": vib_x}}},
                ))
                row_id += 1

                # Ligne Y
                pending_values.append(sql_row(
                    row_id, sid, ts_str, "vibration_y",
                    {"SensorNodeId": sid, "MeasDetails": {"Id": mid},
                     "Vibration": {"RMS": {"Y": vib_y}}},
                ))
                row_id += 1

                # Mesure "acceleration" occasionnelle (~1 session sur 150),
                # horodatage propre, distincte du flux Z/X/Y ci-dessus.
                if rng.random() < 1 / 150:
                    acc_ts = st["ts"] + timedelta(seconds=rng.randint(1, 5))
                    acc_ts_str = acc_ts.strftime("%Y-%m-%d %H:%M:%S")
                    pending_values.append(sql_row(
                        row_id, sid, acc_ts_str, "acceleration,temperature",
                        {
                            "SensorNodeId": sid,
                            "Vibration": {
                                "A-P2P":  {"Y": round(rng.uniform(0, 900), 1)},
                                "A-Z2P":  {"Y": round(rng.uniform(0, 500), 1)},
                                "Crest":  {"Y": round(rng.uniform(0, 500), 1)},
                                "A-RMS":  {"Y": round(rng.uniform(0, 300), 1)},
                            },
                            "mesure": {
                                "x": round(rng.uniform(0, 900), 1),
                                "y": round(rng.uniform(0, 900), 1),
                                "z": round(rng.uniform(0, 900), 1),
                                "temperature": temp,
                            },
                        },
                    ))
                    row_id += 1

                if len(pending_values) >= ROWS_PER_INSERT:
                    flush_insert()
                    written = f.tell()
                    if written >= target_bytes:
                        break

        flush_insert()
        f.write(gen_footer())
        written = f.tell()

    size_mb = out_path.stat().st_size / 1_000_000
    print(f"OK : {out_path} généré — {size_mb:.1f} MB — {row_id - 1:,} lignes 'res'")


if __name__ == "__main__":
    main()
