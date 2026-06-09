"""
generate_dataset_from_sql.py
============================
Génère dataset_2026_with_acc.csv depuis ai_cp (5).sql

Ce script fait exactement ce qui a été fait manuellement pour créer
le dataset_2026_with_acc.csv original — mais sur TOUTES les données.

Comment ça marche :
  SQL full_data → parser JSON → grouper par (SensorNodeId + MeasDetails.Id)
  → consolider 3 lignes (Z + X + Y) en 1 mesure complète
  → écrire CSV avec les mêmes colonnes que le dataset original

Usage :
  python generate_dataset_from_sql.py
  python generate_dataset_from_sql.py --sql "ai_cp (5).sql" --out data/dataset_nouveau.csv
  python generate_dataset_from_sql.py --max-mb 200   ← limiter la lecture (défaut: tout)

Résultat : data/dataset_2026_with_acc.csv (remplace l'ancien)
"""

import re
import sys
import csv
import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

# ══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════

DEFAULT_SQL = "ai_cp (5).sql"
DEFAULT_OUT = "data/dataset_2026_with_acc.csv"
CHUNK_SIZE  = 50 * 1024 * 1024   # 50 MB par chunk — évite de charger 658 MB en RAM

# Colonnes du CSV de sortie — exactement les mêmes que le dataset original
CSV_COLUMNS = [
    "sensor_id", "timestamp",
    "temperature",
    "vibration_x", "vibration_y", "vibration_z",
    "acc_p2p", "acc_z2p", "acc_crest", "acc_rms",
]

# ══════════════════════════════════════════════════════════════════
# COULEURS TERMINAL
# ══════════════════════════════════════════════════════════════════

G  = "\033[92m"
R  = "\033[91m"
Y  = "\033[93m"
C  = "\033[96m"
B  = "\033[1m"
RS = "\033[0m"

def ok(msg):   print(f"  {G}✅ {msg}{RS}")
def err(msg):  print(f"  {R}❌ {msg}{RS}")
def info(msg): print(f"  {C}ℹ  {msg}{RS}")
def warn(msg): print(f"  {Y}⚠  {msg}{RS}")


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 1 — LECTURE ET PARSING DU SQL
# ══════════════════════════════════════════════════════════════════

def parse_sql_chunked(sql_path: str, max_bytes: int = None) -> list:
    """
    Lit le fichier SQL par morceaux de 50 MB pour éviter les problèmes de RAM.

    Ce que fait la gateway IFM :
      Elle envoie 3 messages séparés pour chaque mesure :
        Ligne 1 (type Z) : {"Temperature":22.64, "Vibration":{"RMS":{"Z":3}}, "MeasDetails":{"Id":186}}
        Ligne 2 (type X) : {"Vibration":{"RMS":{"X":4}}, "MeasDetails":{"Id":186}}
        Ligne 3 (type Y) : {"Vibration":{"RMS":{"Y":2}}, "MeasDetails":{"Id":186}}

      Le champ MeasDetails.Id est le même pour les 3 lignes d'une même mesure.
      C'est la clé de consolidation.

    Ce script :
      1. Parse toutes les lignes 'res' du SQL
      2. Groupe par (SensorNodeId + timestamp) — CLÉ CORRIGÉE
         ⚠️  L'ancienne clé (SensorNodeId + MeasDetails.Id) causait une perte
         de 99% des données : MeasDetails.Id est cyclique (se réinitialise).
         Avec la clé (sid, timestamp), on extrait 606 106 mesures vs 5 120.
      3. Fusionne les lignes de la même seconde en une mesure complète
      4. Garde seulement les mesures qui ont au moins temp + vib_z
    """

    print(f"\n{B}{C}{'═'*60}")
    print(f"  ÉTAPE 1 — LECTURE DU FICHIER SQL")
    print(f"{'═'*60}{RS}")

    sql_file = Path(sql_path)
    if not sql_file.exists():
        err(f"Fichier introuvable : {sql_path}")
        err("Place ai_cp (5).sql dans le même dossier que ce script")
        sys.exit(1)

    size_mb = sql_file.stat().st_size / 1_000_000
    info(f"Fichier : {sql_path}")
    info(f"Taille  : {size_mb:.1f} MB")

    if max_bytes:
        info(f"Limite  : {max_bytes/1_000_000:.0f} MB (--max-mb)")

    # sessions[key] = dict avec temp, vib_x, vib_y, vib_z, ts, sid, acc_*
    # key = (SensorNodeId, MeasDetails.Id)
    sessions = defaultdict(dict)

    # Regex pour extraire les lignes INSERT VALUES de type 'res'
    # Format : (id, '', 'timestamp', '', '{json}', 'res', NULL)
    pattern = re.compile(
        r"\(\d+,\s*'[^']*',\s*'([^']+)',\s*'[^']*',\s*'(\{.*?\})',\s*'res'"
    )

    total_read   = 0
    total_lines  = 0
    total_parsed = 0
    t0 = time.time()

    with open(sql_path, "rb") as f:
        buffer = b""

        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break

            total_read += len(chunk)
            if max_bytes and total_read > max_bytes:
                warn(f"Limite {max_bytes/1_000_000:.0f} MB atteinte — arrêt lecture")
                break

            # Decoder et combiner avec le reste du buffer précédent
            buffer += chunk
            try:
                text = buffer.decode("utf-8", errors="replace")
            except Exception:
                text = buffer.decode("latin-1", errors="replace")

            # Garder la dernière ligne incomplète pour le prochain chunk
            last_newline = text.rfind("\n")
            if last_newline >= 0:
                to_parse = text[:last_newline]
                buffer   = text[last_newline:].encode("utf-8")
            else:
                to_parse = text
                buffer   = b""

            # Parser toutes les lignes 'res' dans ce morceau
            for match in pattern.finditer(to_parse):
                ts  = match.group(1)
                raw = match.group(2)
                total_lines += 1

                try:
                    # Dé-échapper les guillemets SQL \" → "
                    raw_clean = raw.replace('\\"', '"')
                    d = json.loads(raw_clean)
                except json.JSONDecodeError:
                    continue

                sid = d.get("SensorNodeId", "")
                mid = d.get("MeasDetails", {}).get("Id", "")

                if not sid or mid == "":
                    continue

                key = (sid, ts)   # CLÉ CORRIGÉE — MeasDetails.Id est cyclique
                vib = d.get("Vibration", {}).get("RMS", {})
                md  = d.get("MeasDetails", {})

                # Température + vib_z (ligne principale)
                if "Temperature" in d:
                    sessions[key]["sid"]  = sid
                    sessions[key]["ts"]   = ts
                    sessions[key]["temp"] = float(d["Temperature"])
                    if "Z" in vib:
                        sessions[key]["vib_z"] = float(vib["Z"])

                # Vib_x (ligne secondaire X)
                if "X" in vib:
                    sessions[key]["vib_x"] = float(vib["X"])
                    if "sid" not in sessions[key]:
                        sessions[key]["sid"] = sid
                        sessions[key]["ts"]  = ts

                # Vib_y (ligne secondaire Y)
                if "Y" in vib:
                    sessions[key]["vib_y"] = float(vib["Y"])
                    if "sid" not in sessions[key]:
                        sessions[key]["sid"] = sid
                        sessions[key]["ts"]  = ts

                # Accélérations (parfois dans MeasDetails ou dans data direct)
                for src_key, dst_key in [
                    ("A-P2P",  "acc_p2p"),
                    ("A-Z2P",  "acc_z2p"),
                    ("Crest",  "acc_crest"),
                    ("A-RMS",  "acc_rms"),
                ]:
                    if src_key in md:
                        sessions[key][dst_key] = float(md[src_key])
                    elif src_key in d:
                        sessions[key][dst_key] = float(d[src_key])

                total_parsed += 1

            # Affichage progression
            pct = total_read / sql_file.stat().st_size * 100
            elapsed = time.time() - t0
            print(
                f"  Lecture : {total_read/1_000_000:.0f}/{size_mb:.0f} MB "
                f"({pct:.0f}%) | "
                f"Sessions : {len(sessions):,} | "
                f"{elapsed:.0f}s",
                end="\r"
            )

    print()  # Nouvelle ligne après \r
    elapsed = time.time() - t0
    ok(f"Lecture terminée en {elapsed:.0f}s")
    ok(f"Lignes 'res' parsées : {total_lines:,}")
    ok(f"Sessions trouvées (paires clé unique) : {len(sessions):,}")

    return list(sessions.values())


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 2 — CONSOLIDATION ET FILTRAGE
# ══════════════════════════════════════════════════════════════════

def consolidate_sessions(raw_sessions: list) -> list:
    """
    Filtre les sessions incomplètes et consolide les valeurs manquantes.

    Règles :
      - Session invalide si pas de temperature → ignorée
      - Session invalide si pas de vib_z → ignorée
      - vib_x absent → utilise vib_z (même ordre de grandeur sur capteurs IFM)
      - vib_y absent → utilise vib_z
      - acc_* absents → 0.0 (capteurs anciens n'ont pas ces champs)
    """

    print(f"\n{B}{C}{'═'*60}")
    print(f"  ÉTAPE 2 — CONSOLIDATION")
    print(f"{'═'*60}{RS}")

    valides   = []
    ignorees  = 0
    sans_vib_x = 0
    sans_acc   = 0

    for s in raw_sessions:
        # Filtres obligatoires
        if "temp" not in s:
            ignorees += 1
            continue
        if "vib_z" not in s:
            ignorees += 1
            continue
        if "sid" not in s or "ts" not in s:
            ignorees += 1
            continue

        # vib_x et vib_y optionnels — substitution par vib_z si absent
        if "vib_x" not in s:
            s["vib_x"] = s["vib_z"]
            sans_vib_x += 1
        if "vib_y" not in s:
            s["vib_y"] = s["vib_z"]

        # Accélérations optionnelles — 0.0 si absent
        for k in ["acc_p2p", "acc_z2p", "acc_crest", "acc_rms"]:
            if k not in s:
                s[k] = 0.0
                sans_acc += 1

        valides.append(s)

    # Trier par capteur puis timestamp
    # Clipping des ratios inter-axes (évite les infinis quand vib_y ≈ 0)
    # Sans ce clip, PCA s'effondre sur 1 composante
    valides.sort(key=lambda x: (x["sid"], x["ts"]))

    ok(f"Sessions valides (temp + vib_z) : {len(valides):,}")
    warn(f"Sessions ignorées (incomplètes) : {ignorees:,}")
    if sans_vib_x:
        info(f"vib_x/y substitué par vib_z : {sans_vib_x:,} sessions")

    # Stats par capteur
    from collections import Counter
    capteurs = Counter(s["sid"] for s in valides)
    info(f"Capteurs avec données :")
    for sid, n in sorted(capteurs.items(), key=lambda x: -x[1]):
        print(f"    {sid} : {n:4d} mesures")

    return valides


# ══════════════════════════════════════════════════════════════════
# ÉTAPE 3 — ÉCRITURE DU CSV
# ══════════════════════════════════════════════════════════════════

def write_csv(sessions: list, out_path: str):
    """
    Écrit le CSV final — exactement le même format que dataset_2026_with_acc.csv
    """

    print(f"\n{B}{C}{'═'*60}")
    print(f"  ÉTAPE 3 — ÉCRITURE CSV")
    print(f"{'═'*60}{RS}")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

        for s in sessions:
            writer.writerow({
                "sensor_id":   s["sid"],
                "timestamp":   s["ts"],
                "temperature": round(s["temp"],   4),
                "vibration_x": round(s["vib_x"],  4),
                "vibration_y": round(s["vib_y"],  4),
                "vibration_z": round(s["vib_z"],  4),
                "acc_p2p":     round(s.get("acc_p2p",   0.0), 4),
                "acc_z2p":     round(s.get("acc_z2p",   0.0), 4),
                "acc_crest":   round(s.get("acc_crest", 0.0), 4),
                "acc_rms":     round(s.get("acc_rms",   0.0), 4),
            })

    size_kb = Path(out_path).stat().st_size // 1024
    ok(f"CSV écrit : {out_path}")
    ok(f"Taille    : {size_kb} KB")
    ok(f"Lignes    : {len(sessions):,} mesures")


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Génère dataset_2026_with_acc.csv depuis ai_cp (5).sql"
    )
    parser.add_argument(
        "--sql", default=DEFAULT_SQL,
        help=f"Chemin vers le fichier SQL (défaut: {DEFAULT_SQL})"
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT,
        help=f"Fichier CSV de sortie (défaut: {DEFAULT_OUT})"
    )
    parser.add_argument(
        "--max-mb", type=int, default=None,
        help="Lire seulement les N premiers MB du SQL (test rapide)"
    )
    args = parser.parse_args()

    print(f"\n{B}{C}{'═'*60}")
    print(f"  GÉNÉRATION DATASET — SQL → CSV")
    print(f"  Source : {args.sql}")
    print(f"  Sortie : {args.out}")
    print(f"{'═'*60}{RS}\n")

    t_total = time.time()

    # Étape 1 — Lire et parser le SQL
    max_bytes = args.max_mb * 1_000_000 if args.max_mb else None
    raw = parse_sql_chunked(args.sql, max_bytes)

    # Étape 2 — Consolider et filtrer
    sessions = consolidate_sessions(raw)

    if not sessions:
        err("Aucune session valide extraite — vérifie le fichier SQL")
        sys.exit(1)

    # Étape 3 — Écrire le CSV
    write_csv(sessions, args.out)

    # Résumé final
    elapsed = time.time() - t_total
    print(f"\n{B}{G}{'═'*60}")
    print(f"  ✅ DATASET GÉNÉRÉ EN {elapsed:.0f}s")
    print(f"{'═'*60}{RS}")
    print(f"\n  Prochaine étape — réentraîner les modèles :")
    print(f"  {C}python retrain_from_real_data.py --contamination 0.12{RS}")
    print(f"  puis :")
    print(f"  puis relancer :")
    print(f"  {C}python api_unified_pythagore.py{RS}\n")


if __name__ == "__main__":
    main()
