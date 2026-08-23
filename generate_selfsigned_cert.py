"""
generate_selfsigned_cert.py
============================
Génère un certificat TLS auto-signé pour tester HTTPS en local ou sur le
réseau interne d'une usine (sans nom de domaine public).

⚠️  Un certificat auto-signé chiffre bien le trafic (protège contre l'écoute
passive sur le réseau), mais ne prouve pas l'identité du serveur à un tiers :
le navigateur affichera un avertissement "connexion non sécurisée" tant que
le certificat n'est pas installé comme "approuvé" sur les postes clients.
Pour un vrai déploiement avec nom de domaine, remplacer par un certificat
Let's Encrypt (gratuit, https://certbot.eff.org/) ou un certificat émis par
l'autorité de certification interne de l'entreprise.

Usage :
    python generate_selfsigned_cert.py
    python generate_selfsigned_cert.py --cn 192.168.120.58 --days 825
"""

import argparse
import datetime
import ipaddress
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def generate(cn: str, out_dir: Path, days: int) -> None:
    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COUNTRY_NAME, "TN"),
        x509.NameAttribute(NameOID.ORGANIZATION_NAME, "Maintenance Predictive PFE"),
        x509.NameAttribute(NameOID.COMMON_NAME, cn),
    ])

    # SAN (Subject Alternative Name) -- essentiel : les navigateurs modernes
    # ignorent le CN seul et exigent une entrée SAN correspondante.
    san_entries = [x509.DNSName("localhost"), x509.DNSName(cn)]
    try:
        san_entries.append(x509.IPAddress(ipaddress.ip_address(cn)))
    except ValueError:
        pass  # cn n'est pas une IP, on garde juste le DNSName
    san_entries.append(x509.IPAddress(ipaddress.ip_address("127.0.0.1")))

    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(datetime.datetime.now(datetime.timezone.utc))
        .not_valid_after(datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=days))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .add_extension(x509.BasicConstraints(ca=True, path_length=None), critical=True)
        .sign(key, hashes.SHA256())
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    key_path  = out_dir / "server.key"
    cert_path = out_dir / "server.crt"

    key_path.write_bytes(key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.TraditionalOpenSSL,
        encryption_algorithm=serialization.NoEncryption(),
    ))
    cert_path.write_bytes(cert.public_bytes(serialization.Encoding.PEM))

    print(f"Certificat genere :")
    print(f"  Cle privee  -> {key_path}")
    print(f"  Certificat  -> {cert_path}")
    print(f"  CN / SAN    -> {cn}, localhost, 127.0.0.1")
    print(f"  Validite    -> {days} jours")
    print()
    print("Pour activer HTTPS, ajouter dans .env :")
    print(f"  TLS_CERT_FILE={cert_path.as_posix()}")
    print(f"  TLS_KEY_FILE={key_path.as_posix()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cn", default="localhost",
                         help="Common Name -- nom d'hote ou IP du serveur (defaut: localhost)")
    parser.add_argument("--out", default="certs", help="Dossier de sortie (defaut: certs/)")
    parser.add_argument("--days", type=int, default=825, help="Duree de validite en jours (defaut: 825)")
    args = parser.parse_args()

    generate(args.cn, Path(args.out), args.days)
