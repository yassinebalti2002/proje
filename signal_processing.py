"""
signal_processing.py
=====================
Pipeline de traitement du signal pour extraction de caractéristiques
fréquentielles et détection de défauts de roulements.

Fonctionnalités :
    - FFT et analyse spectrale (densité spectrale de puissance)
    - Analyse d'enveloppe (démodulation AM — détection défauts roulements)
    - Extraction features domaine fréquentiel (bande d'énergie, fréquences dominantes)
    - Fréquences caractéristiques des défauts roulements (BPFI, BPFO, BSF, FTF)
    - Transformée en ondelettes (CWT — Morlet)
    - Filtres passe-bande Butterworth pour isolation bandes critiques

Usage autonome :
    python signal_processing.py --demo

Intégration API :
    from signal_processing import extract_spectral_features, BearingFaultDetector
"""

import numpy as np
import warnings
from typing import Optional
from scipy import signal as scipy_signal
from scipy.fft import fft, fftfreq, rfft, rfftfreq
from scipy.signal import hilbert, butter, sosfilt, welch, find_peaks
from scipy.stats import kurtosis, skew

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  CONSTANTES ROULEMENTS INDUSTRIELS (IFM VVB001/VSE002)
#  Source : catalogue SKF 6205-2RS (roulement standard moteur industriel)
# ══════════════════════════════════════════════════════════════════════════════

class BearingGeometry:
    """Géométrie d'un roulement — calcul des fréquences de défaut."""

    def __init__(
        self,
        n_balls: int = 9,
        ball_diameter_mm: float = 7.94,
        pitch_diameter_mm: float = 38.5,
        contact_angle_deg: float = 0.0,
        name: str = "SKF 6205-2RS"
    ):
        self.n_balls = n_balls
        self.bd = ball_diameter_mm
        self.pd = pitch_diameter_mm
        self.ca = np.radians(contact_angle_deg)
        self.name = name

    def fault_frequencies(self, rpm: float) -> dict:
        """
        Calcule les fréquences caractéristiques de défauts à partir de la vitesse RPM.

        Returns dict avec BPFI, BPFO, BSF, FTF en Hz.
        """
        fr = rpm / 60.0  # Fréquence de rotation (Hz)
        ratio = (self.bd / self.pd) * np.cos(self.ca)

        bpfo = (self.n_balls / 2) * fr * (1 - ratio)   # Outer race defect
        bpfi = (self.n_balls / 2) * fr * (1 + ratio)   # Inner race defect
        bsf  = (self.pd / (2 * self.bd)) * fr * (1 - ratio**2)  # Ball spin
        ftf  = (fr / 2) * (1 - ratio)                  # Cage (fundamental train)

        return {
            "fr_hz":   round(fr, 4),
            "bpfo_hz": round(bpfo, 4),   # Bague extérieure
            "bpfi_hz": round(bpfi, 4),   # Bague intérieure
            "bsf_hz":  round(bsf, 4),    # Défaut bille
            "ftf_hz":  round(ftf, 4),    # Cage
        }


# Roulement par défaut pour les moteurs du banc d'essai
DEFAULT_BEARING = BearingGeometry(
    n_balls=9,
    ball_diameter_mm=7.94,
    pitch_diameter_mm=38.5,
    contact_angle_deg=0.0,
    name="SKF 6205-2RS"
)


# ══════════════════════════════════════════════════════════════════════════════
#  FONCTIONS UTILITAIRES
# ══════════════════════════════════════════════════════════════════════════════

def _validate_signal(sig: np.ndarray, min_len: int = 8) -> np.ndarray:
    """Valide et nettoie le signal d'entrée."""
    sig = np.asarray(sig, dtype=float)
    sig = sig[np.isfinite(sig)]
    if len(sig) < min_len:
        return np.zeros(min_len)
    return sig - np.mean(sig)  # Retirer la composante DC


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 4):
    """Filtre Butterworth passe-bande."""
    nyq = 0.5 * fs
    low = max(lowcut / nyq, 1e-6)
    high = min(highcut / nyq, 1 - 1e-6)
    if low >= high:
        return None
    return butter(order, [low, high], btype="band", output="sos")


def butter_lowpass(cutoff: float, fs: float, order: int = 4):
    """Filtre Butterworth passe-bas."""
    nyq = 0.5 * fs
    cut = min(cutoff / nyq, 1 - 1e-6)
    return butter(order, cut, btype="low", output="sos")


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSE FFT COMPLÈTE
# ══════════════════════════════════════════════════════════════════════════════

def compute_fft(sig: np.ndarray, fs: float = 1000.0) -> dict:
    """
    Calcule la FFT et le spectre de puissance unilatéral.

    Args:
        sig : Signal temporel (array 1D)
        fs  : Fréquence d'échantillonnage en Hz (défaut 1 kHz)

    Returns dict avec freqs, amplitudes, psd (Welch), peak_freq, spectral_features
    """
    sig = _validate_signal(sig)
    N = len(sig)

    # FFT unilatérale
    yf = rfft(sig * np.hanning(N))  # Fenêtrage Hanning
    xf = rfftfreq(N, 1.0 / fs)
    amplitudes = (2.0 / N) * np.abs(yf)

    # Densité spectrale de puissance (méthode Welch — plus stable)
    nperseg = min(256, N // 2)
    freqs_psd, psd = welch(sig, fs=fs, nperseg=nperseg)

    # Fréquence dominante
    dominant_idx = np.argmax(amplitudes[1:]) + 1
    peak_freq = float(xf[dominant_idx])
    peak_amp  = float(amplitudes[dominant_idx])

    # Énergie par bandes (Hz)
    bands = {
        "sub_10hz":   (0.1, 10),
        "10_50hz":    (10, 50),
        "50_200hz":   (50, 200),
        "200_500hz":  (200, 500),
        "500_1000hz": (500, 1000),
        "high_1khz":  (1000, fs / 2),
    }
    band_energies = {}
    total_energy = float(np.sum(amplitudes**2)) + 1e-9
    for band_name, (flo, fhi) in bands.items():
        mask = (xf >= flo) & (xf <= min(fhi, fs / 2))
        e = float(np.sum(amplitudes[mask]**2))
        band_energies[band_name] = round(e, 6)
        band_energies[f"{band_name}_ratio"] = round(e / total_energy, 6)

    # Centroïde spectral
    spectral_centroid = float(
        np.sum(xf * amplitudes**2) / (total_energy + 1e-9)
    )

    # Largeur de bande spectrale (RMS des fréquences)
    spectral_bandwidth = float(
        np.sqrt(np.sum((xf - spectral_centroid)**2 * amplitudes**2) / (total_energy + 1e-9))
    )

    # Entropie spectrale
    psd_norm = psd / (np.sum(psd) + 1e-9)
    spectral_entropy = float(-np.sum(psd_norm * np.log2(psd_norm + 1e-9)))

    # Aplatissement spectral
    spectral_flatness = float(
        np.exp(np.mean(np.log(psd + 1e-9))) / (np.mean(psd) + 1e-9)
    )

    return {
        "freqs":              xf.tolist(),
        "amplitudes":         amplitudes.tolist(),
        "freqs_psd":          freqs_psd.tolist(),
        "psd":                psd.tolist(),
        "peak_freq_hz":       round(peak_freq, 4),
        "peak_amplitude":     round(peak_amp, 6),
        "spectral_centroid":  round(spectral_centroid, 4),
        "spectral_bandwidth": round(spectral_bandwidth, 4),
        "spectral_entropy":   round(spectral_entropy, 4),
        "spectral_flatness":  round(spectral_flatness, 6),
        "band_energies":      band_energies,
        "total_energy":       round(total_energy, 6),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSE D'ENVELOPPE (démodulation AM — méthode Hilbert)
# ══════════════════════════════════════════════════════════════════════════════

def compute_envelope_analysis(
    sig: np.ndarray,
    fs: float = 1000.0,
    bandpass_low: float = 200.0,
    bandpass_high: float = 400.0
) -> dict:
    """
    Analyse d'enveloppe pour détection de défauts de roulements.

    Méthode :
    1. Filtrage passe-bande dans la bande de résonance structurale
    2. Calcul de l'enveloppe via transformée de Hilbert
    3. FFT de l'enveloppe (spectre d'enveloppe)
    4. Détection des fréquences de défaut dans le spectre d'enveloppe

    Args:
        sig           : Signal de vibration brut
        fs            : Fréquence d'échantillonnage (Hz)
        bandpass_low  : Borne basse filtre passe-bande (Hz)
        bandpass_high : Borne haute filtre passe-bande (Hz)
    """
    sig = _validate_signal(sig)

    # 1. Filtre passe-bande sur la bande de résonance
    sos = butter_bandpass(bandpass_low, bandpass_high, fs, order=4)
    if sos is not None:
        sig_filtered = sosfilt(sos, sig)
    else:
        sig_filtered = sig.copy()

    # 2. Enveloppe via Hilbert
    analytic = hilbert(sig_filtered)
    envelope = np.abs(analytic)

    # 3. Retirer la composante DC de l'enveloppe
    envelope_ac = envelope - np.mean(envelope)

    # 4. FFT de l'enveloppe
    N = len(envelope_ac)
    env_fft = rfft(envelope_ac * np.hanning(N))
    env_freqs = rfftfreq(N, 1.0 / fs)
    env_amps  = (2.0 / N) * np.abs(env_fft)

    # 5. Pics dans le spectre d'enveloppe
    peaks_idx, props = find_peaks(
        env_amps, height=np.mean(env_amps) + 2 * np.std(env_amps), distance=3
    )
    top_peaks = sorted(
        [(float(env_freqs[i]), float(env_amps[i])) for i in peaks_idx],
        key=lambda x: -x[1]
    )[:5]

    # 6. Indicateurs d'état
    env_kurtosis = float(kurtosis(envelope))
    env_rms      = float(np.sqrt(np.mean(envelope**2)))
    env_crest    = float(np.max(np.abs(envelope)) / (env_rms + 1e-9))

    return {
        "envelope_kurtosis":  round(env_kurtosis, 4),
        "envelope_rms":       round(env_rms, 4),
        "envelope_crest":     round(env_crest, 4),
        "envelope_freqs":     env_freqs.tolist(),
        "envelope_spectrum":  env_amps.tolist(),
        "top_peaks_hz":       top_peaks,
        "bandpass_hz":        [bandpass_low, bandpass_high],
    }


# ══════════════════════════════════════════════════════════════════════════════
#  DÉTECTEUR DE DÉFAUTS ROULEMENTS
# ══════════════════════════════════════════════════════════════════════════════

class BearingFaultDetector:
    """
    Détecte les défauts de roulements en comparant le spectre d'enveloppe
    aux fréquences caractéristiques théoriques (BPFI, BPFO, BSF, FTF).

    Usage :
        detector = BearingFaultDetector(bearing=DEFAULT_BEARING)
        result = detector.analyze(vib_signal, rpm=1450, fs=1000)
    """

    def __init__(self, bearing: BearingGeometry = None, tolerance_hz: float = 2.0):
        self.bearing = bearing or DEFAULT_BEARING
        self.tol = tolerance_hz  # Tolérance fréquentielle ±Hz

    def analyze(
        self,
        sig: np.ndarray,
        rpm: float = 1450.0,
        fs: float = 1000.0,
        harmonics: int = 3
    ) -> dict:
        """
        Analyse complète : FFT + enveloppe + correspondance fréquences défauts.

        Args:
            sig       : Signal de vibration (array 1D, en mg)
            rpm       : Vitesse de rotation du moteur (tr/min)
            fs        : Fréquence d'échantillonnage (Hz)
            harmonics : Nombre d'harmoniques à vérifier pour chaque défaut
        """
        sig = _validate_signal(sig)
        fault_freqs = self.bearing.fault_frequencies(rpm)

        # Analyse FFT globale
        fft_result = compute_fft(sig, fs)

        # Analyse d'enveloppe
        env_result = compute_envelope_analysis(
            sig, fs,
            bandpass_low=max(50, fault_freqs["bpfo_hz"] * 2),
            bandpass_high=min(fs / 2 - 1, fault_freqs["bpfi_hz"] * 4)
        )

        # Correspondance fréquences de défaut dans le spectre d'enveloppe
        env_freqs = np.array(env_result["envelope_freqs"])
        env_amps  = np.array(env_result["envelope_spectrum"])
        mean_amp  = np.mean(env_amps) + 1e-9

        fault_matches = {}
        fault_severity = {}

        for fault_name, base_freq in [
            ("BPFO", fault_freqs["bpfo_hz"]),
            ("BPFI", fault_freqs["bpfi_hz"]),
            ("BSF",  fault_freqs["bsf_hz"]),
            ("FTF",  fault_freqs["ftf_hz"]),
        ]:
            harmonic_energies = []
            for h in range(1, harmonics + 1):
                hf = base_freq * h
                mask = np.abs(env_freqs - hf) <= self.tol
                if np.any(mask):
                    peak_val = float(np.max(env_amps[mask]))
                    harmonic_energies.append(peak_val / mean_amp)
                else:
                    harmonic_energies.append(0.0)

            # Énergie cumulée des harmoniques (indicateur de présence du défaut)
            cumulative_snr = float(np.mean(harmonic_energies))
            fault_matches[fault_name] = {
                "base_freq_hz":   round(base_freq, 3),
                "harmonics_snr":  [round(v, 3) for v in harmonic_energies],
                "cumulative_snr": round(cumulative_snr, 3),
                "detected":       cumulative_snr > 2.5,  # Seuil SNR 2.5x bruit moyen
            }
            fault_severity[fault_name] = cumulative_snr

        # Niveau de sévérité global
        max_snr = max(fault_severity.values()) if fault_severity else 0.0
        env_kurtosis = env_result["envelope_kurtosis"]

        if env_kurtosis > 10 or max_snr > 5:
            severity = "CRITIQUE"
        elif env_kurtosis > 5 or max_snr > 3:
            severity = "MODÉRÉ"
        elif env_kurtosis > 3 or max_snr > 2:
            severity = "FAIBLE"
        else:
            severity = "OK"

        # Fault dominant
        dominant_fault = max(fault_severity, key=fault_severity.get) if fault_severity else "AUCUN"
        if fault_severity.get(dominant_fault, 0) < 2.5:
            dominant_fault = "AUCUN"

        return {
            "severity":           severity,
            "dominant_fault":     dominant_fault,
            "fault_frequencies":  fault_freqs,
            "fault_matches":      fault_matches,
            "envelope_kurtosis":  env_kurtosis,
            "envelope_crest":     env_result["envelope_crest"],
            "spectral_centroid":  fft_result["spectral_centroid"],
            "spectral_entropy":   fft_result["spectral_entropy"],
            "peak_freq_hz":       fft_result["peak_freq_hz"],
            "band_energies":      fft_result["band_energies"],
            "rpm_analyzed":       rpm,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  FEATURES SPECTRALES POUR LE MODÈLE ML
# ══════════════════════════════════════════════════════════════════════════════

def extract_spectral_features(
    vib_signal: list,
    fs: float = 1000.0,
    rpm: float = 1450.0
) -> dict:
    """
    Extrait un vecteur de 20 features spectrales à partir d'un signal de vibration.
    Compatible avec le format de features de l'API (peut être ajouté aux 25 features existantes).

    Args:
        vib_signal : Liste de valeurs de vibration (mg), minimum 16 points
        fs         : Fréquence d'échantillonnage estimée (Hz)
        rpm        : Vitesse de rotation estimée (tr/min)

    Returns dict avec 20 features spectrales nommées.
    """
    sig = np.array(vib_signal, dtype=float)
    sig = sig[np.isfinite(sig)]

    if len(sig) < 8:
        return {
            "spec_peak_freq":      0.0, "spec_centroid":        0.0,
            "spec_bandwidth":      0.0, "spec_entropy":         0.0,
            "spec_flatness":       0.0, "spec_total_energy":    0.0,
            "spec_band_low_ratio": 0.0, "spec_band_mid_ratio":  0.0,
            "spec_band_high_ratio":0.0, "env_kurtosis":         0.0,
            "env_rms":             0.0, "env_crest":            0.0,
            "bearing_bpfo_snr":    0.0, "bearing_bpfi_snr":     0.0,
            "bearing_bsf_snr":     0.0, "bearing_fault_severity": 0,
            "spectral_skewness":   0.0, "spectral_kurtosis":    0.0,
            "harmonic_ratio":      0.0, "noise_ratio":          0.0,
        }

    sig_centered = sig - np.mean(sig)

    # FFT
    fft_res = compute_fft(sig_centered, fs)

    # Enveloppe
    env_res = compute_envelope_analysis(sig_centered, fs)

    # Détecteur de défauts roulements
    detector = BearingFaultDetector()
    bearing_res = detector.analyze(sig_centered, rpm=rpm, fs=fs)

    # Features temporelles-fréquentielles supplémentaires
    amps = np.array(fft_res["amplitudes"])
    freqs = np.array(fft_res["freqs"])

    # Asymétrie et aplatissement spectraux
    psd_norm = amps**2 / (np.sum(amps**2) + 1e-9)
    spectral_mean = float(np.sum(freqs * psd_norm))
    spectral_var  = float(np.sum((freqs - spectral_mean)**2 * psd_norm))
    spectral_std  = np.sqrt(spectral_var + 1e-9)
    spectral_skew = float(np.sum(((freqs - spectral_mean) / spectral_std)**3 * psd_norm))
    spectral_kurt = float(np.sum(((freqs - spectral_mean) / spectral_std)**4 * psd_norm))

    # Ratio harmonique vs bruit
    be = fft_res["band_energies"]
    total_e = fft_res["total_energy"] + 1e-9
    harmonic_e = be.get("sub_10hz", 0) + be.get("10_50hz", 0)
    noise_e    = be.get("500_1000hz", 0) + be.get("high_1khz", 0)

    severity_map = {"OK": 0, "FAIBLE": 1, "MODÉRÉ": 2, "CRITIQUE": 3}
    severity_int = severity_map.get(bearing_res["severity"], 0)

    return {
        "spec_peak_freq":         round(fft_res["peak_freq_hz"], 4),
        "spec_centroid":          round(fft_res["spectral_centroid"], 4),
        "spec_bandwidth":         round(fft_res["spectral_bandwidth"], 4),
        "spec_entropy":           round(fft_res["spectral_entropy"], 4),
        "spec_flatness":          round(fft_res["spectral_flatness"], 6),
        "spec_total_energy":      round(total_e, 4),
        "spec_band_low_ratio":    round(be.get("sub_10hz_ratio", 0) + be.get("10_50hz_ratio", 0), 4),
        "spec_band_mid_ratio":    round(be.get("50_200hz_ratio", 0) + be.get("200_500hz_ratio", 0), 4),
        "spec_band_high_ratio":   round(be.get("500_1000hz_ratio", 0) + be.get("high_1khz_ratio", 0), 4),
        "env_kurtosis":           round(env_res["envelope_kurtosis"], 4),
        "env_rms":                round(env_res["envelope_rms"], 4),
        "env_crest":              round(env_res["envelope_crest"], 4),
        "bearing_bpfo_snr":       round(bearing_res["fault_matches"].get("BPFO", {}).get("cumulative_snr", 0), 4),
        "bearing_bpfi_snr":       round(bearing_res["fault_matches"].get("BPFI", {}).get("cumulative_snr", 0), 4),
        "bearing_bsf_snr":        round(bearing_res["fault_matches"].get("BSF",  {}).get("cumulative_snr", 0), 4),
        "bearing_fault_severity": severity_int,
        "spectral_skewness":      round(spectral_skew, 4),
        "spectral_kurtosis":      round(spectral_kurt, 4),
        "harmonic_ratio":         round(harmonic_e / total_e, 4),
        "noise_ratio":            round(noise_e / total_e, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  ANALYSE ONDELETTES (CWT Morlet — détection transitoires)
# ══════════════════════════════════════════════════════════════════════════════

def _ricker_wavelet(points: int, a: float) -> np.ndarray:
    """Ondelette de Ricker (Mexican hat) — compatible scipy 1.12+ (cwt supprimé)."""
    t = np.linspace(-a * 2, a * 2, points)
    factor = 2.0 / (np.sqrt(3 * a) * np.pi**0.25)
    return factor * (1 - (t / a)**2) * np.exp(-0.5 * (t / a)**2)


def compute_wavelet_energy(sig: np.ndarray, scales: int = 8) -> dict:
    """
    Décomposition CWT manuelle avec ondelette de Ricker (Mexican hat).
    Calcule l'énergie par échelle pour détecter les transitoires et impulsions.
    Compatible scipy >= 1.12 (cwt() retiré dans cette version).
    """
    sig = _validate_signal(sig)
    if len(sig) < 16:
        return {"wavelet_energies": [0.0] * scales, "wavelet_entropy": 0.0}

    widths = np.geomspace(1, max(2, len(sig) // 8), num=scales)
    energies = []

    for w in widths:
        wavelet = _ricker_wavelet(int(min(10 * w + 1, len(sig))), w)
        # Convolution via FFT pour performance
        n = len(sig) + len(wavelet) - 1
        fft_sig = np.fft.rfft(sig, n=n)
        fft_wav = np.fft.rfft(wavelet, n=n)
        conv = np.fft.irfft(fft_sig * fft_wav, n=n)[:len(sig)]
        e = float(np.sum(conv**2))
        energies.append(round(e, 4))

    total_e = sum(energies) + 1e-9
    probs   = [e / total_e for e in energies]
    w_entropy = float(-sum(p * np.log2(p + 1e-9) for p in probs))

    return {
        "wavelet_energies": energies,
        "wavelet_entropy":  round(w_entropy, 4),
        "wavelet_scales":   widths.tolist(),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  PIPELINE COMPLET
# ══════════════════════════════════════════════════════════════════════════════

def full_signal_pipeline(
    vib_signal: list,
    fs: float = 1000.0,
    rpm: float = 1450.0,
    include_raw_spectra: bool = False
) -> dict:
    """
    Pipeline complet : temporal + spectral + envelope + wavelet + bearing faults.

    Args:
        vib_signal          : Signal de vibration (mg)
        fs                  : Fréquence d'échantillonnage (Hz)
        rpm                 : Vitesse rotation moteur (tr/min)
        include_raw_spectra : Inclure les spectres bruts dans la sortie (volumineux)

    Returns dict consolidé avec toutes les features.
    """
    sig = np.array(vib_signal, dtype=float)

    # Features spectrales (20 scalaires)
    spectral = extract_spectral_features(sig, fs, rpm)

    # Ondelettes (8 niveaux)
    wavelet = compute_wavelet_energy(sig)

    # Analyse de roulements détaillée
    detector = BearingFaultDetector()
    bearing = detector.analyze(sig, rpm=rpm, fs=fs)

    result = {
        "spectral_features": spectral,
        "wavelet": {
            "wavelet_entropy":  wavelet["wavelet_entropy"],
            "wavelet_energies": wavelet["wavelet_energies"],
        },
        "bearing_analysis": {
            "severity":       bearing["severity"],
            "dominant_fault": bearing["dominant_fault"],
            "fault_matches":  bearing["fault_matches"],
        },
        "metadata": {
            "signal_length": len(sig),
            "fs_hz":         fs,
            "rpm":           rpm,
            "bearing_model": DEFAULT_BEARING.name,
        }
    }

    if include_raw_spectra:
        fft_res = compute_fft(sig - np.mean(sig), fs)
        result["raw_spectra"] = {
            "freqs":      fft_res["freqs"],
            "amplitudes": fft_res["amplitudes"],
            "psd":        fft_res["psd"],
            "freqs_psd":  fft_res["freqs_psd"],
        }

    return result


# ══════════════════════════════════════════════════════════════════════════════
#  DEMO / TEST AUTONOME
# ══════════════════════════════════════════════════════════════════════════════

def _demo():
    """Génère un signal synthétique et démontre toutes les analyses."""
    print("\n" + "="*70)
    print("  DEMO — signal_processing.py")
    print("  Pipeline traitement du signal — Détection défauts roulements")
    print("="*70)

    fs  = 1000.0   # Hz
    rpm = 1450.0   # tr/min
    T   = 2.0      # secondes
    t   = np.linspace(0, T, int(fs * T))

    # Signal synthétique : vibration normale + défaut BPFO + bruit
    bearing_geo = DEFAULT_BEARING
    fault_freqs = bearing_geo.fault_frequencies(rpm)
    bpfo = fault_freqs["bpfo_hz"]

    # Signal normal
    sig_normal = (
        300 * np.sin(2 * np.pi * (rpm / 60) * t)     # Harmonique rotation
        + 50 * np.sin(2 * np.pi * 100 * t)            # 100 Hz résiduel
        + 20 * np.random.randn(len(t))                # Bruit blanc
    )

    # Signal défectueux : impulsions périodiques à BPFO + modulation AM
    impulse_train = np.zeros(len(t))
    impulse_period = int(fs / bpfo)
    for i in range(0, len(t), impulse_period):
        decay = np.exp(-np.arange(min(50, len(t) - i)) / 10.0)
        end = min(i + 50, len(t))
        impulse_train[i:end] += 500 * decay[:end - i]

    sig_defect = sig_normal + impulse_train * (1 + 0.3 * np.sin(2 * np.pi * bpfo * t))

    print(f"\n  Roulement : {bearing_geo.name}")
    print(f"  Vitesse   : {rpm} tr/min")
    print(f"  BPFO      : {bpfo:.2f} Hz")
    print(f"  BPFI      : {fault_freqs['bpfi_hz']:.2f} Hz")

    print("\n  --- Signal NORMAL ---")
    res_normal = full_signal_pipeline(sig_normal.tolist(), fs=fs, rpm=rpm)
    sf = res_normal["spectral_features"]
    ba = res_normal["bearing_analysis"]
    print(f"  Entropie spectrale : {sf['spec_entropy']:.3f}")
    print(f"  Kurtosis enveloppe : {sf['env_kurtosis']:.3f}")
    print(f"  Sévérité roulement : {ba['severity']}")
    print(f"  Défaut dominant    : {ba['dominant_fault']}")

    print("\n  --- Signal DÉFECTUEUX (BPFO simulé) ---")
    res_defect = full_signal_pipeline(sig_defect.tolist(), fs=fs, rpm=rpm)
    sf = res_defect["spectral_features"]
    ba = res_defect["bearing_analysis"]
    print(f"  Entropie spectrale : {sf['spec_entropy']:.3f}")
    print(f"  Kurtosis enveloppe : {sf['env_kurtosis']:.3f}")
    print(f"  SNR BPFO           : {sf['bearing_bpfo_snr']:.3f}")
    print(f"  Sévérité roulement : {ba['severity']}")
    print(f"  Défaut dominant    : {ba['dominant_fault']}")

    print("\n  20 features spectrales (signal défectueux) :")
    for k, v in res_defect["spectral_features"].items():
        print(f"    {k:<30} : {v}")

    print("\n  Énergie ondelettes :", res_defect["wavelet"]["wavelet_energies"])
    print(f"  Entropie ondelettes : {res_defect['wavelet']['wavelet_entropy']:.4f}")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    import sys
    if "--demo" in sys.argv or len(sys.argv) == 1:
        _demo()
