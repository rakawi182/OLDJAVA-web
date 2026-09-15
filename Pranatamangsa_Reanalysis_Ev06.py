#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pranata Mangsa — Kalender Pertanian Tropis Berbasis Reanalisis Iklim
====================================================================

Modul ini mengimplementasikan kalender *Pranata Mangsa* (Jawa, 1855) dalam
kerangka kalibrasi multi-sumber: reanalisis meteorologi resolusi tinggi,
indeks oseanografi ENSO dan IOD, serta efemerida astronomis presisi.

Ruang lingkup
-------------
Modul menyatukan empat lapis informasi dalam satu struktur kalender:

1. **Lapisan meteorologis** — klimatologi 12 mangsa dan 4 musim dari
   reanalisis ERA5/ERA5-Land dan IFS HRES pada titik target (7.52°LS,
   112.57°BT, 28 m dpl), diinterpolasi dari dua stasiun dengan metode
   *Inverse Distance Weighting* (IDW, pangkat 2).

2. **Lapisan oseanografi** — koreksi empiris berbasis fase ENSO
   (Niño3.4 dari DUACS SLA dan NOAA OISST v2.1) dan IOD
   (*Dipole Mode Index* JMA/BoM), diterapkan sebagai delta per-mangsa
   yang diskalakan terhadap tumpang-tindih rentang *dopy*.

3. **Lapisan astronomis** — efemerida VSOP87D dengan presesi-nutasi
   IAU 2006/2000A dan ΔT HMNAO, mengoreksi tanggal-tanggal penanda
   tradisional (solstis, ekuinoks, zenith Matahari, fase Orion).

4. **Lapisan statistik** — *Hidden Markov Model* (HMM) 8-dimensi untuk
   klasifikasi rejim iklim, dan *Shift-Register Kalman Filter* (SR-EKF)
   dengan varians ARCH(1) untuk estimasi neraca air 30-hari.

Konvensi *dopy* (day-of-pranata-year)
-------------------------------------
Seluruh kalender direferensikan pada jangkar 22 Juni (tradisional) atau
21 Juni (presisi astronomis). Variabel `dopy` menyatakan offset hari
dari jangkar: `dopy = 0` untuk 22 Juni, `dopy = 182` untuk 21 Desember.
Rentang satu tahun-pranata adalah `[0, 365)`.

Arsitektur koreksi
------------------
Koreksi iklim pada kalender terkalibrasi dilakukan secara hierarkis:

    base_climatology(dopy_range)
        └─ + ΔENSO(phase, dopy_range)          [empiris, 1996–2025]
            └─ + ΔIOD(phase, dopy_range)       [bobot 0.30/0.50]

Setiap delta dihitung sebagai rata-rata berbobot tumpang-tindih antara
rentang dopy baru dengan rentang mangsa R30 — pendekatan *dopy-anchored*
yang menjamin konsistensi fisis antar-skenario.

Catatan versi
-------------
EV06 mengadopsi rekalibrasi *hourly* (1H) menggantikan basis 6-jam EV05
untuk field VPD, TCWV, cloud, dan sunshine. Perubahan utama:

- Sampling precision (SE) membaik 2.4× (ratio √6, terverifikasi).
- Bias diurnal 6-jam dieliminasi (VPD −5.8%, sun_h hingga −4.6 jam/hari).
- Tren VPD signifikan terdeteksi di Kapitu (+0.016 kPa/yr, p=0.014),
  Kasanga (+0.013 kPa/yr, p=0.070), Kasadasa (+0.024 kPa/yr, p=0.097).

Atribusi ilmiah lengkap tersedia di :data:`DATA_ATTRIBUTION` dan dapat
dicetak melalui :func:`print_data_attribution`.

Referensi
---------
.. [1] Hersbach, H., et al. (2020). The ERA5 global reanalysis.
       *QJRMS*, 146(730), 1999–2049. DOI:10.1002/qj.3803
.. [2] Muñoz-Sabater, J., et al. (2021). ERA5-Land: a state-of-the-art
       global reanalysis dataset for land applications. *ESSD*, 13(9).
       DOI:10.5194/essd-13-4349-2021
.. [3] Bretagnon, P., & Francou, G. (1988). VSOP87 planetary theories.
       *A&A*, 202, 309–315.
.. [4] Petit, G., & Luzum, B. (2010). *IERS Conventions (2010)*,
       IERS Technical Note 36.
.. [5] Ammarell, G. (1991). The astronomical basis of the Javanese
       Pranata Mangsa. *Journal of Southeast Asian Studies*.

Dependensi
----------
Wajib: `numpy`. Opsional: `pandas` (untuk modul nowcast), `scipy`
(untuk HMM berbasis `multivariate_normal`). Ketidakhadiran paket
opsional memicu *graceful degradation* ke fallback internal.
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from dataclasses import dataclass
from datetime import date, timedelta, datetime
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

if TYPE_CHECKING:
    import pandas as pd  # noqa: F811

try:
    from scipy.stats import multivariate_normal as _scipy_mvn_logpdf
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ══════════════════════════════════════════════════════════════════════
# §1  KONSTANTA TAMPILAN
# ══════════════════════════════════════════════════════════════════════

W: int = 70
IND: str = "  "

BULAN_ID: Dict[int, str] = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Mei", 6: "Jun",
    7: "Jul", 8: "Agu", 9: "Sep", 10: "Okt", 11: "Nov", 12: "Des",
}

BULAN_FULL: Dict[int, str] = {
    1: "Januari", 2: "Februari", 3: "Maret", 4: "April",
    5: "Mei", 6: "Juni", 7: "Juli", 8: "Agustus",
    9: "September", 10: "Oktober", 11: "November", 12: "Desember",
}

MONTHS_ID_SHORT: Dict[int, str] = BULAN_ID


# ── Koefisien fisis untuk koreksi delta (EV06) ─────────────────────
# Semua nilai dikalibrasi terhadap data hourly P1 2015–2025. Lihat
# §3 untuk justifikasi dan interval kepercayaan.
DVPD_DT: float = 0.075           # kPa/K     ∂VPD/∂T pada T ≈ 30°C, RH ≈ 72%
DVPD_DRH: float = -0.030         # kPa/%     ∂VPD/∂RH, dikalibrasi empiris
DTCWV_DT: float = -1.75          # kg/m²/K   regresi Maritime Continent
DSUN_DRAD: float = 0.20          # jam/(MJ/m²) konversi radiasi → sunshine
DCLOUD_DRAD: float = -5.0        # %/(MJ/m²)  tutupan awan vs radiasi
HJ_EXPONENT: float = 0.60        # hari_hujan ∝ hj_d^0.60 (Maritime Continent)
VPD_MIN_KPA: float = 0.0         # clamp bawah VPD
TCWV_MIN_KGM2: float = 0.0       # clamp bawah kolom uap air
SUN_H_MIN: float = 6.0           # clamp bawah sunshine (lintang 7°LS)
SUN_H_MAX: float = 13.0          # clamp atas sunshine


def fmt(d: date) -> str:
    """Format tanggal dalam konvensi Indonesia: ``DD Mmm YYYY``."""
    return f"{d.day:02d} {BULAN_ID[d.month]} {d.year}"


# ── Utilitas kotak tampilan ────────────────────────────────────────

def box_top(title: str = "") -> str:
    if not title:
        return "╔" + "═" * (W - 2) + "╗"
    inner = f"  {title}  "
    pad = W - 2 - len(inner)
    if pad < 0:
        inner = inner[:W - 2]
        pad = 0
    left = pad // 2
    return "╔" + "═" * left + inner + "═" * (pad - left) + "╗"


def box_mid() -> str:
    return "╠" + "═" * (W - 2) + "╣"


def box_bot() -> str:
    return "╚" + "═" * (W - 2) + "╝"


def box_row(text: str) -> str:
    """Render satu atau beberapa baris ber-border dengan wrapping otomatis.

    Teks yang melebihi lebar konten (W−6) di-wrap, sehingga border selalu
    presisi W kolom dan isi tidak pernah terpotong.
    """
    cw = W - 6
    s = str(text)
    if len(s) <= cw:
        return "║  " + s + " " * (cw - len(s)) + "  ║"
    lines = textwrap.wrap(s, width=cw) or [""]
    return "\n".join(
        "║  " + ln + " " * (cw - len(ln)) + "  ║"
        for ln in lines
    )


def thin_hbar(indent: int = 2) -> str:
    return " " * indent + "─" * (W - indent)


def sec_header(label: str, sub: str = "", dopy_range: str = "") -> None:
    """Cetak header seksi. Judul panjang di-wrap; dopy_range pindah baris."""
    right = f"[dopy: {dopy_range}]" if dopy_range else ""
    title = f"▌▌ {label.upper()}"
    if sub:
        title += f" — {sub}"
    gap = W - len(title) - len(right)
    print()
    if gap >= 1 or not right:
        line = title + (" " * max(1, gap)) + right if right else title
        print(line[:W])
    else:
        print(title[:W])
        print(right.rjust(W))
    print(thin_hbar(0))


def wline(label: str, value: str, lw: int = 12, indent: int = 6) -> str:
    """Baris berlabel dengan hanging indent, di-wrap pada lebar W."""
    pre = " " * indent + f"{label:<{lw}}: "
    sub_ind = " " * (indent + lw + 2)
    return textwrap.fill(value, width=W, initial_indent=pre,
                         subsequent_indent=sub_ind)


def wprint(label: str, value: str, lw: int = 12, indent: int = 6) -> None:
    print(wline(label, value, lw, indent))


def _wrap_ciri_line(raw: str, width: int) -> List[str]:
    """Wrap satu baris teks CIRI dengan hanging-indent bullet ``•``."""
    if not raw:
        return [""]
    stripped = raw.lstrip()
    if not stripped:
        return [""]
    leading = len(raw) - len(stripped)
    if stripped.startswith("•"):
        prefix = " " * leading + "• "
        content = stripped[1:].strip()
        return textwrap.wrap(
            content, width=width,
            initial_indent=prefix,
            subsequent_indent=" " * len(prefix),
        ) or [prefix.rstrip()]
    prefix = " " * leading
    return textwrap.wrap(
        stripped, width=width,
        initial_indent=prefix,
        subsequent_indent=prefix,
    ) or [prefix.rstrip()]


# ══════════════════════════════════════════════════════════════════════
# §2  ATRIBUSI SUMBER DATA
# ══════════════════════════════════════════════════════════════════════
#
# Setiap entri berisi metadata ilmiah lengkap: nama produk, deskripsi,
# institusi, sitasi, DOI, input, dan lisensi. Ini memenuhi standar
# *data provenance* untuk publikasi ilmiah dan memudahkan audit.

DATA_ATTRIBUTION: Dict[str, Dict[str, Optional[str]]] = {
    # ── Meteorologi ────────────────────────────────────────────────
    "era5": {
        "nama": "ERA5",
        "deskripsi": "Reanalisis global, resolusi 0.25° (~31 km), 1940–sekarang",
        "institusi": "ECMWF / Copernicus Climate Change Service (C3S)",
        "sitasi": ("Hersbach, H., et al. (2020). The ERA5 global reanalysis. "
                   "Quarterly Journal of the Royal Meteorological Society, "
                   "146(730), 1999–2049."),
        "doi": "10.1002/qj.3803",
        "lisensi": "Copernicus Licence — CC-BY 4.0",
    },
    "era5_land": {
        "nama": "ERA5-Land",
        "deskripsi": "Reanalisis permukaan darat, 0.1° (~9 km), 1950–sekarang",
        "institusi": "ECMWF / Copernicus Climate Change Service (C3S)",
        "sitasi": ("Muñoz-Sabater, J., et al. (2021). ERA5-Land: a "
                   "state-of-the-art global reanalysis dataset for land "
                   "applications. Earth System Science Data, 13(9)."),
        "doi": "10.5194/essd-13-4349-2021",
        "lisensi": "Copernicus Licence — CC-BY 4.0",
    },
    "ecmwf_ifs": {
        "nama": "ECMWF IFS (HRES)",
        "deskripsi": "Model NWP global, 9 km, 2017–sekarang",
        "institusi": "European Centre for Medium-Range Weather Forecasts",
        "sitasi": "ECMWF. (2024). IFS Documentation CY49r1.",
        "doi": None,
        "lisensi": "CC-BY 4.0",
    },
    "open_meteo": {
        "nama": "Open-Meteo",
        "deskripsi": "API agregator data meteorologi terbuka",
        "institusi": "Open-Meteo (open-source, non-komersial)",
        "sitasi": "Zippenfenig, P. (2023). Open-Meteo.com Weather API.",
        "doi": "10.5281/ZENODO.7970649",
        "lisensi": "CC-BY 4.0",
    },

    # ── ENSO ────────────────────────────────────────────────────────
    "aviso_duacs_sla": {
        "nama": "DUACS SLA Niño3.4 Index",
        "deskripsi": ("Indeks SLA terfilter, 85-hari rolling, "
                      "region Niño3.4 (5°S–5°N, 190°E–240°E), mingguan"),
        "institusi": "CNES / CLS — AVISO+ / DUACS",
        "sitasi": ("AVISO/DUACS. (2025). ENSO Ocean Indicator product "
                   "(vDT2024) [Data set]. CNES."),
        "doi": "10.24400/527896/A01-2025.008",
        "lisensi": "Copernicus Marine Licence",
    },
    "noaa_oisst_sst": {
        "nama": "NOAA OISST v2.1 SST Niño3.4",
        "deskripsi": ("Indeks SST terfilter, 85-hari rolling, "
                      "region Niño3.4, mingguan"),
        "institusi": "NOAA NCEI",
        "sitasi": ("Huang, B., et al. (2021). Improvements of DOISST "
                   "Version 2.1. Journal of Climate, 34, 2923–2939."),
        "doi": "10.1175/JCLI-D-20-0166.1",
        "lisensi": "NOAA Open Data",
    },

    # ── IOD ─────────────────────────────────────────────────────────
    "jma_iod": {
        "nama": "JMA Dipole Mode Index (DMI)",
        "deskripsi": ("DMI = SST anomali WIN − EIN; ambang ±0.4°C, "
                      "3-bulan running, periode Jun–Nov"),
        "institusi": "Japan Meteorological Agency",
        "sitasi": ("Saji, N. H., et al. (1999). A dipole mode in the "
                   "tropical Indian Ocean. Nature, 401, 360–363."),
        "doi": "10.1038/43854",
        "lisensi": "JMA Open Data",
    },
    "bom_iod": {
        "nama": "BoM IOD Index",
        "deskripsi": "IOD dari ERSSTv5/HadISST, ambang ±0.4°C",
        "institusi": "Australian Bureau of Meteorology",
        "sitasi": "Australian Bureau of Meteorology. (2025). IOD monitoring.",
        "doi": None,
        "lisensi": "CC-BY 4.0",
    },

    # ── Astronomi ───────────────────────────────────────────────────
    "vsop87d": {
        "nama": "VSOP87D",
        "deskripsi": ("Solusi analitik gerak planet, variabel heliosentrik "
                      "sferis, ekliptika tanggal. Akurasi ~1″ untuk 1800–2200."),
        "institusi": "IMCCE — Observatoire de Paris",
        "sitasi": ("Bretagnon, P., & Francou, G. (1988). Planetary theories "
                   "in rectangular and spherical variables: VSOP87 solution. "
                   "Astronomy & Astrophysics, 202, 309–315."),
        "doi": None,
        "lisensi": "IMCCE Open Data",
    },
    "iers2010": {
        "nama": "IERS Conventions 2010",
        "deskripsi": ("Presesi-nutasi IAU 2006/2000A (P03 + Mathews 2002). "
                      "Akurasi sub-mikroarcsecond."),
        "institusi": "International Earth Rotation and Reference Systems Service",
        "sitasi": ("Petit, G., & Luzum, B. (eds.). (2010). IERS Conventions "
                   "(2010). IERS Technical Note No. 36."),
        "doi": None,
        "lisensi": "IERS Open Access",
    },
    "iau2006_precession": {
        "nama": "IAU 2006 Precession (P03)",
        "deskripsi": "Model presesi IAU 2006, diadopsi Resolusi IAU 2006",
        "institusi": "IAU Working Group on Precession",
        "sitasi": ("Wallace, P. T., & Capitaine, N. (2006). Precession-"
                   "nutation procedures consistent with IAU 2006 "
                   "resolutions. A&A, 459(3), 981–985."),
        "doi": "10.1051/0004-6361:20065897",
        "lisensi": "CC-BY 4.0",
    },
    "hmnao_deltat": {
        "nama": "HMNAO ΔT Polynomials",
        "deskripsi": "Tabel ΔT (TT−UT1), −720 s.d. 2019, ekstrapolasi ke 2100",
        "institusi": "HM Nautical Almanac Office, UK Hydrographic Office",
        "sitasi": ("HM Nautical Almanac Office. (2020). Polynomial "
                   "Coefficients for ΔT and LOD: Version 2020."),
        "doi": None,
        "lisensi": "HMNAO Open Data",
    },
    "sofa": {
        "nama": "SOFA",
        "deskripsi": "Pustaka algoritma standar IAU untuk astronomi fundamental",
        "institusi": "IAU SOFA Center",
        "sitasi": ("Hohenkerk, C. Y. (2011). Standards of Fundamental "
                   "Astronomy. Scholarpedia, 6(1), 11404."),
        "doi": "10.4249/scholarpedia.11404",
        "lisensi": "SOFA Licence (non-commercial)",
    },
}


def print_data_attribution(detail: str = "ringkas") -> None:
    """Cetak atribusi sumber data dalam format kotak.

    Parameters
    ----------
    detail : {'ringkas', 'lengkap'}, default 'ringkas'
        - ``'ringkas'`` — nama + institusi + DOI (1 baris per sumber).
        - ``'lengkap'`` — deskripsi, sitasi, input, catatan, lisensi.
    """
    print()
    print(box_top("ATRIBUSI SUMBER DATA"))
    print(box_row("Sumber ilmiah untuk seluruh data meteorologi, "
                  "oseanografi, dan iklim"))
    print(box_mid())

    kategori = [
        ("METEOROLOGI", ["era5", "era5_land", "ecmwf_ifs", "open_meteo"]),
        ("ENSO (Niño3.4)", ["aviso_duacs_sla", "noaa_oisst_sst"]),
        ("IOD", ["jma_iod", "bom_iod"]),
        ("ASTRONOMI", ["vsop87d", "iers2010", "iau2006_precession",
                       "hmnao_deltat", "sofa"]),
    ]

    for label, keys in kategori:
        print(box_row(""))
        print(box_row(f"  ── {label} ─────────────────────────────────"))
        for k in keys:
            a = DATA_ATTRIBUTION[k]
            if detail == "lengkap":
                print(box_row(""))
                print(box_row(f"  ▸ {a['nama']}"))
                for ln in textwrap.wrap(a["deskripsi"], width=W - 8,
                                        initial_indent="    ",
                                        subsequent_indent="    "):
                    print(box_row(ln))
                print(box_row(f"    Institusi : {a['institusi']}"))
                print(box_row(f"    Sitasi    : {a['sitasi']}"))
                if a.get("doi"):
                    print(box_row(f"    DOI       : https://doi.org/{a['doi']}"))
                print(box_row(f"    Lisensi   : {a['lisensi']}"))
            else:
                print(box_row(f"  {a['nama']:<32} {a['institusi']}"))
                if a.get("doi"):
                    print(box_row(f"  {'':<32} DOI: {a['doi']}"))
    print(box_row(""))
    print(box_bot())
    print()


# ══════════════════════════════════════════════════════════════════════
# §3  DATA DASAR PRANATA MANGSA TRADISIONAL
# ══════════════════════════════════════════════════════════════════════
#
# Sumber: Serat Pranata Mangsa, reformasi Paku Buwana VII (1855).
# 12 mangsa, masing-masing dengan durasi, bulan/tanggal mulai, dan
# keanggotaan musim. Struktur ini adalah fondasi seluruh kalender.

MANGSAS: List[Dict] = [
    {"no": 1, "nama": "Kasa", "bulan": 6, "tgl": 22, "durasi": 41},
    {"no": 2, "nama": "Karo", "bulan": 8, "tgl": 2, "durasi": 23},
    {"no": 3, "nama": "Katiga", "bulan": 8, "tgl": 25, "durasi": 24},
    {"no": 4, "nama": "Kapat", "bulan": 9, "tgl": 18, "durasi": 25},
    {"no": 5, "nama": "Kalima", "bulan": 10, "tgl": 13, "durasi": 27},
    {"no": 6, "nama": "Kanem", "bulan": 11, "tgl": 9, "durasi": 43},
    {"no": 7, "nama": "Kapitu", "bulan": 12, "tgl": 22, "durasi": 43},
    {"no": 8, "nama": "Kawolu", "bulan": 2, "tgl": 3, "durasi": 26},
    {"no": 9, "nama": "Kasanga", "bulan": 3, "tgl": 1, "durasi": 25},
    {"no": 10, "nama": "Kasadasa", "bulan": 3, "tgl": 26, "durasi": 24},
    {"no": 11, "nama": "Desta", "bulan": 4, "tgl": 19, "durasi": 23},
    {"no": 12, "nama": "Sada", "bulan": 5, "tgl": 12, "durasi": 41},
]

MUSIM_MEMBERS: Dict[str, List[int]] = {
    "Katiga": [1, 2, 3],
    "Labuh": [4, 5, 6],
    "Rendheng": [7, 8, 9],
    "Mareng": [10, 11, 12],
}

MUSIM_DESKRIPSI: Dict[str, str] = {
    "Katiga": "Kemarau Puncak",
    "Labuh": "Peralihan → Hujan",
    "Rendheng": "Musim Hujan Puncak",
    "Mareng": "Peralihan → Kemarau",
}

MUSIM_ORDER: List[str] = ["Katiga", "Labuh", "Rendheng", "Mareng"]

# ── Ciri fenologis per mangsa (bahasa Indonesia) ──────────────────
CIRI: Dict[int, str] = {
    1: ("Solstis Juni 21 Jun (λ☉=90°, dopy≈−0.8). Weluku/Orion terbit "
        "fajar ~25 Jun (dopy≈3.1, 3 hr setelah solstis). Awal tahun "
        "pertanian; membersihkan lahan, tanah mengering."),
    2: "Pohon randu/kapuk mulai merekah. Tanah retak. Pengolahan lahan kering.",
    3: "Puncak kemarau, sumur mengering. Panen palawija (jagung, kacang).",
    4: ("Burung gelatik di sawah, manyar membuat sarang. Angin mulai "
        "berubah ke barat. Ekuinoks September (23 Sep, dopy≈92.9) jatuh "
        "di akhir Kapat."),
    5: ("Zenith Matahari I: 12 Okt (δ☉=−7.52°, dopy≈112.4, 1 hr lebih "
        "awal dari tradisional). Awal hujan. Pleiades terlihat di senja. "
        "Embun beracun."),
    6: ("Weluku/Orion Acronychal Rise (pertama terlihat di senja): ~5 Des "
        "(dopy≈166.2). Kulminasi tengah malam Orion: ~8 Des (dopy≈168.9). "
        "Hujan lebat. Menabur benih padi."),
    7: ("Solstis Desember (21 Des, dopy≈182.7) secara astronomis berada di "
        "Mangsa-6 Kanem, bukan Kapitu. Tradisi menaruh Solstis di Kapitu "
        "karena batas lama (22 Des). Pleiades setinggi pecat sawad (~50°). "
        "Memindah bibit padi ke sawah."),
    8: ("Transplantasi selesai. Pleiades kulminasi di senja. Padi tumbuh. "
        "Zenith Matahari II (1 Mar, dopy≈252.5) dan Orion Kulminasi Senja "
        "(~1 Mar) jatuh di Kawolu untuk skenario R30/R10."),
    9: ("Zenith Matahari II (dopy≈252.5) & Orion Evening Heliacal "
        "Culmination (~26 Feb–1 Mar) secara astronomis berada di Mangsa-8 "
        "Kawolu untuk R30/R10; hanya pada skenario TRAD jatuh di Kasanga. "
        "Ekuinoks Maret 20 Mar (dopy≈271.8) di akhir Kasanga (R30). "
        "Jangkrik berbunyi. Padi berbulir."),
    10: "Hujan reda, angin timur. Panen raya mulai.",
    11: ("Orion terbalik di barat (terbenam awal). Kapuk mekar. "
         "Hutang dilunasi."),
    12: ("Orion terakhir terlihat senja: ~18 Jun (dopy≈361.9, acronychal "
         "set). Perkiraan tradisional 4 Jun (heliacal set fajar, dopy≈347) "
         "berbeda definisi (+15 hr). Panen selesai. Masa bera (Apit Lemah)."),
}

# ── Candra (teks Jawa asli dari Serat Pranata Mangsa) ─────────────
CIRI_JAWA: Dict[int, str] = {
    1: ("Sotya murca ing êmbanan, punika candranipun măngsa kasa = I "
        "mangsanipun gêgodhongan sami gogrog, kêkajêngan sami paruthul, "
        "têgêsipun: sotya murca ing êmbanan = sêsotya coplok saking ing "
        "êmbanan, gêgodhongan kaupamèkakên: sêsotya, uwit kaupamèkakên: "
        "êmbananipun."),
    2: ("Bantala rêngka, candranipun măngsa kalih = II têgêsipun: bantala "
        "rêngka = siti bênthèt, bantala = siti, rêngka = bênthèt, punika "
        "mangsanipun siti nêla."),
    3: ("Suta manut ing bapa, candranipun măngsa katiga = III têgêsipun: "
        "anak manut ing bapa, punika mangsanipun lung-lungan nurut lanjaran."),
    4: ("Waspa kumêmbêng jroning kalbu, candranipun măngsa sakawan = IV, "
        "têgêsipun: êluh kumêmbêng salêbêting manah, punika mangsanipun "
        "sumbêr pêpêt (= pêpêt sumbêr) êluh kadamêl upami: toya, manah: "
        "kadamêl upami: sumbêr."),
    5: ("Pancuran êmas sumawur ing jagad, candranipun măngsa gangsal = V, "
        "pancuran: kadamêl upami: jawah, sumawur: dhawahipun ing jawah."),
    6: ("Rasa mulya kasucian, candranipun măngsa kanêm = VI, mangsanipun "
        "wowohan nêdhêng."),
    7: ("Wisa kentar ing maruta, candranipun măngsa kapitu = VII, têgêsipun: "
        "wisa larut dening angin, punika mangsanipun kathah sêsakit."),
    8: ("Anjrah jroning kayun, candranipun măngsa kawolu = VIII, punika "
        "mangsanipun kucing gandhik."),
    9: ("Wêdharing wacana mulya, candranipun măngsa kasanga = IX, têgêsipun "
        "wêdaling wicantên linakung, punika mangsanipun gangsir sami "
        "ngênthir, garèng sami ngêrèng."),
    10: ("Gêdhong minêb jroning kalbu, candranipun măngsa sadasa = X, punika "
         "mangsanipun sato kewan sami mêtêng."),
    11: ("Sotya sinarawèdi, candranipun măngsa dhêstha = XI, punika "
         "mangsanipun pêksi sami ngloloh, têgêsipun: sêsotya, kadamêl "
         "upami: anaking pêksi, sinarawèdi = pinulasara, punika ngibaratipun "
         "dipun loloh."),
    12: ("Tirta sah saking sasana, candranipun măngsa sadha = XII, têgêsipun: "
         "toya pisah saking panggenan, punika măngsa badhidhing, tirta punika "
         "ngibarat kringêt, sasana ngibarat badan, dados awis-awis tiyang "
         "kringêtên, amargi saking asrêpipun."),
}

ANCHOR_MONTH: int = 6
ANCHOR_DAY: int = 22


def is_leap_year(y: int) -> bool:
    """True bila ``y`` adalah tahun kabisat dalam kalender Gregorian."""
    return (y % 4 == 0 and y % 100 != 0) or (y % 400 == 0)


def orig_dopy_table() -> Dict[int, int]:
    """Offset dopy kumulatif tiap mangsa pada kalender tradisional."""
    out, cum = {}, 0
    for m in MANGSAS:
        out[m["no"]] = cum
        cum += m["durasi"]
    return out


ORIG_DOPY: Dict[int, int] = orig_dopy_table()
ORIG_MUSIM_START: Dict[str, int] = {
    mu: ORIG_DOPY[mem[0]] for mu, mem in MUSIM_MEMBERS.items()
}
ORIG_MUSIM_START_NEXT: Dict[str, int] = {
    "Katiga": ORIG_MUSIM_START["Labuh"],
    "Labuh": ORIG_MUSIM_START["Rendheng"],
    "Rendheng": ORIG_MUSIM_START["Mareng"],
    "Mareng": 365 + ORIG_MUSIM_START["Katiga"],
}


# ══════════════════════════════════════════════════════════════════════
# §4  KALIBRASI ASTRONOMIS
# ══════════════════════════════════════════════════════════════════════
#
# Rata-rata 2020–2029 dari JRC_Ephemeris v5.0 (VSOP87D + IERS 2010 +
# ΔT HMNAO). Jangkar: −7.521951°LS, 112.566089°BT, 28 m dpl.

ASTRO_CALIB: Dict[str, Dict] = {
    "solstis_juni": {
        "mean_dopy": -0.81, "std_dopy": 0.27,
        "mean_month": 6, "mean_day": 21,
        "trad_dopy": 0, "delta": -0.81,
        "catatan": ("Solstis Juni 21 Jun ~11:00 WIB. Jangkar tradisional "
                    "22 Jun terlambat ~20 jam."),
    },
    "solstis_des": {
        "mean_dopy": 182.71, "std_dopy": 0.28,
        "mean_month": 12, "mean_day": 21,
        "trad_dopy": 184, "delta": -1.29,
        "catatan": ("Solstis Desember 21 Des (dopy 182.7). Secara astronomis "
                    "jatuh di Mangsa-6 Kanem untuk semua skenario modern, "
                    "bukan di Kapitu seperti pada tradisi."),
    },
    "equinox_maret": {
        "mean_dopy": 271.75, "std_dopy": 0.30,
        "mean_month": 3, "mean_day": 20,
        "trad_dopy": 273, "delta": -1.25,
        "catatan": ("Ekuinoks Maret 20 Mar ~11:00 WIB. Tidak dipakai sebagai "
                    "penanda mangsa dalam tradisi."),
    },
    "equinox_sept": {
        "mean_dopy": 92.85, "std_dopy": 0.27,
        "mean_month": 9, "mean_day": 23,
        "trad_dopy": 92, "delta": 0.85,
        "catatan": ("Ekuinoks September 23 Sep ~07:00 WIB. Jatuh di "
                    "Mangsa-4 Kapat pada skenario TRAD/R30."),
    },
    "zenith_I_okt": {
        "mean_dopy": 112.37, "std_dopy": 0.27,
        "mean_month": 10, "mean_day": 12,
        "trad_dopy": 113, "delta": -0.63,
        "catatan": ("Zenith I 12 Okt ~09:00 WIB, 0.6 hr lebih awal dari "
                    "tradisional 13 Okt. Penanda awal Kalima."),
    },
    "zenith_II_mar": {
        "mean_dopy": 252.46, "std_dopy": 0.27,
        "mean_month": 3, "mean_day": 1,
        "trad_dopy": 253, "delta": -0.54,
        "catatan": ("Zenith II 1 Mar ~10:00 WIB. Jatuh di Mangsa-8 Kawolu "
                    "untuk skenario R30/R10."),
    },
    "orion_helrise": {
        "mean_dopy": 3.11, "std_dopy": 0.40,
        "mean_month": 6, "mean_day": 25,
        "trad_dopy": 0, "delta": 3.11,
        "catatan": ("Orion heliacal rise 25 Jun — 3 hr setelah solstis. "
                    "Tradisi 22 Jun kurang tepat karena presesi."),
    },
    "orion_evening_rise": {
        "mean_dopy": 166.16, "std_dopy": 0.46,
        "mean_month": 12, "mean_day": 5,
        "trad_dopy": 167, "delta": -0.84,
        "catatan": ("Orion Acronychal Rise 5 Des (~18:30 WIB). Konsisten "
                    "dengan tradisional 6 Des."),
    },
    "orion_evening_culm": {
        "mean_dopy": 252.5, "std_dopy": 0.40,
        "mean_month": 3, "mean_day": 1,
        "trad_dopy": 253, "delta": -0.5,
        "catatan": ("Orion Evening Heliacal Culmination ~26 Feb–1 Mar. "
                    "Ammarell (1991) Tbl.3: epoch 1850 = 26 Feb, kini ≈1 Mar."),
    },
    "orion_midnight_culm": {
        "mean_dopy": 168.91, "std_dopy": 0.40,
        "mean_month": 12, "mean_day": 8,
        "trad_dopy": 252, "delta": -83.09,
        "catatan": ("Orion kulminasi tengah malam ~8 Des (00:00 WIB). "
                    "Berbeda definisi dari evening heliacal culmination."),
    },
    "orion_acron_set": {
        "mean_dopy": 361.93, "std_dopy": 0.50,
        "mean_month": 6, "mean_day": 18,
        "trad_dopy": 347, "delta": 14.93,
        "catatan": ("Orion Acronychal Set 18 Jun (dopy 361.9). Tradisional "
                    "4 Jun (dopy 347) memakai definisi heliacal set fajar."),
    },
}


def astro_event_date(key: str, year: int) -> Optional[date]:
    """Tanggal Gregorian untuk peristiwa astro dalam tahun-pranata ``year``."""
    ev = ASTRO_CALIB.get(key)
    if not ev:
        return None
    month, day = ev["mean_month"], ev["mean_day"]
    actual_year = year if month >= 6 else year + 1
    try:
        return date(actual_year, month, day)
    except ValueError:
        return date(actual_year, month, min(day, 28))


def astro_delta_str(key: str) -> str:
    """Representasi singkat Δ hari peristiwa astro vs tradisional."""
    ev = ASTRO_CALIB.get(key)
    if not ev:
        return ""
    delta = ev["delta"]
    if abs(delta) < 0.5:
        return f"Δ={delta:+.1f} hr (konsisten tradisional)"
    arah = "lebih awal" if delta < 0 else "lebih lambat"
    return f"Δ={delta:+.1f} hr ({abs(delta):.1f} hr {arah} dari tradisional)"


# ── Ciri dasar per mangsa (tanpa penanda astro) ────────────────────
CIRI_BASE: Dict[int, str] = {
    1: "Awal tahun pertanian; membersihkan lahan, tanah mengering.",
    2: "Pohon randu/kapuk mulai merekah. Tanah retak. Pengolahan lahan kering.",
    3: "Puncak kemarau, sumur mengering. Panen palawija (jagung, kacang).",
    4: "Burung gelatik di sawah, manyar membuat sarang. Angin mulai berubah.",
    5: "Awal hujan. Pleiades terlihat di senja. Embun beracun.",
    6: "Hujan lebat. Menabur benih padi.",
    7: "Pleiades setinggi pecat sawad (~50°). Memindah bibit padi ke sawah.",
    8: "Transplantasi selesai. Pleiades kulminasi di senja. Padi tumbuh.",
    9: "Jangkrik berbunyi. Padi berbulir.",
    10: "Hujan reda, angin timur. Panen raya mulai.",
    11: "Orion terbalik di barat (terbenam awal). Kapuk mekar. Hutang dilunasi.",
    12: "Panen selesai. Masa bera (Apit Lemah).",
}

ASTRO_LABEL: Dict[str, str] = {
    "solstis_juni": "Solstis Juni (λ☉=90°)",
    "solstis_des": "Solstis Desember (λ☉=270°)",
    "equinox_maret": "Ekuinoks Maret (λ☉=0°)",
    "equinox_sept": "Ekuinoks September (λ☉=180°)",
    "zenith_I_okt": "Zenith Matahari I (δ☉=−7.52°)",
    "zenith_II_mar": "Zenith Matahari II (δ☉=−7.52°)",
    "orion_helrise": "Orion Heliacal Rise (terbit fajar)",
    "orion_evening_rise": "Orion Acronychal Rise (terbit senja)",
    "orion_evening_culm": "Orion Kulminasi Senja",
    "orion_midnight_culm": "Orion Kulminasi Tengah Malam",
    "orion_acron_set": "Orion Acronychal Set (terbenam senja)",
}


def astro_events_in_range(dopy_start: float, dopy_end: float) -> List[str]:
    """Daftar kunci peristiwa astro yang jatuh dalam rentang dopy.

    Parameters
    ----------
    dopy_start, dopy_end : float
        Batas rentang dalam *day-of-pranata-year*. Rentang half-open
        ``[start, end)`` — tetapi pencarian dilakukan inklusif pada
        kelipatan 365 untuk mengakomodasi siklus tahunan.

    Returns
    -------
    list of str
        Kunci peristiwa, terurut menaik berdasarkan ``mean_dopy``.
    """
    out: List[str] = []
    s = dopy_start % 365
    e = dopy_end % 365
    for key, ev in ASTRO_CALIB.items():
        d = ev["mean_dopy"] % 365
        if s <= e:
            if s <= d <= e:
                out.append(key)
        else:
            if d >= s or d <= e:
                out.append(key)
    return sorted(out, key=lambda k: ASTRO_CALIB[k]["mean_dopy"])


def build_ciri(scenario_key: str, mangsa_no: int,
               dopy_start: float, dopy_end: float) -> str:
    """Bangun deskripsi fenologi + penanda astro untuk satu mangsa.

    Penanda astro digabung inline dengan separator `` | `` agar dapat
    di-wrap rapi oleh :func:`textwrap.fill`. Sebelumnya penanda ditulis
    sebagai bullet multi-baris, tetapi konvensi itu mudah terpotong di
    tepi konsol dengan lebar terbatas.

    Parameters
    ----------
    scenario_key : str
        Kunci skenario (``'R30'``, ``'R10'``, ...). Hanya dipakai untuk
        *namespace*; penempatan astro murni berdasarkan rentang dopy.
    mangsa_no : int
        Nomor mangsa 1–12, untuk lookup teks dasar fenologi.
    dopy_start, dopy_end : float
        Rentang dopy aktual mangsa ini pada skenario yang dipilih.

    Returns
    -------
    str
        Teks ciri lengkap dengan penanda astro inline.
    """
    base = CIRI_BASE.get(mangsa_no, "")
    events = astro_events_in_range(dopy_start, dopy_end)
    if not events:
        return base
    parts = []
    for key in events:
        ev = ASTRO_CALIB[key]
        label = ASTRO_LABEL.get(key, key)
        tgl = f"{ev['mean_day']:02d} {MONTHS_ID_SHORT[ev['mean_month']]}"
        parts.append(f"{label} {tgl} (dopy≈{ev['mean_dopy']:.1f})")
    return base + "  |  " + "  |  ".join(parts)


# ══════════════════════════════════════════════════════════════════════
# §5  SKENARIO KALIBRASI & KLIMATOLOGI EMPIRIS
# ══════════════════════════════════════════════════════════════════════

CALIB_SCENARIOS: Dict[str, Dict] = {
    "R30": {
        "label": "Normal Iklim Terkini (1996–2025, 30 th)",
        "musim_start": {"Katiga": 19, "Labuh": 94, "Rendheng": 208, "Mareng": 286},
        "catatan": ("Skenario UTAMA yang direkomendasikan untuk pemakaian "
                    "sehari-hari saat ini."),
    },
    "ALL": {
        "label": "Rata-rata Seluruh Data (1950–2025, 76 th)",
        "musim_start": {"Katiga": 32, "Labuh": 91, "Rendheng": 187, "Mareng": 286},
        "catatan": ("Baseline jangka panjang — menunjukkan pergeseran "
                    "vs. kondisi terkini."),
    },
    "R10": {
        "label": "10 Tahun Terakhir (2016–2025)",
        "musim_start": {"Katiga": 12, "Labuh": 131, "Rendheng": 185, "Mareng": 305},
        "catatan": ("Basis 10 tahun terakhir. Lebih responsif terhadap tren "
                    "iklim terkini; sampel kecil sehingga uncertainty lebih "
                    "besar dari R30."),
    },
    "ELNINO": {
        "label": "Tahun El Niño (ASO Niño3.4 ≥ +0.5)",
        "musim_start": {"Katiga": 2, "Labuh": 133, "Rendheng": 217, "Mareng": 288},
        "catatan": ("Katiga jauh lebih panjang & lambat berakhir "
                    "(rata-rata 131 hr vs 88 hr tradisional)."),
    },
    "LANINA": {
        "label": "Tahun La Niña (ASO Niño3.4 ≤ −0.5)",
        "musim_start": {"Katiga": 35, "Labuh": 79, "Rendheng": 210, "Mareng": 284},
        "catatan": ("Katiga jauh lebih pendek (44 hr); musim hujan datang "
                    "lebih awal."),
    },
    "NETRAL": {
        "label": "Tahun ENSO Netral",
        "musim_start": {"Katiga": 11, "Labuh": 102, "Rendheng": 189, "Mareng": 288},
        "catatan": "Paling mendekati pola ALL — kondisi tanpa ENSO kuat.",
    },
}

DEFAULT_SCENARIO: str = "R30"

# ── Klimatologi per mangsa (R30 1996–2025, ERA5/Land/IFS, IDW P1+P2) ─
# Format: (hj, hj_d, hhr, et0, wb, sm, rh, tx, tn, angin, rad)
#   hj  = mm/musim · hj_d = mm/hari · hhr = hari hujan · et0 = mm/hari
#   wb  = neraca air (P − ET₀) mm/hari · sm = soil moisture m³/m³
#   rh  = % · tx/tn = °C · angin = km/j · rad = MJ/m²
METEO_MANGSA: Dict[int, Tuple] = {
    1: (23, 0.6, 5, 4.40, -3.75, 0.180, 67.1, 32.2, 21.5, 10.1, 19.9),
    2: (9, 0.5, 2, 4.98, -4.51, 0.150, 63.7, 33.2, 21.7, 10.7, 22.1),
    3: (17, 0.8, 3, 5.36, -4.51, 0.147, 62.3, 34.0, 22.3, 11.1, 23.3),
    4: (85, 2.8, 9, 5.31, -2.49, 0.183, 64.3, 34.3, 23.1, 10.8, 23.0),
    5: (244, 7.4, 21, 4.51, 2.88, 0.273, 72.5, 32.9, 23.7, 9.4, 20.4),
    6: (628, 12.7, 43, 3.61, 9.12, 0.369, 81.8, 30.6, 23.3, 9.6, 17.3),
    7: (559, 15.5, 35, 3.41, 12.12, 0.394, 84.4, 29.7, 23.1, 11.4, 16.8),
    8: (344, 15.6, 21, 3.55, 12.07, 0.398, 84.6, 29.9, 23.0, 10.0, 17.6),
    9: (238, 11.9, 18, 3.72, 8.16, 0.389, 83.5, 30.3, 23.0, 8.8, 18.3),
    10: (209, 7.8, 20, 3.75, 4.01, 0.364, 81.4, 30.6, 23.0, 8.2, 18.3),
    11: (97, 3.7, 12, 3.84, -0.10, 0.309, 77.0, 31.2, 23.0, 8.6, 18.2),
    12: (83, 1.9, 14, 3.87, -1.99, 0.249, 72.8, 31.4, 22.2, 9.0, 18.0),
}

METEO_MUSIM: Dict[str, Tuple] = {
    "Katiga": (75, 60, 0.8, 4.64, -3.84, 0.180, 66.1, 32.6, 21.8, 10.3, 20.8),
    "Labuh": (114, 978, 8.6, 4.32, 4.26, 0.290, 74.5, 32.3, 23.3, 9.9, 19.7),
    "Rendheng": (78, 1140, 14.6, 3.53, 11.09, 0.390, 84.2, 29.9, 23.0, 10.3, 17.4),
    "Mareng": (98, 453, 4.6, 3.79, 0.82, 0.310, 77.6, 31.1, 22.9, 8.5, 18.1),
}

METEO_BULANAN: Dict[int, Tuple] = {
    1: (441, 14.2, 3.47, 10.75, 0.390, 83.5, 30.0, 23.1, 10.8, 16.9),
    2: (457, 16.2, 3.46, 12.72, 0.400, 84.7, 29.7, 23.0, 11.3, 17.1),
    3: (407, 13.1, 3.67, 9.44, 0.390, 83.9, 30.2, 22.9, 9.2, 18.1),
    4: (241, 8.1, 3.73, 4.32, 0.370, 81.6, 30.6, 23.0, 8.2, 18.2),
    5: (113, 3.7, 3.83, -0.18, 0.300, 76.7, 31.2, 22.9, 8.6, 18.2),
    6: (55, 1.8, 3.83, -2.00, 0.250, 73.3, 31.4, 22.3, 9.0, 17.9),
    7: (30, 1.0, 4.19, -3.21, 0.200, 68.8, 31.8, 21.6, 9.7, 19.2),
    8: (14, 0.4, 4.78, -4.34, 0.160, 64.6, 32.8, 21.5, 10.4, 21.4),
    9: (29, 1.0, 5.36, -4.40, 0.150, 62.4, 34.0, 22.3, 11.1, 23.3),
    10: (110, 3.5, 5.19, -1.65, 0.200, 65.8, 34.1, 23.3, 10.7, 22.6),
    11: (261, 8.7, 4.28, 4.41, 0.300, 74.6, 32.5, 23.7, 9.0, 19.7),
    12: (402, 13.0, 3.60, 9.35, 0.370, 81.7, 30.7, 23.3, 9.4, 17.2),
}

# ── Klimatologi 6H-derived (rekalibrasi hourly, EV06) ──────────────
# Basis: Open-Meteo Best Match, ERA5/Land IFS HRES 1H, stasiun P1
# Periode: 2015–2025 (11 thn, 96.432 jam, coverage 100%)
# Metode (lihat header modul §Catatan versi EV06):
#   vpd   — rata² 24 jam · sun_h — sum(sunshine_duration)/3600
#   tcwv  — rata² 24 jam · cloud — rata² 24 jam
#   cloud_aft — rata² jam 12–18 WIB
#   sm_sh, sm_dp, sT_sh, sT_dp — dari EV05 6H IDW (belum di-update)
# Format: (vpd, tcwv, cloud, cloud_aft, sun_h, sm_sh, sm_dp, sT_sh, sT_dp)
METEO_MANGSA_6H: Dict[int, Tuple] = {
    1: (1.387, 35.4, 49, 55, 10.9, 0.154, 0.335, 26.9, 26.8),
    2: (1.597, 33.9, 48, 51, 11.0, 0.132, 0.321, 27.7, 27.0),
    3: (1.716, 36.1, 55, 59, 11.0, 0.126, 0.312, 28.4, 27.2),
    4: (1.621, 40.0, 66, 66, 10.9, 0.134, 0.301, 29.1, 27.5),
    5: (1.208, 47.0, 77, 78, 10.1, 0.271, 0.291, 28.4, 27.9),
    6: (0.649, 52.7, 90, 90, 8.6, 0.379, 0.291, 26.8, 27.9),
    7: (0.455, 54.0, 92, 93, 7.8, 0.394, 0.377, 26.2, 27.2),
    8: (0.451, 52.9, 87, 90, 8.5, 0.396, 0.404, 26.2, 26.9),
    9: (0.509, 52.4, 84, 87, 9.1, 0.386, 0.403, 26.5, 26.7),
    10: (0.644, 49.7, 76, 78, 9.6, 0.360, 0.395, 26.6, 26.7),
    11: (0.953, 45.4, 62, 65, 10.3, 0.313, 0.377, 26.9, 26.8),
    12: (1.041, 43.4, 60, 63, 10.3, 0.235, 0.356, 26.7, 26.8),
}

METEO_MUSIM_6H: Dict[str, Tuple] = {
    # Format: (vpd, tcwv, cloud, sun_h) — EV06 rekalibrasi hourly
    "Katiga": (1.531, 35.2, 50, 10.9),
    "Labuh": (1.062, 47.7, 80, 9.6),
    "Rendheng": (0.468, 53.3, 89, 8.4),
    "Mareng": (0.877, 46.2, 66, 10.1),
}

# ── Ekstrem absolut per mangsa (R30 1996–2025) ─────────────────────
# Format: (Tx_abs, Tn_abs) — °C. Nilai per-musim = agregasi max/min
# atas mangsa anggotanya.
METEO_MANGSA_EXTREME: Dict[int, Tuple[float, float]] = {
    1: (36.4, 16.3), 2: (36.7, 17.5), 3: (37.2, 17.3),
    4: (38.5, 18.5), 5: (39.3, 19.9), 6: (37.7, 20.3),
    7: (35.1, 20.1), 8: (33.6, 19.9), 9: (33.8, 18.2),
    10: (33.9, 19.4), 11: (34.9, 18.0), 12: (35.3, 17.1),
}

METEO_MUSIM_EXTREME: Dict[str, Tuple[float, float]] = {
    "Katiga": (37.2, 16.3),
    "Labuh": (39.3, 18.5),
    "Rendheng": (35.1, 18.2),
    "Mareng": (35.3, 17.1),
}

# ── Rentang dopy R30 per mangsa (basis interpolasi lintas-skenario) ─
R30_DOPY_RANGES: Dict[int, Tuple[float, float]] = {
    1: (19.00, 53.94), 2: (53.94, 73.55), 3: (73.55, 94.00),
    4: (94.00, 124.00), 5: (124.00, 156.40), 6: (156.40, 208.00),
    7: (208.00, 243.68), 8: (243.68, 265.26), 9: (265.26, 286.00),
    10: (286.00, 312.73), 11: (312.73, 338.34), 12: (338.34, 384.00),
}


# ══════════════════════════════════════════════════════════════════════
# §6  KOREKSI ENSO DAN IOD
# ══════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ClimateDelta:
    """Delta iklim terstruktur (menggantikan tuple posisional).

    Menggunakan :class:`dataclass` alih-alih tuple 6-field untuk
    menghindari kesalahan urutan indeks yang sulit dilacak saat kode
    diubah. Field-nya identik dengan layout komposit empiris.

    Attributes
    ----------
    tx : float
        Delta suhu maksimum harian (°C).
    tn : float
        Delta suhu minimum harian (°C).
    hj_d : float
        Delta curah hujan harian (mm/hari).
    et0 : float
        Delta evapotranspirasi referensi (mm/hari).
    rad : float
        Delta radiasi matahari (MJ/m²).
    rh : float
        Delta kelembapan relatif (%).
    """
    tx: float = 0.0
    tn: float = 0.0
    hj_d: float = 0.0
    et0: float = 0.0
    rad: float = 0.0
    rh: float = 0.0

    def __iter__(self):
        return iter((self.tx, self.tn, self.hj_d, self.et0, self.rad, self.rh))

    def __abs__(self) -> bool:
        return any(abs(x) > 1e-9 for x in self)


# ── Delta ENSO per mangsa (empiris ERA5/Land R30 1996–2025) ────────
# Klasifikasi tahun (ASO Niño3.4 MSLA/DUACS):
#   El Niño : 1997, 2002, 2004, 2006, 2009, 2015, 2018, 2023 (8 thn)
#   La Niña : 1998, 1999, 2007, 2008, 2010, 2011, 2016, 2017, 2020,
#             2024, 2025 (11 thn)
ENSO_DELTA: Dict[str, Dict[int, ClimateDelta]] = {
    "ELNINO": {
        1: ClimateDelta(+0.19, -0.40, -0.014, +0.084, +0.24, -2.0),
        2: ClimateDelta(-0.04, -0.62, -0.018, +0.031, +0.22, -1.5),
        3: ClimateDelta(+0.13, -0.58, -0.029, +0.168, +0.66, -1.8),
        4: ClimateDelta(+0.82, -0.57, -0.080, +0.584, +1.94, -5.6),
        5: ClimateDelta(+2.14, +0.38, -0.134, +0.835, +2.65, -8.7),
        6: ClimateDelta(+0.42, +0.21, -0.026, +0.173, +0.75, -1.1),
        7: ClimateDelta(-0.31, -0.12, +0.002, -0.108, -0.51, +0.2),
        8: ClimateDelta(-0.14, -0.22, -0.008, -0.004, +0.06, -0.1),
        9: ClimateDelta(+0.01, -0.12, -0.040, +0.078, +0.42, -0.7),
        10: ClimateDelta(+0.18, -0.13, -0.059, +0.140, +0.59, -1.5),
        11: ClimateDelta(+0.02, -0.24, -0.009, +0.055, +0.16, -0.9),
        12: ClimateDelta(+0.40, -0.27, -0.027, +0.261, +0.88, -3.2),
    },
    "LANINA": {
        1: ClimateDelta(-0.24, +0.35, +0.004, -0.108, -0.34, +2.0),
        2: ClimateDelta(-0.06, +0.38, +0.017, +0.010, -0.06, +1.0),
        3: ClimateDelta(-0.15, +0.38, +0.033, -0.110, -0.45, +1.8),
        4: ClimateDelta(-0.45, +0.38, +0.037, -0.243, -0.92, +3.0),
        5: ClimateDelta(-1.06, -0.14, +0.058, -0.456, -1.54, +4.7),
        6: ClimateDelta(-0.03, +0.09, -0.007, -0.003, -0.04, -0.1),
        7: ClimateDelta(+0.46, +0.25, -0.021, +0.144, +0.60, -0.6),
        8: ClimateDelta(+0.20, +0.23, +0.057, -0.022, -0.16, +0.3),
        9: ClimateDelta(+0.04, +0.25, +0.060, -0.157, -0.86, +0.7),
        10: ClimateDelta(-0.14, +0.27, +0.050, -0.176, -0.78, +1.6),
        11: ClimateDelta(-0.00, +0.23, +0.026, -0.072, -0.29, +1.0),
        12: ClimateDelta(-0.09, +0.24, +0.010, -0.110, -0.31, +1.4),
    },
    "NETRAL": {no: ClimateDelta() for no in range(1, 13)},
}

# ── Delta IOD per mangsa (DMI 1950–2025, detrend, ENSO-filtered) ───
# Ambang: SON DMI ≥ +0.40 (pIOD) / ≤ −0.40 (nIOD), BOM Australia.
# Filter: |ASO Niño3.4| ≥ 0.50 dikecualikan (ENSO-kuat).
# Sampel: pIOD 8 thn · nIOD 7 thn · Netral 41 thn.
# Mask: hanya mangsa 3–5 (overlap ≥ 15 hari dengan SON aktif).
IOD_DELTA: Dict[str, Dict[int, ClimateDelta]] = {
    "pIOD": {
        3: ClimateDelta(+0.534, -0.617, -0.943, +0.476, +1.175, -4.860),
        4: ClimateDelta(+1.278, -0.316, -2.224, +0.730, +1.690, -7.271),
        5: ClimateDelta(+2.345, +0.244, -4.980, +0.879, +2.461, -10.023),
    },
    "nIOD": {
        3: ClimateDelta(-0.841, +0.468, +2.365, -0.423, -1.459, +5.183),
        4: ClimateDelta(-0.831, +0.217, +2.437, -0.383, -1.149, +4.998),
        5: ClimateDelta(-0.363, -0.064, +1.149, -0.112, -0.319, +1.964),
    },
}
IOD_DELTA["NETRAL"] = {}
for _ph in ("pIOD", "nIOD"):
    for _no in range(1, 13):
        IOD_DELTA[_ph].setdefault(_no, ClimateDelta())

BOBOT_IOD_STANDALONE: float = 0.30
BOBOT_IOD_KOMBINASI: float = 0.50

ENSO_TO_IOD: Dict[str, str] = {
    "ELNINO": "pIOD",
    "LANINA": "nIOD",
}


def _weighted_delta_over_range(
    dopy_s: float, dopy_e: float,
    delta_table: Dict[int, ClimateDelta],
) -> ClimateDelta:
    """Delta terberat tumpang-tindih untuk rentang dopy.

    Logika *dopy-anchored*: delta untuk mangsa bernama sama (mis. Kalima)
    pada skenario ENSO bergantung pada rentang dopy aktual, bukan label.
    Ini menghindari artefak di mana mangsa yang sama punya delta berbeda
    hanya karena bergeser jendela waktunya.

    Parameters
    ----------
    dopy_s, dopy_e : float
        Rentang dopy aktual (half-open ``[s, e)``).
    delta_table : dict of {int: ClimateDelta}
        Delta per-nomor-mangsa R30 (basis bobot).

    Returns
    -------
    ClimateDelta
        Delta rata-rata berbobot; nol bila tidak ada irisan.
    """
    weights: Dict[int, float] = {}
    for no, (rs, re) in R30_DOPY_RANGES.items():
        ovl = max(0.0, min(dopy_e, re) - max(dopy_s, rs))
        if ovl > 0:
            weights[no] = ovl
    if not weights:
        return ClimateDelta()
    total = sum(weights.values())
    fields = ("tx", "tn", "hj_d", "et0", "rad", "rh")
    return ClimateDelta(*[
        sum(weights[no] / total * getattr(delta_table[no], f) for no in weights)
        for f in fields
    ])


def _enso_delta_for_dopy_range(
    dopy_s: float, dopy_e: float, enso_phase: str,
) -> ClimateDelta:
    """Delta ENSO untuk rentang dopy (fungsi *dopy-anchored*).

    Lihat :func:`_weighted_delta_over_range` untuk metodologi. Fungsi ini
    hanya menambahkan *dispatch* fase ENSO dan *short-circuit* bila fase
    bukan ``ELNINO``/``LANINA``.
    """
    if enso_phase not in ("ELNINO", "LANINA"):
        return ClimateDelta()
    return _weighted_delta_over_range(dopy_s, dopy_e, ENSO_DELTA[enso_phase])


def _iod_delta_for_dopy_range(
    dopy_s: float, dopy_e: float,
    iod_phase: str, enso_phase: str = "NETRAL",
) -> ClimateDelta:
    """Delta IOD terbobot, dengan skala bobot global.

    Bobot global dinaikkan dari 0.30 (standalone) menjadi 0.50 bila IOD
    dan ENSO saling menguatkan (El Niño + pIOD, atau La Niña + nIOD) —
    fenomena yang dikenal sebagai *Indian Ocean capacitor effect*.

    Parameters
    ----------
    dopy_s, dopy_e : float
        Rentang dopy aktual.
    iod_phase : {'pIOD', 'nIOD', 'NETRAL'}
    enso_phase : {'ELNINO', 'LANINA', 'NETRAL'}

    Returns
    -------
    ClimateDelta
        Delta sudah dikalikan bobot global.
    """
    if iod_phase not in ("pIOD", "nIOD"):
        return ClimateDelta()
    sinergis = (
        (enso_phase == "ELNINO" and iod_phase == "pIOD")
        or (enso_phase == "LANINA" and iod_phase == "nIOD")
    )
    bobot = BOBOT_IOD_KOMBINASI if sinergis else BOBOT_IOD_STANDALONE
    raw = _weighted_delta_over_range(dopy_s, dopy_e, IOD_DELTA[iod_phase])
    return ClimateDelta(*[bobot * x for x in raw])


# ══════════════════════════════════════════════════════════════════════
# §7  INTERPOLASI KLIMATOLOGI LINTAS-SKENARIO
# ══════════════════════════════════════════════════════════════════════

# Indeks field pada METEO_MANGSA (tuple 11-field):
_IDX_HJ, _IDX_HJD, _IDX_HHR, _IDX_ET0, _IDX_WB, _IDX_SM, _IDX_RH, \
    _IDX_TX, _IDX_TN, _IDX_ANGIN, _IDX_RAD = range(11)

_RATE_IDX: Tuple[int, ...] = (
    _IDX_HJD, _IDX_ET0, _IDX_WB, _IDX_SM, _IDX_RH, _IDX_TX, _IDX_TN,
    _IDX_ANGIN, _IDX_RAD,
)


def meteo_for_dopy_range(
    dopy_s: float, dopy_e: float,
    enso_phase: str = "NETRAL",
    iod_phase: str = "NETRAL",
) -> Tuple[Optional[Tuple], Optional[Tuple]]:
    """Klimatologi terinterpolasi untuk rentang dopy manapun.

    Menyediakan klimatologi per-mangsa yang konsisten untuk **semua**
    skenario kalender (R30/R10/ALL/ELNINO/LANINA/NETRAL) berdasarkan
    posisi dopy aktual, bukan nomor mangsa R30 yang *hardcoded*.

    Metode
    ------
    Untuk setiap mangsa R30 dalam :data:`R30_DOPY_RANGES`, hitung panjang
    irisan dengan rentang target ``[dopy_s, dopy_e)``. Bobotnya = irisan
    / total irisan. Field *rate* (hujan harian, ET₀, neraca air, dst.)
    dihitung sebagai rata-rata berbobot; field *intensif* (total hujan,
    hari hujan) diskalakan sesuai durasi rentang baru.

    Koreksi iklim diterapkan secara berurutan:

    1. Base climatology dari :data:`METEO_MANGSA` dan :data:`METEO_MANGSA_6H`.
    2. :func:`_enso_delta_for_dopy_range` bila fase ENSO ≠ NETRAL.
    3. :func:`_iod_delta_for_dopy_range` bila fase IOD ≠ NETRAL.

    Hari hujan diskalakan parsial dengan eksponen :data:`HJ_EXPONENT`
    (0.60) — hasil kalibrasi empiris bahwa perubahan curah hujan ENSO
    terdistribusi antara frekuensi dan intensitas, bukan murni frekuensi.

    Field 6H (VPD, TCWV, sun_h) dikoreksi memakai formulasi Tetens
    (VPD), regresi Maritime Continent (TCWV, tanda negatif), dan
    koefisien konversi radiasi→sunshine (lihat :data:`DVPD_DT` dkk.).

    Parameters
    ----------
    dopy_s, dopy_e : float
        Batas rentang dopy (half-open ``[s, e)``).
    enso_phase : {'ELNINO', 'LANINA', 'NETRAL'}, default 'NETRAL'
    iod_phase : {'pIOD', 'nIOD', 'NETRAL'}, default 'NETRAL'

    Returns
    -------
    m_tuple : tuple of 11 floats or None
        (hj, hj_d, hhr, et0, wb, sm, rh, tx, tn, angin, rad)
    m6h_tuple : tuple of 9 floats or None
        (vpd, tcwv, cloud, cloud_aft, sun_h, sm_sh, sm_dp, sT_sh, sT_dp)

    Notes
    -----
    Verifikasi **tidak boleh** membandingkan ``EN[no]`` vs ``R30[no]``
    karena batas dopy berbeda antar-skenario. Selalu gunakan rentang
    aktual dari :func:`build_calibrated_mangsa`.
    """
    weights: Dict[int, float] = {}
    for no, (rs, re) in R30_DOPY_RANGES.items():
        ovl = max(0.0, min(dopy_e, re) - max(dopy_s, rs))
        if ovl > 0:
            weights[no] = ovl
    if not weights:
        return None, None

    total_w = sum(weights.values())
    wn = {k: v / total_w for k, v in weights.items()}
    dur = max(dopy_e - dopy_s, 1.0)

    # ── (1) Interpolasi field daily ────────────────────────────────
    vals = [0.0] * 11
    hhr_frac = sum(
        wn[no] * METEO_MANGSA[no][_IDX_HHR]
        / (R30_DOPY_RANGES[no][1] - R30_DOPY_RANGES[no][0])
        for no in wn
    )
    vals[_IDX_HHR] = round(hhr_frac * dur)
    for fi in _RATE_IDX:
        vals[fi] = sum(wn[no] * METEO_MANGSA[no][fi] for no in wn)
    vals[_IDX_HJ] = round(vals[_IDX_HJD] * dur)

    hj_d_base = vals[_IDX_HJD]
    hhr_base = vals[_IDX_HHR]

    # ── (2) Koreksi ENSO pada field daily ──────────────────────────
    if enso_phase in ("ELNINO", "LANINA"):
        d = _enso_delta_for_dopy_range(dopy_s, dopy_e, enso_phase)
        hj_d_new = hj_d_base + d.hj_d
        vals[_IDX_HJD] = hj_d_new
        vals[_IDX_HJ] = round(hj_d_new * dur)

        if hj_d_base > 0.05:
            scale = hj_d_new / hj_d_base
            vals[_IDX_HHR] = max(0, round(hhr_base * (scale ** HJ_EXPONENT)))

        vals[_IDX_ET0] += d.et0
        vals[_IDX_RH] += d.rh
        vals[_IDX_TX] += d.tx
        vals[_IDX_TN] += d.tn
        vals[_IDX_RAD] += d.rad
        vals[_IDX_WB] = vals[_IDX_HJD] - vals[_IDX_ET0]

    m_tuple = tuple(vals)

    # ── (3) Interpolasi 6H ─────────────────────────────────────────
    m6h_vals = [
        sum(wn[no] * METEO_MANGSA_6H[no][fi] for no in wn)
        for fi in range(9)
    ]

    # ── (4) Koreksi ENSO pada 6H ───────────────────────────────────
    if enso_phase in ("ELNINO", "LANINA"):
        d = _enso_delta_for_dopy_range(dopy_s, dopy_e, enso_phase)
        d_tmean = 0.5 * (d.tx + d.tn)
        m6h_vals[0] = max(VPD_MIN_KPA,
                          m6h_vals[0] + DVPD_DT * d_tmean + DVPD_DRH * d.rh)
        m6h_vals[1] = max(TCWV_MIN_KGM2, m6h_vals[1] + DTCWV_DT * d_tmean)
        m6h_vals[4] = float(np.clip(m6h_vals[4] + DSUN_DRAD * d.rad,
                                    SUN_H_MIN, SUN_H_MAX))

    # ── (5) Koreksi IOD pada field daily ───────────────────────────
    if iod_phase in ("pIOD", "nIOD"):
        di = _iod_delta_for_dopy_range(dopy_s, dopy_e, iod_phase, enso_phase)
        if abs(di):
            v = list(m_tuple)
            hj_d_pre, hhr_pre = v[_IDX_HJD], v[_IDX_HHR]
            hj_d_new = hj_d_pre + di.hj_d
            v[_IDX_HJD] = hj_d_new
            v[_IDX_HJ] = round(hj_d_new * dur)
            if hj_d_pre > 0.05:
                scale = hj_d_new / hj_d_pre
                v[_IDX_HHR] = max(0, round(hhr_pre * (scale ** HJ_EXPONENT)))
            v[_IDX_ET0] += di.et0
            v[_IDX_WB] = v[_IDX_HJD] - v[_IDX_ET0]
            v[_IDX_RH] += di.rh
            v[_IDX_TX] += di.tx
            v[_IDX_TN] += di.tn
            v[_IDX_RAD] += di.rad
            m_tuple = tuple(v)

    # ── (6) Koreksi IOD pada 6H ────────────────────────────────────
    if iod_phase in ("pIOD", "nIOD"):
        di = _iod_delta_for_dopy_range(dopy_s, dopy_e, iod_phase, enso_phase)
        d_tmean_i = 0.5 * (di.tx + di.tn)
        if any(abs(x) > 1e-9 for x in (di.tx, di.tn, di.rad, di.rh)):
            m6h_vals[0] = max(VPD_MIN_KPA,
                              m6h_vals[0] + DVPD_DT * d_tmean_i
                              + DVPD_DRH * di.rh)
            m6h_vals[1] = max(TCWV_MIN_KGM2,
                              m6h_vals[1] + DTCWV_DT * d_tmean_i)
            m6h_vals[2] = float(np.clip(m6h_vals[2] + DCLOUD_DRAD * di.rad,
                                        0.0, 100.0))
            m6h_vals[3] = float(np.clip(m6h_vals[3] + DCLOUD_DRAD * di.rad,
                                        0.0, 100.0))
            m6h_vals[4] = float(np.clip(m6h_vals[4] + DSUN_DRAD * di.rad,
                                        SUN_H_MIN, SUN_H_MAX))

    return m_tuple, tuple(m6h_vals)


def _extreme_for_dopy_range(
    dopy_s: float, dopy_e: float,
) -> Optional[Tuple[float, float]]:
    """Ekstrem absolut (Tx_abs, Tn_abs) untuk rentang dopy.

    Karena max/min bersifat monoton, agregasi cukup menggunakan
    max atas Tx dan min atas Tn dari mangsa-mangsa R30 yang beririsan —
    tidak perlu pembobotan overlap.
    """
    tx_max, tn_min = -np.inf, +np.inf
    found = False
    for no, (rs, re) in R30_DOPY_RANGES.items():
        if max(0.0, min(dopy_e, re) - max(dopy_s, rs)) > 0:
            tx, tn = METEO_MANGSA_EXTREME[no]
            tx_max = max(tx_max, tx)
            tn_min = min(tn_min, tn)
            found = True
    return (tx_max, tn_min) if found else None


def _fmt_suhu(
    tx: float, tn: float,
    dopy_s: Optional[float] = None,
    dopy_e: Optional[float] = None,
) -> str:
    """Format baris suhu: rata-rata + (ekstrem absolut bila rentang diberikan)."""
    if dopy_s is not None and dopy_e is not None:
        ext = _extreme_for_dopy_range(dopy_s, dopy_e)
        if ext is not None:
            tx_a, tn_a = ext
            return (f"Tx̄ {tx:.1f}°C ({tx_a:.1f}°C) · "
                    f"Tn̄ {tn:.1f}°C ({tn_a:.1f}°C)")
    return f"Tx̄ {tx:.1f}°C · Tn̄ {tn:.1f}°C"


# ══════════════════════════════════════════════════════════════════════
# §8  PARAMETER HMM (4-D LEGACY & 8-D EV04/EV05)
# ══════════════════════════════════════════════════════════════════════
#
# Dua set parameter:
#   · 4-D  — legacy EV01, vektor (rain_30d, wb_30d, sm_30d, rh_30d)
#   · 8-D  — EV04/EV05, + (tcwv, dtr, cloud, sm-deep)
#
# Keduanya memakai 4 state iklim yang sama (lihat HMM_T_STATE).

HMM_T_STATE: Dict[int, str] = {
    0: "Katiga      — kering (kemarau puncak)",
    1: "Labuh/Mareng — transisi kering → sedang",
    2: "Rendheng    — hujan puncak (basah)",
    3: "Labuh/Mareng — transisi sedang → basah",
}

HMM_T_pi: List[float] = [0.0, 1.0, 0.0, 0.0]
HMM_T_A: List[List[float]] = [
    [0.9425, 0.0575, 0.0000, 0.0000],
    [0.0529, 0.8871, 0.0000, 0.0600],
    [0.0000, 0.0000, 0.9543, 0.0457],
    [0.0000, 0.0574, 0.0516, 0.8910],
]
HMM_T_means: List[List[float]] = [
    [-1.0662, -1.1139, -1.4893, -1.4005],
    [-0.7011, -0.6798, -0.4381, -0.4862],
    [1.2559, 1.2481, 0.9896, 1.0313],
    [0.1795, 0.2094, 0.5963, 0.5180],
]
HMM_T_covs: List[List[List[float]]] = [
    [[0.00333, 0.00337, 0.00666, 0.00613],
     [0.00337, 0.01149, 0.02119, 0.04043],
     [0.00666, 0.02119, 0.06652, 0.08894],
     [0.00613, 0.04043, 0.08894, 0.19196]],
    [[0.08677, 0.07744, 0.02992, 0.02422],
     [0.07744, 0.07613, 0.05928, 0.05991],
     [0.02992, 0.05928, 0.25335, 0.23600],
     [0.02422, 0.05991, 0.23600, 0.26591]],
    [[0.26552, 0.25858, 0.02309, 0.05224],
     [0.25858, 0.25298, 0.02295, 0.05313],
     [0.02309, 0.02295, 0.00357, 0.00743],
     [0.05224, 0.05313, 0.00743, 0.02512]],
    [[0.29796, 0.27850, 0.04217, 0.03377],
     [0.27850, 0.26355, 0.04823, 0.04586],
     [0.04217, 0.04823, 0.07873, 0.07706],
     [0.03377, 0.04586, 0.07706, 0.10589]],
]
HMM_T_mu: List[float] = [218.561, 96.474, 0.29663, 75.725]
HMM_T_sd: List[float] = [193.949, 210.002, 0.09978, 8.9105]

# ── 8-D (EV04/EV05) ─────────────────────────────────────────────────
HMM_T8_pi: List[float] = [0.0, 1.0, 0.0, 0.0]
HMM_T8_mu: List[float] = [
    218.50968, 94.95218, 0.27494, 75.28310,
    44.07459, 7.80143, 71.16249, 0.27615,
]
HMM_T8_sd: List[float] = [
    196.21025, 213.10715, 0.10689, 9.14500,
    8.20608, 2.05462, 17.96527, 0.09706,
]
HMM_T8_means: List[List[float]] = [
    [-0.99199, -0.96909, -0.98870, -0.93154, -1.12689, 1.09432, -1.08972, -0.58020],
    [0.01315, -0.04637, -0.21487, -0.34309, -0.06776, 0.11755, 0.13064, -0.53510],
    [1.22106, 1.20684, 0.92099, 0.97957, 0.94877, -0.93550, 1.05342, 0.86449],
    [-0.22744, -0.16457, 0.27292, 0.33157, 0.18566, -0.22910, -0.15607, 0.37753],
]
HMM_T8_A: List[List[float]] = [
    [0.98667, 0.01333, 0.00000, 0.00000],
    [0.00000, 0.99123, 0.00877, 0.00000],
    [0.00000, 0.00000, 0.98718, 0.01282],
    [0.01018, 0.00000, 0.00000, 0.98982],
]
HMM_T8_covs: List[List[List[float]]] = [
    [[ 0.03054,  0.03668,  0.08960,  0.08275,  0.07756, -0.08183,  0.04637,  0.05648],
     [ 0.03668,  0.04814,  0.12625,  0.11997,  0.09626, -0.10651,  0.06110,  0.09406],
     [ 0.08960,  0.12625,  0.44146,  0.38006,  0.22655, -0.25419,  0.17190,  0.40982],
     [ 0.08275,  0.11997,  0.38006,  0.37334,  0.24233, -0.27635,  0.15879,  0.36183],
     [ 0.07756,  0.09626,  0.22655,  0.24233,  0.33738, -0.32056,  0.13814,  0.10726],
     [-0.08183, -0.10651, -0.25419, -0.27635, -0.32056,  0.37360, -0.13584, -0.13805],
     [ 0.04637,  0.06110,  0.17190,  0.15879,  0.13814, -0.13584,  0.21501,  0.13741],
     [ 0.05648,  0.09406,  0.40982,  0.36183,  0.10726, -0.13805,  0.13741,  0.52406]],
    [[ 0.84378,  0.88919,  0.85516,  0.92098,  0.78767, -0.89999,  0.76891,  0.62175],
     [ 0.88919,  0.93970,  0.91392,  0.98682,  0.83864, -0.95997,  0.81964,  0.66598],
     [ 0.85516,  0.91392,  1.03046,  1.05695,  0.81745, -0.99372,  0.81979,  0.81700],
     [ 0.92098,  0.98682,  1.05695,  1.14499,  0.94168, -1.08900,  0.91625,  0.78475],
     [ 0.78767,  0.83864,  0.81745,  0.94168,  0.99774, -1.00594,  0.88420,  0.42368],
     [-0.89999, -0.95997, -0.99372, -1.08900, -1.00594,  1.15089, -0.93833, -0.62722],
     [ 0.76891,  0.81964,  0.81979,  0.91625,  0.88420, -0.93833,  0.91116,  0.49956],
     [ 0.62175,  0.66598,  0.81700,  0.78475,  0.42368, -0.62722,  0.49956,  0.98265]],
    [[ 0.29130,  0.28385,  0.10930,  0.09243,  0.03951, -0.02144,  0.07098,  0.17482],
     [ 0.28385,  0.27805,  0.10946,  0.09354,  0.04045, -0.02818,  0.07384,  0.17654],
     [ 0.10930,  0.10946,  0.11171,  0.06105, -0.02207,  0.00894,  0.00656,  0.19937],
     [ 0.09243,  0.09354,  0.06105,  0.05042,  0.01226, -0.01845,  0.02233,  0.10996],
     [ 0.03951,  0.04045, -0.02207,  0.01226,  0.09004, -0.05524,  0.04682, -0.03493],
     [-0.02144, -0.02818,  0.00894, -0.01845, -0.05524,  0.09147, -0.04639,  0.00206],
     [ 0.07098,  0.07384,  0.00656,  0.02233,  0.04682, -0.04639,  0.09373,  0.00445],
     [ 0.17482,  0.17654,  0.19937,  0.10996, -0.03493,  0.00206,  0.00445,  0.39624]],
    [[ 0.49767,  0.48123,  0.37605,  0.36813,  0.37565, -0.28695,  0.47199,  0.32410],
     [ 0.48123,  0.46752,  0.37697,  0.36632,  0.36476, -0.27978,  0.45925,  0.32669],
     [ 0.37605,  0.37697,  0.54927,  0.43013,  0.27650, -0.20948,  0.38532,  0.51309],
     [ 0.36813,  0.36632,  0.43013,  0.39387,  0.33140, -0.26910,  0.39826,  0.38236],
     [ 0.37565,  0.36476,  0.27650,  0.33140,  0.50766, -0.37421,  0.47227,  0.19054],
     [-0.28695, -0.27978, -0.20948, -0.26910, -0.37421,  0.34751, -0.36659, -0.13466],
     [ 0.47199,  0.45925,  0.38532,  0.39826,  0.47227, -0.36659,  0.59074,  0.29875],
     [ 0.32410,  0.32669,  0.51309,  0.38236,  0.19054, -0.13466,  0.29875,  0.53854]],
]


# ══════════════════════════════════════════════════════════════════════
# §9  TANGGAL & KALENDER
# ══════════════════════════════════════════════════════════════════════


def get_pranatamangsa_year_and_dopy(d: date) -> Tuple[int, int]:
    """Konversi tanggal Gregorian ke (tahun-pranata, dopy).

    Parameters
    ----------
    d : date

    Returns
    -------
    pyear : int
        Tahun mulai tahun-pranata (jangkar 22 Juni).
    dopy : int
        Offset hari dari jangkar, dalam ``[0, 365)``.
    """
    anchor = date(d.year, ANCHOR_MONTH, ANCHOR_DAY)
    if d >= anchor:
        return d.year, (d - anchor).days
    anchor = date(d.year - 1, ANCHOR_MONTH, ANCHOR_DAY)
    return d.year - 1, (d - anchor).days


def dopy_to_date(pyear: int, dopy: float) -> date:
    """Konversi (tahun-pranata, dopy) → tanggal Gregorian."""
    return date(pyear, ANCHOR_MONTH, ANCHOR_DAY) + timedelta(days=int(round(dopy)))


def build_calibrated_mangsa(scenario_key: str) -> Dict[int, float]:
    """Hitung dopy awal setiap mangsa untuk skenario terpilih.

    Metode *time-warp*: mangsa-mangsa R30 dipetakan linier ke rentang
    musim skenario baru. Setiap musim (Katiga/Labuh/Rendheng/Mareng)
    memiliki batas ``musim_start`` yang berbeda antar-skenario; proporsi
    internal mangsa dipertahankan.

    Parameters
    ----------
    scenario_key : str
        Kunci di :data:`CALIB_SCENARIOS`.

    Returns
    -------
    dict of {int: float}
        Pemetaan nomor-mangsa → dopy awal.
    """
    starts = CALIB_SCENARIOS[scenario_key]["musim_start"]
    starts_next = {
        "Katiga": starts["Labuh"],
        "Labuh": starts["Rendheng"],
        "Rendheng": starts["Mareng"],
        "Mareng": 365 + starts["Katiga"],
    }
    out: Dict[int, float] = {}
    for musim, members in MUSIM_MEMBERS.items():
        o_start = ORIG_MUSIM_START[musim]
        o_len = ORIG_MUSIM_START_NEXT[musim] - o_start
        n_start = starts[musim]
        n_len = starts_next[musim] - n_start
        for mno in members:
            frac = (ORIG_DOPY[mno] - o_start) / o_len
            out[mno] = n_start + frac * n_len
    return out


def build_calendar_tradisional(pyear: int) -> List[Dict]:
    """Bangun siklus tradisional 12 mangsa untuk tahun-pranata ``pyear``.

    Siklus dimulai pada 22 Juni ``pyear`` dan berakhir pada 21 Juni
    ``pyear + 1``. Mangsa yang jatuh pada bulan Januari–Mei (Gregorian)
    berada pada tahun ``pyear + 1``, sehingga pemeriksaan kabisat untuk
    Kawolu (Februari) menggunakan tahun tersebut.

    Parameters
    ----------
    pyear : int
        Tahun mulai siklus (tahun-pranata), bukan tahun Gregorian penuh.

    Returns
    -------
    list of dict
        12 entri mangsa dengan kunci ``no``, ``nama``, ``mulai``,
        ``akhir``, ``durasi``, ``musim``, ``ciri``, ``candra``.
    """
    calendar = []
    for m in MANGSAS:
        cy = (pyear if (m["bulan"] > ANCHOR_MONTH or
                        (m["bulan"] == ANCHOR_MONTH and m["tgl"] >= ANCHOR_DAY))
              else pyear + 1)
        start = date(cy, m["bulan"], m["tgl"])
        durasi = m["durasi"] + (1 if m["no"] == 8 and is_leap_year(cy) else 0)
        end = start + timedelta(days=durasi - 1)
        musim = next(mu for mu, mem in MUSIM_MEMBERS.items() if m["no"] in mem)
        calendar.append({
            "no": m["no"], "nama": m["nama"], "mulai": start,
            "akhir": end, "durasi": durasi, "musim": musim,
            "ciri": CIRI.get(m["no"], ""),
            "candra": CIRI_JAWA.get(m["no"], ""),
        })
    return calendar


@lru_cache(maxsize=32)
def _build_calendar_terkalibrasi_cached(
    pyear: int, scenario_key: str,
) -> Tuple[Dict, ...]:
    """Versi cached dari :func:`build_calendar_terkalibrasi`.

    `lru_cache` membutuhkan argumen hashable dan return hashable — oleh
    karena itu fungsi ini mengembalikan tuple of dict, dan wrapper publik
    mengonversinya kembali ke list mutable.
    """
    new_dopy = build_calibrated_mangsa(scenario_key)
    sorted_nos = sorted(new_dopy.keys())
    out: List[Dict] = []
    for i, no in enumerate(sorted_nos):
        m = next(x for x in MANGSAS if x["no"] == no)
        start = dopy_to_date(pyear, new_dopy[no])
        nxt_no = sorted_nos[(i + 1) % 12]
        nxt_dp = new_dopy[nxt_no] if nxt_no != 1 else 365 + new_dopy[1]
        end = dopy_to_date(pyear, nxt_dp) - timedelta(days=1)
        musim = next(mu for mu, mem in MUSIM_MEMBERS.items() if no in mem)
        ciri = build_ciri(scenario_key, no, new_dopy[no], nxt_dp - 1)
        out.append({
            "no": no, "nama": m["nama"], "mulai": start,
            "akhir": end, "durasi": (end - start).days + 1,
            "musim": musim, "ciri": ciri,
            "dopy_start": new_dopy[no], "dopy_end": nxt_dp - 1,
        })
    return tuple(out)


def build_calendar_terkalibrasi(
    pyear: int, scenario_key: str = DEFAULT_SCENARIO,
) -> List[Dict]:
    """Kalender terkalibrasi untuk tahun-pranata ``pyear`` dan skenario.

    Mengembalikan salinan mutable dari cache internal untuk mencegah
    modifikasi tak-sengaja merusak state global.
    """
    cached = _build_calendar_terkalibrasi_cached(pyear, scenario_key)
    return [dict(m) for m in cached]


def get_mangsa_by_date(
    tanggal: date, mode: str = "tradisional",
    scenario_key: str = DEFAULT_SCENARIO,
) -> Optional[Dict]:
    """Cari mangsa yang mencakup ``tanggal``.

    Menguji tahun-pranata ``pyear``, lalu ``pyear−1`` dan ``pyear+1``
    sebagai fallback untuk kasus tepi (mangsa yang melintasi batas
    tahun-pranata).
    """
    pyear, _ = get_pranatamangsa_year_and_dopy(tanggal)
    for py in (pyear, pyear - 1, pyear + 1):
        cal = (build_calendar_tradisional(py) if mode == "tradisional"
               else build_calendar_terkalibrasi(py, scenario_key))
        for m in cal:
            if m["mulai"] <= tanggal <= m["akhir"]:
                return m
    return None


def classify_enso_phase(aso_mean: float) -> str:
    """Klasifikasi fase ENSO dari rata-rata ASO Niño3.4."""
    if aso_mean >= 0.5:
        return "El Niño"
    if aso_mean <= -0.5:
        return "La Niña"
    return "Netral"


SCENARIO_FOR_PHASE: Dict[str, str] = {
    "El Niño": "ELNINO",
    "La Niña": "LANINA",
    "Netral": "NETRAL",
}


# ══════════════════════════════════════════════════════════════════════
# §10  NOWCAST — HMM + SR-EKF + INTEGRASI 6H/HOURLY
# ══════════════════════════════════════════════════════════════════════

DEFAULT_METEO_CSV = "open-meteo-7.49S112.54E28m.csv"
DEFAULT_METEO_CSV2 = "open-meteo-7.56S112.56E28m.csv"
DEFAULT_METEO_6H = "open-meteo-7.49S112.54E28m_6hour10yr.csv"
DEFAULT_METEO_6H_2 = "open-meteo-7.56S112.56E28m_6hour10yr.csv"
DEFAULT_METEO_HOURLY = "open-meteo-7.49S112.54E28m_hourly10yr.csv"
DEFAULT_ENSO_CSV = "Sst_nino34_index.csv"
DEFAULT_MSLA_CSV = "Msla_nino34_index.csv"
DEFAULT_IOD_DMI = "30yr_dmi_3rmean.txt"
DEFAULT_IOD_WEEKLY = "iod_1.txt"

LAT_TARGET, LON_TARGET = -7.521951, 112.566089
LAT_P1, LON_P1 = -7.486819, 112.53821
LAT_P2, LON_P2 = -7.5571175, 112.55735


def _log_mvn(X: np.ndarray, mean: Sequence[float],
             cov: Sequence[Sequence[float]]) -> np.ndarray:
    """Log-PDF multivariat Gauss; memakai SciPy bila tersedia, else fallback."""
    cov_arr = np.asarray(cov) + 1e-6 * np.eye(len(mean))
    if HAS_SCIPY:
        return _scipy_mvn_logpdf(X, mean=mean, cov=cov_arr)
    d = len(mean)
    diff = X - np.array(mean)
    inv = np.linalg.inv(cov_arr)
    _, logdet = np.linalg.slogdet(cov_arr)
    quad = (np.einsum("ij,jk,ik->i", diff, inv, diff)
            if diff.ndim == 2 else diff @ inv @ diff)
    return -0.5 * (d * np.log(2 * np.pi) + logdet + quad)


def hmm_causal_filter(
    Xz: np.ndarray,
    pi: Optional[Sequence[float]] = None,
    A: Optional[Sequence[Sequence[float]]] = None,
    means: Optional[Sequence[Sequence[float]]] = None,
    covs: Optional[Sequence] = None,
) -> np.ndarray:
    """Filter HMM *forward* (causal) — tanpa look-ahead bias.

    Menghitung probabilitas posterior state ``P(s_t | x_{1:t})`` untuk
    setiap langkah. Default parameter adalah 4-D legacy; untuk 8-D,
    lewatkan ``HMM_T8_*`` secara eksplisit.

    Returns
    -------
    probs : ndarray of shape (n, K)
        Baris = langkah waktu, kolom = state. Setiap baris berjumlah 1.
    """
    pi_arr = np.array(pi if pi is not None else HMM_T_pi) + 1e-12
    pi_arr /= pi_arr.sum()
    A_arr = np.array(A if A is not None else HMM_T_A)
    mu_arr = means if means is not None else HMM_T_means
    cv_arr = covs if covs is not None else HMM_T_covs
    n, K = len(Xz), len(mu_arr)
    logB = np.column_stack([_log_mvn(Xz, mu_arr[k], cv_arr[k]) for k in range(K)])
    alpha = pi_arr * np.exp(logB[0] - logB[0].max())
    alpha /= alpha.sum()
    probs = [alpha]
    for t in range(1, n):
        pred = alpha @ A_arr
        w = np.exp(logB[t] - logB[t].max())
        alpha = pred * w
        alpha /= alpha.sum()
        probs.append(alpha)
    return np.array(probs)


class _ARCH1:
    """Model varians ARCH(1) sederhana untuk inovasi Kalman.

    .. math:: R_t = \\max(r_{\\min}, \\omega + \\alpha \\epsilon_{t-1}^2)
    """

    def __init__(self, omega: float = 5.0, alpha: float = 0.3,
                 r_min: float = 1.0) -> None:
        self.omega = omega
        self.alpha = alpha
        self.r_min = r_min

    def update(self, innov: float) -> float:
        return max(self.r_min, self.omega + self.alpha * innov ** 2)


def sr_kf_local_trend(
    y: np.ndarray,
    q_level: float = 0.8, q_trend: float = 0.02,
    r_init: float = 400.0,
    arch_omega: float = 5.0, arch_alpha: float = 0.3,
) -> Tuple[float, float]:
    """Shift-Register Kalman Filter untuk model *local linear trend*.

    Memakai dekomposisi QR untuk menjaga stabilitas numerik matriks
    kovarians (formulasi *square-root*), dan varians observasi
    time-varying melalui :class:`_ARCH1`.

    Parameters
    ----------
    y : ndarray
        Deret waktu (mis. neraca air 30-hari).
    q_level, q_trend : float
        Varians noise proses untuk komponen level dan tren.
    r_init : float
        Varians observasi awal.
    arch_omega, arch_alpha : float
        Parameter ARCH(1) untuk varians observasi.

    Returns
    -------
    level, trend : tuple of float
        Estimasi level dan tren pada langkah terakhir.
    """
    F = np.array([[1.0, 1.0], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q_sqrt = np.diag([np.sqrt(q_level), np.sqrt(q_trend)])
    x = np.array([y[0], 0.0])
    S = np.eye(2) * np.sqrt(r_init)
    arch = _ARCH1(arch_omega, arch_alpha)
    for t in range(len(y)):
        x = F @ x
        compound = np.vstack((S.T @ F.T, Q_sqrt))
        _, R_qr = np.linalg.qr(compound, mode="reduced")
        S = R_qr[:2, :2].T
        y_pred = (H @ x).item()
        innov = y[t] - y_pred
        R_t = arch.update(innov)
        f = S.T @ H.T
        S_s = np.sqrt((f.T @ f).item() + R_t)
        K = (S @ f).flatten() / S_s
        x = x + K * (innov / S_s)
        alpha = 1.0 / (S_s * (S_s + np.sqrt(R_t)))
        S = np.tril(S - alpha * (S @ f @ f.T))
    return float(x[0]), float(x[1])


# ── Loader & IDW ────────────────────────────────────────────────────

def find_data_file(filename: str) -> Optional[str]:
    """Cari berkas data di direktori kerja atau di direktori modul."""
    if not filename:
        return None
    for folder in (".", os.path.dirname(os.path.abspath(__file__))):
        p = os.path.join(folder, filename)
        if os.path.exists(p):
            return p
    alt = filename.replace("_", ".")
    for folder in (".", os.path.dirname(os.path.abspath(__file__))):
        p = os.path.join(folder, alt)
        if os.path.exists(p):
            return p
    return None


def _read_openmeteo_csv(path: str) -> "pd.DataFrame":
    """Baca CSV Open-Meteo (skip 3 baris header) dan normalisasi kolom waktu."""
    df = pd.read_csv(path, skiprows=3)
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)


def _idw_weights(
    lat_t: float, lon_t: float,
    coords: Sequence[Tuple[float, float]], power: float = 2.0,
) -> List[float]:
    """Hitung bobot IDW; titik yang persis sama mengembalikan vektor satuan."""
    dists = [((c[0] - lat_t) ** 2 + (c[1] - lon_t) ** 2) ** 0.5 for c in coords]
    for i, d in enumerate(dists):
        if d < 1e-10:
            return [1.0 if j == i else 0.0 for j in range(len(coords))]
    w_raw = [1.0 / (d ** power) for d in dists]
    s = sum(w_raw)
    return [wi / s for wi in w_raw]


def _idw_merge(df1: "pd.DataFrame", df2: "pd.DataFrame",
               w1: float, w2: float, index_col: str = "time") -> "pd.DataFrame":
    """Gabungkan dua DataFrame dengan IDW; kolom non-shared diambil dari df2."""
    d1_ = df1.set_index(index_col)
    d2_ = df2.set_index(index_col)
    shared = [c for c in d1_.columns if c in d2_.columns]
    idx = d1_.index.union(d2_.index)
    out = pd.DataFrame(index=idx)
    for col in shared:
        v1 = d1_[col].reindex(idx)
        v2 = d2_[col].reindex(idx)
        both = v1.notna() & v2.notna()
        out.loc[both, col] = w1 * v1[both] + w2 * v2[both]
        only1 = v1.notna() & ~v2.notna()
        only2 = ~v1.notna() & v2.notna()
        out.loc[only1, col] = v1[only1]
        out.loc[only2, col] = v2[only2]
    for col in [c for c in d2_.columns if c not in shared]:
        out[col] = d2_[col].reindex(idx)
    return out.reset_index().rename(columns={"index": index_col})


def load_interpolated_meteo(
    csv1: str = DEFAULT_METEO_CSV, csv2: str = DEFAULT_METEO_CSV2,
    lat_t: float = LAT_TARGET, lon_t: float = LON_TARGET,
) -> Optional["pd.DataFrame"]:
    """Muat meteo harian dua stasiun, IDW-merge ke titik target."""
    if not HAS_PANDAS:
        return None
    path1, path2 = find_data_file(csv1), find_data_file(csv2)
    if path1 is None and path2 is None:
        return None
    w1, w2 = _idw_weights(lat_t, lon_t,
                          [(LAT_P1, LON_P1), (LAT_P2, LON_P2)])
    if path1 is None:
        return _read_openmeteo_csv(path2)
    if path2 is None:
        return _read_openmeteo_csv(path1)
    return _idw_merge(_read_openmeteo_csv(path1),
                      _read_openmeteo_csv(path2), w1, w2)


def _aggregate_to_daily(
    m: "pd.DataFrame", aft_slots: Sequence[int], sec_per_slot: int,
) -> "pd.DataFrame":
    """Agregasi sub-harian ke harian secara vektorisasi.

    Menghindari ``groupby.apply`` per-baris (lambat untuk 96k baris) dengan
    cara memakai reducer groupby langsung. Kolom output terbatas pada
    yang dibutuhkan pipeline nowcast.
    """
    m = m.copy()
    m["_date"] = pd.to_datetime(m["time"]).dt.normalize()
    m["_hour"] = pd.to_datetime(m["time"]).dt.hour
    g = m.groupby("_date")

    agg: Dict[str, "pd.Series"] = {}
    col_map = {
        "tcwv": "total_column_integrated_water_vapour (kg/m²)",
        "cloud_mean": "cloud_cover (%)",
        "sm28_100": "soil_moisture_28_to_100cm (m³/m³)",
        "sm_sh": "soil_moisture_0_to_7cm (m³/m³)",
        "sT_sh": "soil_temperature_0_to_7cm (°C)",
        "sT_dp": "soil_temperature_100_to_255cm (°C)",
        "vpd": "vapour_pressure_deficit (kPa)",
    }
    for out_name, src in col_map.items():
        if src in m.columns:
            agg[out_name] = g[src].mean()

    if "temperature_2m (°C)" in m.columns:
        agg["dtr"] = (g["temperature_2m (°C)"].max()
                      - g["temperature_2m (°C)"].min())

    if "cloud_cover (%)" in m.columns:
        aft = m[m["_hour"].isin(aft_slots)].groupby("_date")["cloud_cover (%)"].mean()
        agg["cloud_aft"] = aft

    if "sunshine_duration (s)" in m.columns:
        agg["sunshine_h"] = g["sunshine_duration (s)"].sum() / 3600.0

    if "shortwave_radiation (W/m²)" in m.columns:
        agg["sw_rad_MJ"] = (g["shortwave_radiation (W/m²)"].sum()
                            * sec_per_slot / 1e6)

    daily = pd.DataFrame(agg).reset_index().rename(columns={"_date": "time"})
    return daily


def load_interpolated_6h(
    csv1: str = DEFAULT_METEO_6H, csv2: str = DEFAULT_METEO_6H_2,
    lat_t: float = LAT_TARGET, lon_t: float = LON_TARGET,
    hourly_csv: str = DEFAULT_METEO_HOURLY,
) -> Optional["pd.DataFrame"]:
    """Muat data sub-harian, agregasi harian per stasiun, lalu IDW-merge.

    Prioritas sumber untuk P1:

    1. ``hourly_csv`` — akurasi tertinggi: ``sunshine_h`` benar (sum 24
       slot), ``dtr`` benar (peak siang + min pre-dawn), ``cloud_aft``
       dari slot 12–17, ``sw_rad_MJ`` integral 1 jam per slot.
    2. ``csv1`` — fallback 6H P1.

    P2 selalu dari ``csv2`` (6H). Strategi agregasi per-stasiun lalu
    merge menghindari artefak resample lintas-resolusi yang merusak
    akumulasi (sunshine, sw_rad).
    """
    if not HAS_PANDAS:
        return None
    path_h = find_data_file(hourly_csv)
    path1, path2 = find_data_file(csv1), find_data_file(csv2)
    use_hourly = path_h is not None
    if not use_hourly and path1 is None and path2 is None:
        return None
    w1, w2 = _idw_weights(lat_t, lon_t,
                          [(LAT_P1, LON_P1), (LAT_P2, LON_P2)])

    agg1 = None
    if use_hourly:
        agg1 = _aggregate_to_daily(_read_openmeteo_csv(path_h),
                                   aft_slots=list(range(12, 18)),
                                   sec_per_slot=3600)
    elif path1 is not None:
        agg1 = _aggregate_to_daily(_read_openmeteo_csv(path1),
                                   aft_slots=[12, 18], sec_per_slot=21600)
    agg2 = (_aggregate_to_daily(_read_openmeteo_csv(path2),
                                aft_slots=[12, 18], sec_per_slot=21600)
            if path2 is not None else None)

    if agg1 is None:
        return agg2
    if agg2 is None:
        return agg1
    return _idw_merge(agg1, agg2, w1, w2)


def _prep_8d_from_daily(df: "pd.DataFrame") -> Optional["pd.DataFrame"]:
    """Bangun vektor fitur 8-D dari DataFrame harian."""
    df = df.sort_values("time").reset_index(drop=True).copy()
    df["wb"] = (df["precipitation_sum (mm)"]
                - df["et0_fao_evapotranspiration (mm)"])
    df["rh_mean"] = ((df["relative_humidity_2m_max (%)"]
                      + df["relative_humidity_2m_min (%)"]) / 2)
    win = 30
    df["rain_30d"] = df["precipitation_sum (mm)"].rolling(win, min_periods=15).sum()
    df["wb_30d"] = df["wb"].rolling(win, min_periods=15).sum()
    df["sm_30d"] = (df["soil_moisture_0_to_7cm_mean (m³/m³)"]
                    .ffill().rolling(win, min_periods=15).mean())
    df["rh_30d"] = df["rh_mean"].rolling(win, min_periods=15).mean()
    df["tcwv_30d"] = df["tcwv"].ffill().rolling(win, min_periods=15).mean()
    df["dtr_30d"] = df["dtr"].ffill().rolling(win, min_periods=15).mean()
    df["cloud_30d"] = df["cloud_mean"].ffill().rolling(win, min_periods=15).mean()
    df["smd_30d"] = df["sm28_100"].ffill().rolling(win, min_periods=15).mean()
    need = ["rain_30d", "wb_30d", "sm_30d", "rh_30d",
            "tcwv_30d", "dtr_30d", "cloud_30d", "smd_30d"]
    df = df.dropna(subset=need).reset_index(drop=True)
    return df if len(df) > 0 else None


def live_nowcast(
    meteo_csv: str = DEFAULT_METEO_CSV,
    meteo_csv2: str = DEFAULT_METEO_CSV2,
    meteo_6h: str = DEFAULT_METEO_6H,
    meteo_6h2: str = DEFAULT_METEO_6H_2,
    enso_csv: str = DEFAULT_ENSO_CSV,
    msla_csv: str = DEFAULT_MSLA_CSV,
) -> Optional[Dict]:
    """Jalankan analisis nowcast lengkap: HMM + SR-EKF + ENSO/IOD.

    Mengembalikan dictionary hasil atau ``None`` bila dependensi
    (pandas, berkas meteo) tidak tersedia.
    """
    if not HAS_PANDAS:
        print("  [!] Modul pandas tidak tersedia — nowcast dilewati.")
        return None
    df = load_interpolated_meteo(meteo_csv, meteo_csv2)
    if df is None:
        print("  [!] Tidak ada file meteorologi harian — nowcast dilewati.")
        return None

    p1_ok = find_data_file(meteo_csv) is not None
    p2_ok = find_data_file(meteo_csv2) is not None
    interp_mode = ("IDW 2 stasiun" if (p1_ok and p2_ok)
                   else ("stasiun P1 saja" if p1_ok else "stasiun P2 saja"))

    df6 = load_interpolated_6h(meteo_6h, meteo_6h2)
    has_6h = df6 is not None and len(df6) > 0
    if has_6h:
        df = df.merge(df6, on="time", how="left")
    df = df.sort_values("time").reset_index(drop=True)

    df["wb"] = (df["precipitation_sum (mm)"]
                - df["et0_fao_evapotranspiration (mm)"])
    df["rh_mean"] = ((df["relative_humidity_2m_max (%)"]
                      + df["relative_humidity_2m_min (%)"]) / 2)
    win = 30
    df["rain_30d"] = df["precipitation_sum (mm)"].rolling(win, min_periods=15).sum()
    df["wb_30d"] = df["wb"].rolling(win, min_periods=15).sum()
    df["sm_30d"] = (df["soil_moisture_0_to_7cm_mean (m³/m³)"]
                    .ffill().rolling(win, min_periods=15).mean())
    df["rh_30d"] = df["rh_mean"].rolling(win, min_periods=15).mean()

    df_8d = _prep_8d_from_daily(df) if has_6h else None
    if df_8d is not None and len(df_8d) >= 60:
        tail8 = df_8d.tail(400).reset_index(drop=True)
        X8 = tail8[["rain_30d", "wb_30d", "sm_30d", "rh_30d",
                    "tcwv_30d", "dtr_30d", "cloud_30d", "smd_30d"]].values
        X8z = (X8 - np.array(HMM_T8_mu)) / np.array(HMM_T8_sd)
        probs = hmm_causal_filter(X8z, pi=HMM_T8_pi, A=HMM_T8_A,
                                  means=HMM_T8_means, covs=HMM_T8_covs)
        last_probs = probs[-1]
        last_date = tail8["time"].iloc[-1].date()
        hmm_mode = "8-D (EV04: +TCWV+DTR+cloud+SM-dalam)"
        df_trend = df_8d
    else:
        df4 = df.dropna(subset=["rain_30d", "wb_30d", "sm_30d", "rh_30d"]
                        ).reset_index(drop=True)
        tail4 = df4.tail(400).reset_index(drop=True)
        X4 = tail4[["rain_30d", "wb_30d", "sm_30d", "rh_30d"]].values
        X4z = (X4 - np.array(HMM_T_mu)) / np.array(HMM_T_sd)
        probs = hmm_causal_filter(X4z)
        last_probs = probs[-1]
        last_date = tail4["time"].iloc[-1].date()
        hmm_mode = "4-D (legacy)"
        df_trend = df4

    y = (df_trend["wb_30d"].values[-730:]
         if len(df_trend) > 730 else df_trend["wb_30d"].values)
    level, trend = sr_kf_local_trend(y)

    out: Dict = {
        "last_date": last_date,
        "state_probs": last_probs,
        "dominant_state": int(np.argmax(last_probs)),
        "level_wb30": level,
        "trend_wb30_per_day": trend,
        "interp_mode": interp_mode,
        "lat_target": LAT_TARGET,
        "lon_target": LON_TARGET,
        "data_start": df["time"].iloc[0].date() if len(df) > 0 else None,
        "hmm_mode": hmm_mode,
        "has_6h": bool(has_6h),
    }

    # ── ENSO (SST) ─────────────────────────────────────────────────
    epath = find_data_file(enso_csv)
    if epath is not None:
        edf = pd.read_csv(epath)
        base = datetime(1978, 1, 1, 12, 0, 0)
        edf["date"] = edf["time"].apply(
            lambda d: base + timedelta(days=float(d)))
        edf["year"] = edf["date"].dt.year
        edf["month"] = edf["date"].dt.month
        latest_year = int(edf["year"].max())
        aso = edf[(edf["year"] == latest_year)
                  & edf["month"].isin([8, 9, 10])]["enso"]
        if len(aso) > 0:
            aso_mean = float(aso.mean())
            out["enso_year"] = latest_year
            out["enso_aso_mean_sofar"] = aso_mean
            out["enso_phase_sofar"] = classify_enso_phase(aso_mean)
        edf_sorted = edf.sort_values("date")
        out["enso_latest_value"] = float(edf_sorted["enso"].iloc[-1])
        out["enso_latest_date"] = edf_sorted["date"].iloc[-1].date()

    # ── ENSO (SLA) ─────────────────────────────────────────────────
    mpath = find_data_file(msla_csv)
    if mpath is not None:
        mdf = pd.read_csv(mpath)
        mbase = datetime(1950, 1, 1, 0, 0, 0)
        mdf["date"] = mdf["time"].apply(
            lambda d: mbase + timedelta(days=float(d)))
        mdf["year"] = mdf["date"].dt.year
        mdf["month"] = mdf["date"].dt.month
        mdf = mdf.sort_values("date").reset_index(drop=True)

        out["enso_sla_latest_value"] = float(mdf["enso"].iloc[-1])
        out["enso_sla_latest_date"] = mdf["date"].iloc[-1].date()
        out["enso_sla_phase"] = classify_enso_phase(
            out["enso_sla_latest_value"])
        latest_sla_year = int(mdf["year"].max())
        sla_aso = mdf[(mdf["year"] == latest_sla_year)
                      & mdf["month"].isin([8, 9, 10])]["enso"]
        if len(sla_aso) > 0:
            out["enso_sla_aso_mean"] = float(sla_aso.mean())
            out["enso_sla_year"] = latest_sla_year

        # ARX: SLA(t+6 mg) ~ SST(t−4..t) + SLA(t−4..t) + bias
        if epath is not None:
            merged = (edf[["date", "enso"]].rename(columns={"enso": "sst"})
                      .merge(mdf[["date", "enso"]].rename(columns={"enso": "sla"}),
                             on="date", how="inner")
                      .sort_values("date").reset_index(drop=True))
            if len(merged) > 30:
                s_arr = merged["sst"].values.astype(float)
                m_arr = merged["sla"].values.astype(float)
                lag_h, lag_f = 4, 6
                X_list, y_list = [], []
                for i in range(lag_h, len(merged) - lag_f):
                    feats = np.concatenate([
                        s_arr[i - lag_h:i + 1],
                        m_arr[i - lag_h:i + 1],
                        [1.0],
                    ])
                    X_list.append(feats)
                    y_list.append(m_arr[i + lag_f])
                X_mat, y_mat = np.array(X_list), np.array(y_list)
                beta, *_ = np.linalg.lstsq(X_mat, y_mat, rcond=None)
                latest_feats = np.concatenate([
                    s_arr[-(lag_h + 1):], m_arr[-(lag_h + 1):], [1.0]])
                pred_sla = float(np.dot(latest_feats, beta))
                out["enso_sla_arx_6wk"] = pred_sla
                out["enso_sla_arx_6wk_phase"] = classify_enso_phase(pred_sla)

    # ── IOD (DMI) ──────────────────────────────────────────────────
    iw = find_data_file(DEFAULT_IOD_WEEKLY)
    if iw is not None:
        rows = []
        with open(iw) as f:
            for ln in f:
                p = ln.strip().split(",")
                if len(p) < 3:
                    continue
                try:
                    d2 = datetime.strptime(p[1], "%Y%m%d").date()
                    v = float(p[2])
                except ValueError:
                    continue
                rows.append((d2, v))
        if rows:
            rows.sort()
            rec = rows[-8:]
            v = sum(x[1] for x in rec) / len(rec)
            out["iod_value"] = v
            out["iod_phase"] = ("pIOD" if v >= 0.40 else
                                "nIOD" if v <= -0.40 else "NETRAL")
            out["iod_src"] = f"{len(rec)} pekan s.d. {rec[-1][0]}"

    if "iod_phase" not in out:
        im = find_data_file(DEFAULT_IOD_DMI)
        if im is not None:
            with open(im) as f:
                lines = [ln.split() for ln in f if ln.strip()]
            for p in reversed(lines[1:]):
                if len(p) < 13:
                    continue
                try:
                    vals = [float(x) for x in p[1:13]]
                except ValueError:
                    continue
                son = [x for x in (vals[8], vals[9], vals[10])
                       if abs(x - 99.90) > 1e-6]
                if len(son) < 2:
                    continue
                v = sum(son) / len(son)
                out["iod_value"] = v
                out["iod_phase"] = ("pIOD" if v >= 0.40 else
                                    "nIOD" if v <= -0.40 else "NETRAL")
                out["iod_src"] = f"SON {p[0]} (n={len(son)}/3)"
                break

    return out


# ══════════════════════════════════════════════════════════════════════
# §11  TAMPILAN — BLOK KLIMATOLOGI & KALENDER
# ══════════════════════════════════════════════════════════════════════


def _meteo_mangsa_block_dopy(
    dopy_s: float, dopy_e: float, indent: int = 6,
    enso_phase: str = "NETRAL", iod_phase: str = "NETRAL",
) -> None:
    """Cetak blok klimatologi per-mangsa berdasarkan rentang dopy aktual."""
    m, m6h = meteo_for_dopy_range(dopy_s, dopy_e,
                                  enso_phase=enso_phase, iod_phase=iod_phase)
    if m is None:
        return
    hj, hj_d, hhr, et0, wb, sm, rh, tx, tn, angin, rad = m
    pad = " " * indent
    wb_str = f"{wb:+.2f} mm/hari ({'defisit' if wb < 0 else 'surplus'})"
    print(f"{pad}Curah hujan : {hj:>4} mm/musim · {hj_d:.1f} mm/hari · {hhr} hari hujan")
    print(f"{pad}Suhu udara  : {_fmt_suhu(tx, tn, dopy_s, dopy_e)}")
    print(f"{pad}Kelembaban  : RH {rh:.1f}% · SM {sm:.3f} m³/m³ · ET₀ {et0:.2f} mm/hari")
    print(f"{pad}Neraca air  : {wb_str}")
    print(f"{pad}Radiasi/Angin: {rad:.1f} MJ/m² · Angin maks {angin:.1f} km/j")
    if m6h is not None:
        vpd, tcwv, cld, cld_a, sun_h, sm_sh, sm_dp, sT_sh, sT_dp = m6h
        print(f"{pad}[6H] VPD {vpd:.2f} kPa · TCWV {tcwv:.1f} kg/m²")
        print(f"{pad}[6H] Cloud {cld:.0f}% (aft {cld_a:.0f}%) · Sun {sun_h:.1f} h/hari")
        print(f"{pad}[6H] SM 0-7cm {sm_sh:.3f} · SM 28-100cm {sm_dp:.3f} m³/m³")
        print(f"{pad}[6H] sT 0-7cm {sT_sh:.1f}°C · sT 100-255cm {sT_dp:.1f}°C")


def _meteo_musim_block(
    musim: str, indent: int = 4,
    durasi_override: Optional[int] = None,
    dopy_s: Optional[float] = None,
    dopy_e: Optional[float] = None,
    enso_phase: str = "NETRAL",
    iod_phase: str = "NETRAL",
) -> None:
    """Cetak blok klimatologi musim.

    Dua jalur:

    1. **Berbasis dopy** — bila ``dopy_s``/``dopy_e`` diberikan, memakai
       :func:`meteo_for_dopy_range` + koreksi ENSO/IOD. Konsisten dengan
       blok per-mangsa di bawahnya.
    2. **Fallback R30** — bila keduanya None, membaca
       :data:`METEO_MUSIM`/:data:`METEO_MUSIM_6H` langsung (untuk
       ringkasan R30).
    """
    if musim not in METEO_MUSIM:
        return
    pad = " " * indent
    if dopy_s is not None and dopy_e is not None:
        m, m6h = meteo_for_dopy_range(dopy_s, dopy_e,
                                      enso_phase=enso_phase, iod_phase=iod_phase)
        if m is not None:
            hj, hj_d, hhr, et0, wb, sm, rh, tx, tn, angin, rad = m
            dur = int(round(dopy_e - dopy_s + 1))
            wb_str = f"{wb:+.2f} mm/hari ({'defisit' if wb < 0 else 'surplus'})"
            print(f"{pad}Curah hujan : {hj:>5} mm/musim  ·  {hj_d:.1f} mm/hari  ·  ET₀ {et0:.2f} mm/hari")
            print(f"{pad}Neraca air  : {wb_str}  ·  SM {sm:.3f} m³/m³")
            print(f"{pad}Suhu udara  : {_fmt_suhu(tx, tn, dopy_s, dopy_e)}  ·  RH {rh:.1f}%")
            print(f"{pad}Rad./Angin : {rad:.1f} MJ/m² · Angin maks {angin:.1f} km/j · Durasi {dur} hari")
            if m6h is not None:
                vpd, tcwv, cld = m6h[0], m6h[1], m6h[2]
                print(f"{pad}[6H] VPD {vpd:.3f} kPa · TCWV {tcwv:.1f} kg/m²")
                print(f"{pad}[6H] Cloud {cld:.0f}% · Sun {m6h[4]:.1f} h/hari")
            return

    (dur_hard, hj, hj_d, et0, wb, sm, rh, tx, tn, angin, rad) = METEO_MUSIM[musim]
    dur = durasi_override if durasi_override is not None else dur_hard
    r30 = {"Katiga": (19, 93), "Labuh": (94, 207),
           "Rendheng": (208, 285), "Mareng": (286, 383)}
    ds, de = r30.get(musim, (None, None))
    wb_str = f"{wb:+.2f} mm/hari ({'defisit' if wb < 0 else 'surplus'})"
    print(f"{pad}Curah hujan : {hj:>5} mm/musim  ·  {hj_d:.1f} mm/hari  ·  ET₀ {et0:.2f} mm/hari")
    print(f"{pad}Neraca air  : {wb_str}  ·  SM {sm:.3f} m³/m³")
    print(f"{pad}Suhu udara  : {_fmt_suhu(tx, tn, ds, de)}  ·  RH {rh:.1f}%")
    print(f"{pad}Rad./Angin : {rad:.1f} MJ/m² · Angin maks {angin:.1f} km/j · Durasi {dur} hari")
    if musim in METEO_MUSIM_6H:
        vpd, tcwv, cld, sun_h = METEO_MUSIM_6H[musim]
        print(f"{pad}[6H] VPD {vpd:.3f} kPa · TCWV {tcwv:.1f} kg/m²")
        print(f"{pad}[6H] Cloud {cld:.0f}% · Sun {sun_h:.1f} h/hari")


def print_astro_calib_table() -> None:
    """Cetak tabel kalibrasi astronomis lengkap dengan catatan per peristiwa."""
    print()
    print(box_top())
    print(box_row("KALIBRASI ASTRONOMIS — EV06"))
    print(box_row("JRC_Ephemeris · VSOP87D · IERS 2010 · ΔT HMNAO"))
    print(box_row("Lokasi: −7.5220°LS, 112.5661°BT, 28 m  ·  Rata-rata 2020–2029"))
    print(box_mid())
    print(box_row("Peristiwa            Tgl    dopy  σ  Δ vs trad   Mangsa"))
    print(box_bot())
    print()

    rows = [
        ("solstis_juni", "Solstis Juni", "1 Kasa"),
        ("solstis_des", "Solstis Desember", "6 Kanem†"),
        ("equinox_maret", "Ekuinoks Maret", "8/9 Kawolu†"),
        ("equinox_sept", "Ekuinoks September", "4 Kapat†"),
        ("zenith_I_okt", "Zenith Matahari I", "5 Kalima"),
        ("zenith_II_mar", "Zenith Matahari II", "8 Kawolu†"),
        ("orion_helrise", "Orion Heliacal Rise", "1 Kasa"),
        ("orion_evening_rise", "Orion Acronychal Rise", "6 Kanem"),
        ("orion_evening_culm", "Orion Kulminasi Senja", "8 Kawolu†"),
        ("orion_midnight_culm", "Orion Kulminasi Tengah Mlm", "6 Kanem"),
        ("orion_acron_set", "Orion Acronychal Set", "12 Sada"),
    ]
    for key, label, mangsa in rows:
        ev = ASTRO_CALIB.get(key)
        if not ev:
            continue
        tgl = f"{ev['mean_day']:02d} {MONTHS_ID_SHORT.get(ev['mean_month'], '---')}"
        print(f"  {label:<22} {tgl:>6}  {ev['mean_dopy']:>7.1f}  "
              f"{ev['std_dopy']:.2f}  {ev['delta']:>+6.1f}  {mangsa}"[:W])

    print()
    for ln in textwrap.wrap(
        "† = berdasarkan skenario R30/R10 (EV04). "
        "Lihat catatan per peristiwa.",
        width=W, initial_indent="  ", subsequent_indent="    ",
    ):
        print(ln)
    print()
    print(thin_hbar(2))
    print("  Catatan penting per peristiwa:")
    print()
    for key, label, _ in rows:
        ev = ASTRO_CALIB.get(key)
        if not ev or "catatan" not in ev:
            continue
        wprint(label[:20], ev["catatan"], lw=22, indent=2)
        print()


def print_calendar(
    cal: List[Dict], judul: str,
    scenario_key: str = DEFAULT_SCENARIO,
    show_meteo: bool = True, show_astro: bool = True,
) -> None:
    """Cetak kalender lengkap dengan meteo dan penanda astro per mangsa."""
    scn_label = CALIB_SCENARIOS[scenario_key]["label"]
    ms = CALIB_SCENARIOS[scenario_key]["musim_start"]
    enso_phase = {"ELNINO": "ELNINO", "LANINA": "LANINA"}.get(
        scenario_key, "NETRAL")
    iod_phase = ENSO_TO_IOD.get(scenario_key, "NETRAL")

    print()
    print(box_top())
    print(box_row(judul))
    print(box_row("ERA5/ERA5-Land (ECMWF/C3S) · IFS HRES 9km (ECMWF)"))
    print(box_row("−7.5220°LS, 112.5661°BT, 28 m · IDW 2 stasiun (P1+P2)"))
    if show_meteo:
        print(box_row(f"Skenario musim: {scenario_key} — {scn_label}"))
        klim_note = (f"Klimatologi: interpolasi dopy R30 + koreksi Δ{scenario_key} empiris"
                     if enso_phase != "NETRAL"
                     else f"Klimatologi: interpolasi dopy R30 (1996–2025) · skenario {scenario_key}")
        print(box_row(klim_note))
        if iod_phase in ("pIOD", "nIOD"):
            bobot_txt = "0.50 sinergi ENSO–IOD" if enso_phase != "NETRAL" else "0.30 standalone"
            print(box_row(f"Koreksi IOD: {iod_phase}  (DMI 1950–2025, mangsa 3–5, "
                          f"bobot {bobot_txt})"))
        print(box_row("Tx̄/Tn̄ = rata² T maks/min harian · (x) = ekstrem absolut"))
    if show_astro:
        print(box_row("Astro: VSOP87D+IERS2010 · JRC_Ephemeris · 2020–2029"))
    print(box_bot())

    for musim in MUSIM_ORDER:
        ms_n = {"Katiga": ms["Labuh"], "Labuh": ms["Rendheng"],
                "Rendheng": ms["Mareng"], "Mareng": 365 + ms["Katiga"]}
        dopy_s, dopy_e = ms[musim], ms_n[musim] - 1
        sec_header(f"MUSIM {musim}", MUSIM_DESKRIPSI[musim],
                   dopy_range=f"{dopy_s}–{dopy_e}")
        if show_meteo:
            _meteo_musim_block(musim, indent=2,
                               durasi_override=int(dopy_e - dopy_s + 1),
                               dopy_s=dopy_s, dopy_e=dopy_e,
                               enso_phase=enso_phase, iod_phase=iod_phase)
            print()

        hdr = f"{'No':>3}  {'Nama':<10}  {'Mulai':<13} {'Selesai':<13} {'Dur (hr)':>8}"
        print(f"  {hdr}")
        print(f"  {'─' * len(hdr)}")

        for m in cal:
            if m["musim"] != musim:
                continue
            print(f"  {m['no']:>3}  {m['nama']:<10}  "
                  f"{fmt(m['mulai']):<13} {fmt(m['akhir']):<13} "
                  f"{m['durasi']:>8}")
            if show_meteo:
                _meteo_mangsa_block_dopy(
                    m.get("dopy_start", R30_DOPY_RANGES[m["no"]][0]),
                    m.get("dopy_end", R30_DOPY_RANGES[m["no"]][1]),
                    indent=7, enso_phase=enso_phase, iod_phase=iod_phase)
            pre = "       Ciri       : "
            print(textwrap.fill(m["ciri"], width=W,
                                initial_indent=pre,
                                subsequent_indent=" " * len(pre)))
            if m.get("candra"):
                pre_c = "       Candra     : "
                print(textwrap.fill(m["candra"], width=W,
                                    initial_indent=pre_c,
                                    subsequent_indent=" " * len(pre_c)))
            print()
    print()


def print_mangsa_today(tanggal: date,
                       scenario_key: str = DEFAULT_SCENARIO) -> None:
    """Cetak mangsa untuk tanggal tertentu (tradisional + terkalibrasi)."""
    pyear, dopy = get_pranatamangsa_year_and_dopy(tanggal)
    scn_label = CALIB_SCENARIOS[scenario_key]["label"]
    enso_phase = {"ELNINO": "ELNINO", "LANINA": "LANINA"}.get(
        scenario_key, "NETRAL")
    iod_phase = ENSO_TO_IOD.get(scenario_key, "NETRAL")

    print()
    print(box_top())
    print(box_row(f"MANGSA UNTUK TANGGAL: {fmt(tanggal)}"))
    print(box_row(f"Tahun-Pranata: {pyear}/{pyear+1}  ·  "
                  f"Hari ke-{dopy+1} (dopy={dopy})"))
    print(box_mid())

    trad = get_mangsa_by_date(tanggal, "tradisional")
    kal = get_mangsa_by_date(tanggal, "terkalibrasi", scenario_key)

    if trad:
        print(box_row(""))
        print(box_row("[ TRADISIONAL — Reformasi Paku Buwana VII, 1855 ]"))
        print(box_row(f"  Mangsa ke-{trad['no']}: {trad['nama'].upper()}  ·  Musim {trad['musim']}"))
        print(box_row(f"  Periode: {fmt(trad['mulai'])} — "
                      f"{fmt(trad['akhir'])} ({trad['durasi']} hari)"))
        first = True
        for raw in trad["ciri"].split("\n"):
            for ln in _wrap_ciri_line(raw, W - 14):
                print(box_row(f"  Ciri: {ln}" if first else f"        {ln}"))
                first = False
        candra = CIRI_JAWA.get(trad["no"], "")
        if candra:
            print(box_row(""))
            print(box_row("  Candraning Măngsa (tradisional):"))
            for ln in textwrap.wrap(candra, width=W - 12,
                                    initial_indent="      ",
                                    subsequent_indent="      "):
                print(box_row(ln))

    print(box_mid())
    if kal:
        print(box_row(""))
        print(box_row(f"[ TERKALIBRASI — {scn_label} ]"))
        print(box_row(f"  Mangsa ke-{kal['no']}: {kal['nama'].upper()}  ·  Musim {kal['musim']}"))
        print(box_row(f"  Periode: {fmt(kal['mulai'])} — "
                      f"{fmt(kal['akhir'])} ({kal['durasi']} hari)"))
        if trad and kal["no"] != trad["no"]:
            print(box_row(f"  >> BERBEDA dari tradisional (tradisional: mangsa "
                          f"{trad['no']} {trad['nama']})"))
        elif trad:
            sel = (kal["mulai"] - trad["mulai"]).days
            sgn = "lebih awal" if sel < 0 else "lebih lambat"
            print(box_row(f"  Awal mangsa bergeser {sel:+d} hari "
                          f"({abs(sel)} hari {sgn}) vs. tradisional"))

        for ln in textwrap.wrap(kal["ciri"], width=W - 12,
                                initial_indent="  Ciri: ",
                                subsequent_indent="        "):
            print(box_row(ln))

        print(box_mid())
        print(box_row(""))
        klim_src = f"R30 1996–2025 · skenario {scenario_key}"
        if iod_phase in ("pIOD", "nIOD"):
            klim_src += f" · IOD {iod_phase}"
        print(box_row(f"  Klimatologi (sumber {klim_src}):"))
        _ds = kal.get("dopy_start", R30_DOPY_RANGES.get(kal["no"], (0., 0.))[0])
        _de = kal.get("dopy_end", R30_DOPY_RANGES.get(kal["no"], (0., 0.))[1])
        _m, _m6h = meteo_for_dopy_range(_ds, _de, enso_phase=enso_phase,
                                        iod_phase=iod_phase)
        if _m is not None:
            hj, hj_d, hhr, et0, wb, sm, rh, tx, tn, angin, rad = _m
            wb_str = "defisit" if wb < 0 else "surplus"
            print(box_row(f"  Curah hujan : {hj} mm/musim · "
                          f"{hj_d:.1f} mm/hari · {hhr} hari hujan"))
            print(box_row(f"  Suhu udara  : {_fmt_suhu(tx, tn, _ds, _de)}"))
            print(box_row(f"  Kelembaban  : RH {rh:.1f}% · SM {sm:.3f} m³/m³ · "
                          f"ET₀ {et0:.2f} mm/hari"))
            print(box_row(f"  Neraca air  : P−ET₀ {wb:+.2f} mm/hari ({wb_str}) "
                          f"· Rad {rad:.1f} MJ/m²"))
        if _m6h is not None:
            vpd, tcwv, cld, cld_a, sun_h, sm_sh, sm_dp, sT_sh, sT_dp = _m6h
            print(box_row(f"  [6H] VPD {vpd:.2f} kPa · TCWV {tcwv:.1f} kg/m²"))
            print(box_row(f"  [6H] Cloud {cld:.0f}% (aft {cld_a:.0f}%) · "
                          f"Sun {sun_h:.1f} h/hari"))
            print(box_row(f"  [6H] SM 0-7cm {sm_sh:.3f} · "
                          f"SM 28-100cm {sm_dp:.3f} m³/m³"))
            print(box_row(f"  [6H] sT 0-7cm {sT_sh:.1f}°C · "
                          f"sT 100-255cm {sT_dp:.1f}°C"))

        print(box_mid())
        print(box_row(""))
        print(box_row("  Penanda Astronomis (VSOP87D, rata-rata 2020–2029):"))
        ev_keys = astro_events_in_range(_ds, _de)
        if not ev_keys:
            print(box_row("  (tidak ada penanda astronomis khusus untuk mangsa ini)"))
        for ev_key in ev_keys:
            ev = ASTRO_CALIB.get(ev_key, {})
            if not ev:
                continue
            label = ASTRO_LABEL.get(ev_key, ev_key)
            tgl_str = f"{ev['mean_day']:02d} {MONTHS_ID_SHORT.get(ev['mean_month'], '---')}"
            delta_s = astro_delta_str(ev_key)
            print(box_row(f"  {label}"))
            combined = f"→ Tgl rata-rata: {tgl_str}  |  {delta_s}"
            for ln in textwrap.wrap(combined, width=W - 12,
                                    initial_indent="    ",
                                    subsequent_indent="      "):
                print(box_row(ln))
    print(box_bot())
    print()


def print_perbandingan(pyear: int) -> None:
    """Cetak perbandingan selisih hari antar-skenario."""
    trad_cal = build_calendar_tradisional(pyear)
    scn_keys = list(CALIB_SCENARIOS.keys())

    print()
    print(box_top())
    print(box_row(f"PERBANDINGAN SKENARIO — Tahun-Pranata {pyear}/{pyear+1}"))
    print(box_row("Angka = selisih hari awal mangsa vs. Tradisional (– lebih awal)"))
    print(box_bot())
    print()

    hdr = f"  {'No':>2}  {'Nama':<10} {'Tradisional':>12}"
    for k in scn_keys:
        hdr += f"  {k:>6}"
    print(hdr[:W])
    print(thin_hbar(2))

    cal_by_scn = {k: build_calendar_terkalibrasi(pyear, k) for k in scn_keys}
    for td in trad_cal:
        row = f"  {td['no']:>2}  {td['nama']:<10} {fmt(td['mulai']):>12}"
        for k in scn_keys:
            m_cal = next(x for x in cal_by_scn[k] if x["no"] == td["no"])
            delta = (m_cal["mulai"] - td["mulai"]).days
            row += f"  {delta:>+6}"
        print(row[:W])

    print()
    print(thin_hbar(2))
    print("  Legenda skenario:")
    for k, v in CALIB_SCENARIOS.items():
        print()
        print(f"  {k:<7}: {v['label']}")
        wprint("Catatan", v["catatan"], lw=7, indent=10)
    print()


def print_durasi_musim() -> None:
    """Cetak durasi tiap musim per skenario + klimatologi R30."""
    print()
    print(box_top())
    print(box_row("DURASI TIAP MUSIM (hari)  —  Tradisional vs Kalibrasi"))
    print(box_bot())
    print()

    musims = MUSIM_ORDER
    col_w = 11
    hdr = f"  {'Skenario':<13}" + "".join(f"{mu:>{col_w}}" for mu in musims)
    hdr += f"  {'Total':>6}"
    print(hdr[:W])
    print(thin_hbar(2))

    o, on = ORIG_MUSIM_START, ORIG_MUSIM_START_NEXT
    durs = [on[mu] - o[mu] for mu in musims]
    print(f"  {'Tradisional':<13}" + "".join(f"{d:>{col_w}}" for d in durs)
          + f"  {sum(durs):>6}")

    for key, v in CALIB_SCENARIOS.items():
        s = v["musim_start"]
        sn = {"Katiga": s["Labuh"], "Labuh": s["Rendheng"],
              "Rendheng": s["Mareng"], "Mareng": 365 + s["Katiga"]}
        durs = [sn[mu] - s[mu] for mu in musims]
        print(f"  {key:<13}" + "".join(f"{d:>{col_w}}" for d in durs)
              + f"  {sum(durs):>6}")

    print()
    print(thin_hbar(2))
    print("\n  Klimatologi tiap musim — Normal Iklim R30 (1996–2025):")
    print()
    for mu in musims:
        print(f"\n  ▸ {mu.upper()} — {MUSIM_DESKRIPSI[mu]}")
        _meteo_musim_block(mu, indent=4)
    print()


def print_klimatologi_bulanan() -> None:
    """Cetak klimatologi bulanan dan ringkasan per-musim (R30)."""
    print()
    print(box_top())
    print(box_row("KLIMATOLOGI BULANAN — Normal Iklim R30 (1996–2025)"))
    print(box_row("ERA5/Land-IFSHRES · −7.522°LS 112.566°BT · 28 m  [IDW 2 stasiun]"))
    print(box_mid())
    print(box_row("Satuan: mm/bln · mm/hr · MJ/m² · km/j · °C"))
    print(box_bot())
    print()

    params = [
        ("Hujan total (mm/bln)", 0, "{:>6.0f}"),
        ("Hujan (mm/hr)", 1, "{:>6.1f}"),
        ("ET₀ (mm/hr)", 2, "{:>6.2f}"),
        ("P−ET₀ (mm/hr)", 3, "{:>+6.1f}"),
        ("SM (m³/m³)", 4, "{:>6.3f}"),
        ("RH (%)", 5, "{:>6.1f}"),
        ("Tx (°C)", 6, "{:>6.1f}"),
        ("Tn (°C)", 7, "{:>6.1f}"),
        ("Angin (km/j)", 8, "{:>6.1f}"),
        ("Radiasi (MJ/m²)", 9, "{:>6.1f}"),
    ]
    for half_start, bulan_list in [(1, range(1, 7)), (7, range(7, 13))]:
        names = [BULAN_ID[b] for b in bulan_list]
        print(f"  {'Parameter':<20}" + "".join(f"{n:>7}" for n in names))
        print(thin_hbar(2))
        for label, idx, fmt_str in params:
            vals = [METEO_BULANAN[b][idx] for b in bulan_list]
            print(f"  {label:<20}" + "".join(fmt_str.format(v) for v in vals))
        print()

    print(thin_hbar(0))
    print()
    print(f"  {'RINGKASAN PER MUSIM (R30)':^{W-2}}")
    print()
    mus_params = [
        ("Hujan total (mm)", 1, "{:>10.0f}"),
        ("Hujan (mm/hr)", 2, "{:>10.1f}"),
        ("ET₀ (mm/hr)", 3, "{:>10.2f}"),
        ("P−ET₀ (mm/hr)", 4, "{:>+10.2f}"),
        ("SM (m³/m³)", 5, "{:>10.3f}"),
        ("RH (%)", 6, "{:>10.1f}"),
        ("Tx (°C)", 7, "{:>10.1f}"),
        ("Tn (°C)", 8, "{:>10.1f}"),
        ("Angin (km/j)", 9, "{:>10.1f}"),
        ("Radiasi (MJ/m²)", 10, "{:>10.1f}"),
        ("Durasi (hari)", 0, "{:>10.0f}"),
    ]
    col_w = 10
    print(f"  {'Parameter':<20}" + "".join(f"{mu:>{col_w}}" for mu in MUSIM_ORDER))
    print(thin_hbar(2))
    for label, idx, fmt_str in mus_params:
        print(f"  {label:<20}"
              + "".join(fmt_str.format(METEO_MUSIM[mu][idx])
                        for mu in MUSIM_ORDER))
    print()

    # ── Ringkasan 6H per musim ─────────────────────────────────────
    print()
    print(f"  {'RINGKASAN 6H PER MUSIM (EV06)':^{W-2}}")
    print(f"  {'Nilai = rata-rata musiman (R30 1996–2025)':^{W-2}}")
    print()
    print(f"  {'Parameter':<22}" + "".join(f"{mu:>{11}}" for mu in MUSIM_ORDER))
    print(thin_hbar(2))
    h6_labels = [
        ("VPD (kPa)", 0, "{:>11.3f}"),
        ("TCWV (kg/m²)", 1, "{:>11.1f}"),
        ("Cloud (%)", 2, "{:>11.0f}"),
        ("Sunshine (h/d)", 3, "{:>11.1f}"),
    ]
    for lbl, idx, fstr in h6_labels:
        print(f"  {lbl:<22}"
              + "".join(fstr.format(METEO_MUSIM_6H[mu][idx])
                        for mu in MUSIM_ORDER))
    print()
    print(thin_hbar(0))
    print()


def print_live_nowcast() -> None:
    """Cetak hasil :func:`live_nowcast` dalam format panel terstruktur."""
    print()
    print(box_top())
    print(box_row("NOWCAST LANGSUNG — Analisis Iklim Real-Time"))
    print(box_row("HMM 8-D (EV05) + SR-EKF Level/Tren (ARCH(1))"))
    print(box_bot())

    res = live_nowcast()
    if res is None:
        return

    print()
    lat_s = f"{abs(res.get('lat_target', LAT_TARGET)):.4f}°LS"
    lon_s = f"{res.get('lon_target', LON_TARGET):.4f}°BT"
    print(f"  Titik target               : {lat_s}, {lon_s}")
    print(f"  Mode data                  : {res.get('interp_mode', '-')}")
    print(f"  Mode HMM                   : {res.get('hmm_mode', '-')}")
    print(f"  Data 6-jam tersedia        : "
          f"{'Ya' if res.get('has_6h') else 'Tidak (fallback 4-D)'}")
    if res.get("data_start"):
        print(f"  Rentang data               : {res['data_start']} s.d. "
              f"{fmt(res['last_date'])}")
    else:
        print(f"  Data meteorologi terakhir  : {fmt(res['last_date'])}")

    print()
    print(thin_hbar(2))
    print("  Probabilitas rejim iklim (HMM forward/causal, tanpa look-ahead):")
    print(thin_hbar(2))
    for k in range(4):
        p = res["state_probs"][k]
        bar = "█" * int(round(p * 30))
        print(f"  State {k}  {p*100:5.1f}%  {bar:<32}")
        print(f"           {HMM_T_STATE[k]}")
    dom = res["dominant_state"]
    print()
    print(f"  >> Rejim dominan saat ini: State {dom} — {HMM_T_STATE[dom]}")

    print()
    print(thin_hbar(2))
    print("  SR-EKF — Neraca air P−ET₀ 30-hari (level & tren ter-filter):")
    print(thin_hbar(2))
    wb, trnd = res["level_wb30"], res["trend_wb30_per_day"]
    arah = "→ menuju lebih basah" if trnd > 0 else "→ menuju lebih kering"
    print(f"  Level saat ini  : {wb:+.1f} mm / 30 hari")
    print(f"  Tren harian     : {trnd:+.3f} mm/hari  {arah}")

    if "enso_phase_sofar" in res or "enso_sla_latest_value" in res:
        print()
        print(thin_hbar(2))
        print(f"  Status ENSO — Niño3.4 (data s.d. {fmt(res['enso_latest_date'])}):")
        print(thin_hbar(2))
        if "enso_phase_sofar" in res:
            print(f"  [SST]  ASO {res['enso_year']}   : "
                  f"{res['enso_aso_mean_sofar']:+.2f}  →  {res['enso_phase_sofar']}")
            print(f"  [SST]  Terkini        : {res['enso_latest_value']:+.2f}")
        if "enso_sla_latest_value" in res:
            print()
            print(f"  [SLA]  Terkini        : "
                  f"{res['enso_sla_latest_value']:+.2f}  →  {res['enso_sla_phase']}"
                  f"  (s.d. {fmt(res['enso_sla_latest_date'])})")
            if "enso_sla_aso_mean" in res:
                print(f"  [SLA]  ASO {res['enso_sla_year']}   : "
                      f"{res['enso_sla_aso_mean']:+.2f}")
            if "enso_sla_arx_6wk" in res:
                print(f"  [ARX]  Prakiraan +6 minggu : "
                      f"{res['enso_sla_arx_6wk']:+.2f}  →  "
                      f"{res['enso_sla_arx_6wk_phase']}")
        if "enso_latest_value" in res and "enso_sla_latest_value" in res:
            sst_v, sla_v = res["enso_latest_value"], res["enso_sla_latest_value"]
            if classify_enso_phase(sst_v) != classify_enso_phase(sla_v):
                print()
                print(f"  ⚠ Divergensi SST ({classify_enso_phase(sst_v)}) ↔ "
                      f"SLA ({classify_enso_phase(sla_v)}) — pantau 4–6 minggu.")
        phase_for_scn = res.get("enso_phase_sofar",
                                res.get("enso_sla_phase", "Netral"))
        scn = SCENARIO_FOR_PHASE.get(phase_for_scn)
        if scn:
            print()
            print(f"  >> Rekomendasi skenario: '{scn}'")
            wprint("Catatan", CALIB_SCENARIOS[scn]["catatan"], lw=7, indent=5)

    if "iod_phase" in res:
        print()
        print(thin_hbar(2))
        print("  Status IOD (Dipole Mode Index):")
        print(thin_hbar(2))
        print(f"  Fase   : {res['iod_phase']}  (DMI {res['iod_value']:+.2f})")
        print(f"  Sumber : {res['iod_src']}")
        enso_ph = res.get("enso_phase_sofar",
                          res.get("enso_sla_phase", "Netral"))
        if ((enso_ph == "El Niño" and res["iod_phase"] == "pIOD")
                or (enso_ph == "La Niña" and res["iod_phase"] == "nIOD")):
            print("  ⚑ Sinergi ENSO–IOD → bobot koreksi IOD 0.50")
        elif res["iod_phase"] in ("pIOD", "nIOD"):
            print("  · IOD standalone   → bobot koreksi IOD 0.30")

    print()
    print(thin_hbar(2))
    print("  Sumber data:")
    print(thin_hbar(2))
    print("  SLA  : AVISO/DUACS (CNES/CLS) — DOI 10.24400/527896/A01-2025.008")
    print("  SST  : NOAA OISST v2.1 — DOI 10.1175/JCLI-D-20-0166.1")
    print("  IOD  : JMA (DMI) — Saji et al. (1999), Nature 401:360")
    print("  Met  : ERA5/ERA5-Land (ECMWF/C3S) + IFS HRES 9km (ECMWF)")
    print("  Astro: VSOP87D (IMCCE) + IERS 2010 + HMNAO ΔT")
    print()


# ══════════════════════════════════════════════════════════════════════
# §12  ANTARMUKA MENU & ENTRY POINT
# ══════════════════════════════════════════════════════════════════════


def input_int(prompt: str, default: Optional[int] = None) -> Optional[int]:
    """Baca integer dari stdin; kembalikan default bila input kosong/invalid."""
    try:
        s = input(prompt).strip()
        if not s and default is not None:
            return default
        return int(s)
    except (ValueError, EOFError):
        return default


def input_date_str(prompt: str) -> Optional[date]:
    """Baca tanggal YYYY-MM-DD dari stdin; None bila kosong/invalid."""
    try:
        s = input(prompt).strip()
        if not s:
            return None
        y, mo, d = map(int, s.split("-"))
        return date(y, mo, d)
    except (ValueError, EOFError):
        return None


def choose_scenario() -> str:
    """Menu pilih skenario; mengembalikan DEFAULT_SCENARIO bila tidak valid."""
    keys = list(CALIB_SCENARIOS.keys())
    print(f"\n  Skenario tersedia: {', '.join(keys)}")
    s = input(f"  Pilih skenario [{DEFAULT_SCENARIO}]: ").strip().upper()
    return s if s in CALIB_SCENARIOS else DEFAULT_SCENARIO


def show_menu() -> None:
    """Tampilkan menu utama."""
    print()
    print(box_top("PRANATA MANGSA — EV06 METEO(1H+6H)+ENSO+IOD+ASTRO"))
    print(box_row("−7.52S112.56E28m · ERA5/Land IFS HRES 1940–2026 · ENSO 1993–2026"))
    print(box_row("IOD_DELTA: DMI bulanan 1950–2025 · mangsa 3–5 · bobot 0.30/0.50"))
    print(box_row("HMM 8-D (EV05) · VSOP87D + IERS2010 · JRC_Ephemeris 2020–2029"))
    print(box_mid())
    items = [
        "  1 › Kalender Tradisional (Paku Buwana VII, 1855)",
        "  2 › Kalender Terkalibrasi (meteorologi + astro)",
        "  3 › Cek Mangsa Hari Ini / Tanggal Tertentu",
        "  4 › Tabel Perbandingan Skenario (selisih hari)",
        "  5 › Durasi Tiap Musim per Skenario",
        "  6 › Klimatologi Bulanan & Ringkasan Per-Musim",
        "  7 › Nowcast Langsung (HMM 8-D, real-time)",
        "  8 › Tabel Kalibrasi Astronomis",
        "  9 › Atribusi Sumber Data (sitasi ilmiah)",
        "  0 › Keluar",
    ]
    for item in items:
        print(box_row(item))
    print(box_bot())


def laporan_singkat() -> None:
    """Laporan singkat: mangsa hari ini + nowcast bila file tersedia."""
    today = date.today()
    print_mangsa_today(today, DEFAULT_SCENARIO)
    if (find_data_file(DEFAULT_METEO_CSV)
            or find_data_file(DEFAULT_METEO_CSV2)):
        print_live_nowcast()


def main_loop() -> None:
    """Loop menu interaktif utama."""
    while True:
        show_menu()
        pilihan = input("  Pilih menu (0–9): ").strip()

        if pilihan == "0":
            print()
            print(box_top())
            print(box_row("Terima kasih. Sampai jumpa!  — Pranata Mangsa EV06"))
            print(box_bot())
            print()
            break

        elif pilihan == "1":
            default_py = get_pranatamangsa_year_and_dopy(date.today())[0]
            tahun = input_int(
                f"  Tahun mulai siklus (YYYY) [{default_py}]: ", default_py)
            if tahun is not None:
                print_calendar(build_calendar_tradisional(tahun),
                               f"KALENDER TRADISIONAL — SIKLUS {tahun}/{tahun+1}",
                               show_meteo=False, show_astro=False)

        elif pilihan == "2":
            default_py = get_pranatamangsa_year_and_dopy(date.today())[0]
            tahun = input_int(
                f"  Tahun-pranata mulai (YYYY) [{default_py}]: ", default_py)
            scn = choose_scenario()
            if tahun is not None:
                print_calendar(
                    build_calendar_terkalibrasi(tahun, scn),
                    f"KALENDER TERKALIBRASI {tahun}/{tahun+1}",
                    scenario_key=scn, show_meteo=True, show_astro=True)

        elif pilihan == "3":
            s = input("  Tanggal (YYYY-MM-DD) [kosong = hari ini]: ").strip()
            tgl = date.today() if not s else date(*map(int, s.split("-")))
            scn = choose_scenario()
            print_mangsa_today(tgl, scn)

        elif pilihan == "4":
            default_py = get_pranatamangsa_year_and_dopy(date.today())[0]
            tahun = input_int(
                f"  Tahun-pranata mulai (YYYY) [{default_py}]: ", default_py)
            if tahun is not None:
                print_perbandingan(tahun)

        elif pilihan == "5":
            print_durasi_musim()

        elif pilihan == "6":
            print_klimatologi_bulanan()

        elif pilihan == "7":
            print_live_nowcast()

        elif pilihan == "8":
            print_astro_calib_table()

        elif pilihan == "9":
            print_data_attribution(detail="lengkap")

        else:
            print("\n  Pilihan tidak valid. Masukkan angka 0–9.")
            input("\n  Tekan Enter untuk melanjutkan...")
            continue

        input("\n  Tekan Enter untuk kembali ke menu...")


def _build_argparser() -> argparse.ArgumentParser:
    """Bangun parser argumen CLI."""
    ap = argparse.ArgumentParser(
        prog="pranatamangsa",
        description=("Pranata Mangsa EV06 — kalender pertanian tropis "
                     "berbasis reanalisis iklim, ENSO/IOD, dan astronomi."),
    )
    ap.add_argument("--report", action="store_true",
                    help="Laporan singkat: mangsa hari ini + nowcast.")
    ap.add_argument("--astro", action="store_true",
                    help="Cetak tabel kalibrasi astronomis saja.")
    ap.add_argument("--attribution", action="store_true",
                    help="Cetak atribusi sumber data lengkap.")
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point modul."""
    args = _build_argparser().parse_args(argv)
    try:
        if args.report:
            laporan_singkat()
        elif args.astro:
            print_astro_calib_table()
        elif args.attribution:
            print_data_attribution(detail="lengkap")
        else:
            main_loop()
    except KeyboardInterrupt:
        print("\n\n  Program dihentikan. Sampai jumpa!")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())