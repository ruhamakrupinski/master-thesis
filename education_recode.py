"""
education_recode.py
------------------------------------------------------------------
Re-map 5-digit PSCED/CBMS education codes ➜ custom 3-digit scheme.
Exports:
    recode_edu(code)  → int|None
    MAP               → dict[str,int]  (exact one-to-one rules only)
"""

from typing import Dict, Optional

# ================================================================
# 1 ▀▀  EXACT ONE-TO-ONE RULES
#    (irregular values or codes whose prefix shouldn’t be shared)
# ================================================================
_EXACT: Dict[str, int] = {
    # Early childhood & SPED
    "00000": 0, "01000": 10, "02000": 10,
    "191": 191, "192": 192, "10002": 191, "24002": 192,
    # Elementary & JHS
    "10010": 410, "10020": 460,
    "24010": 470, "24020": 500,
    # Senior HS (K-12)
    "34011": 510, "34012": 510, "34021": 510, "34031": 510,
    "34022": 520, "34032": 520,
    "35001": 510, "35002": 520,
    # College years
    **{f"{x:03d}": x for x in range(710, 770, 10)},  # 710-760
    # Post-graduate
    "910": 910, "920": 920, "930": 930, "940": 940,
    # Other / unknown
    "69999": 999, "99999": 999,
}

# ================================================================
# 2 ▀▀  PREFIX RULES  –  POST-SECONDARY (4-series → 6xx codes)
# ================================================================
_POST_SECONDARY: Dict[str, int] = {
    "400": 601,              # basic programmes
    "4002": 608,             # literacy / numeracy
    "4003": 609,             # personal development
    "401": 614,              # teacher-training
    "402": 621,              # arts & humanities
    "403": 631,              # social / behavioural / journalism
    "404": 634,              # business / admin / law
    "40511": 642, "40512": 642, "40519": 642,      # life sciences
    "4052": 685,             # environment
    "405": 644,              # physical & maths
    "406": 648,              # ICT
    "4071": 652,             # engineering trades
    "4072": 654,             # manufacturing / processing
    "4073": 658, "407": 652, # architecture / default engineering
    "4084": 664,             # veterinary
    "408": 662,              # agriculture / forestry
    "4091": 672, "4092": 676,# health / welfare
    "4101": 681, "4104": 684,# personal / transport
    "4103": 686, "410": 681, # security / other services
}

# ================================================================
# 3 ▀▀  PREFIX RULES  –  TERTIARY (5- & 6-series → 8xx codes)
# ================================================================
_ACADEMIC: Dict[str, int] = {
    # Generic / education
    "500": 801, "600": 801,
    "501": 814, "601": 814,
    # Arts & humanities
    "5021": 821, "6021": 821,
    "5022": 822, "6022": 822, "602": 821,      # catch-all 602**
    # Social sciences & journalism
    "5031": 831, "603": 831,
    "5032": 832, "6032": 832,
    # Business & law
    "5041": 834, "504": 834,   "604": 834,
    "50421": 838, "60421": 838,
    # Sciences & ICT
    "50511": 842, "50512": 842, "60511": 842,
    "5052": 885, "6052": 885,                  # environment
    "5053": 844, "6053": 844,                  # physical sci.
    "5054": 846, "6054": 846,                  # math / stats
    "505": 844, "605": 844,                    # generic 505*/605*
    "506": 848, "606": 848,                    # ICT (60610/11/12/13/19/88…)
    # Engineering cluster
    "5071": 852, "507": 852, "607": 852,
    "5072": 854, "6072": 854,
    "5073": 858, "6073": 858, "60700": 852, "60799": 852,
    # Agriculture & veterinary
    "5084": 864, "6084": 864,
    "508": 862, "608": 862,
    # Health & welfare
    "5091": 872, "609": 872,
    "5092": 876, "6092": 876, "60988": 872,
    # Services, transport, environment, security
    "5101": 881, "610": 881,   # services (61088, 61099…)
    "5104": 884, "6104": 884,
    "5102": 885, "6102": 885,
    "5103": 886, "6103": 886,
}

# ================================================================
# 4 ▀▀  POST-GRADUATE QUICK RULE (7- & 8-series)
# ================================================================
def _postgrad(code: str) -> Optional[int]:
    if code[0] == "7":
        return 910 if code[2] == "0" else 920
    if code[0] == "8":
        return 930 if code[2] == "0" else 940
    return None

# ================================================================
# 5 ▀▀  MAIN RECODE FUNCTION
# ================================================================
def recode_edu(raw) -> Optional[int]:
    """Convert any PSCED/CBMS 5-digit code to the custom 3-digit code."""
    code = str(raw or "").strip()
    if not code:
        return None
    code = code.zfill(5)              # preserve leading zeros

    # 1) exact mapping
    if code in _EXACT:
        return _EXACT[code]

    # 2) post-secondary (4xx)
    if code.startswith("4"):
        for pref, new in _POST_SECONDARY.items():
            if code.startswith(pref):
                return new

    # 3) tertiary (5xx / 6xx)
    if code[0] in "56":
        for pref, new in _ACADEMIC.items():
            if code.startswith(pref):
                return new

    # 4) postgraduate (7xx / 8xx)
    pg = _postgrad(code)
    if pg is not None:
        return pg

    # 5) fall-through
    return None

# A minimal dict for Series.map() when you only want the exact rules
MAP: Dict[str, int] = {k: v for k, v in _EXACT.items()}
