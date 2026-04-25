"""Feature definitions for the ExoData Challenge analysis."""

IDENTIFIER_COLUMNS = [
    "rowid",
    "pl_name",
    "hostname",
    "pl_letter",
    "hd_name",
    "hip_name",
    "tic_id",
    "gaia_dr2_id",
    "gaia_dr3_id",
]

CONTEST_KEY_COLUMNS = [
    "pl_name",
    "hostname",
    "discoverymethod",
    "disc_year",
    "pl_rade",
    "pl_radj",
    "pl_bmasse",
    "pl_bmassj",
    "pl_dens",
    "pl_orbper",
    "pl_orbsmax",
    "pl_orbeccen",
    "st_teff",
    "st_met",
    "sy_pnum",
    "pl_insol",
    "pl_eqt",
]

CONTEST_NUMERIC_COLUMNS = [
    "pl_rade",
    "pl_bmasse",
    "pl_dens",
    "pl_orbper",
    "pl_orbsmax",
    "pl_orbeccen",
    "st_teff",
    "st_met",
    "sy_pnum",
    "pl_insol",
    "pl_eqt",
]

CLUSTERING_FEATURE_SETS = {
    "pdf_core_no_mass_density": [
        "pl_rade",
        "pl_orbper",
        "pl_orbsmax",
        "pl_orbeccen",
        "st_teff",
        "st_met",
        "sy_pnum",
    ],
    "pdf_core_with_mass_density": [
        "pl_rade",
        "pl_bmasse",
        "pl_dens",
        "pl_orbper",
        "pl_orbsmax",
        "pl_orbeccen",
        "st_teff",
        "st_met",
        "sy_pnum",
    ],
    "habitability": [
        "pl_rade",
        "pl_orbper",
        "pl_orbsmax",
        "st_teff",
        "pl_insol",
        "pl_eqt",
        "sy_pnum",
    ],
    "wide_physical": [
        "pl_rade",
        "pl_bmasse",
        "pl_dens",
        "pl_orbper",
        "pl_orbsmax",
        "pl_orbeccen",
        "pl_insol",
        "pl_eqt",
        "st_teff",
        "st_met",
        "sy_pnum",
    ],
}

DUPLICATE_UNIT_GROUPS = {
    "planet_radius": ["pl_rade", "pl_radj"],
    "planet_mass": ["pl_bmasse", "pl_bmassj"],
}

NON_MODEL_SUFFIXES = (
    "err1",
    "err2",
    "lim",
    "reflink",
    "systemref",
)

LOG_CANDIDATE_COLUMNS = [
    "pl_rade",
    "pl_bmasse",
    "pl_dens",
    "pl_orbper",
    "pl_orbsmax",
    "pl_insol",
    "pl_eqt",
]

RADIUS_BINS = [-float("inf"), 1.25, 2.0, 4.0, 10.0, float("inf")]
RADIUS_LABELS = [
    "Tierra (<1.25 R_earth)",
    "Supertierra (1.25-2)",
    "Subneptuno (2-4)",
    "Neptuno/transitorio (4-10)",
    "Gigante gaseoso (>10)",
]
