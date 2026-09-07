# ============================================================
# CONUS Agricultural Water-Scarcity / Dryland Drought Forecast
# ============================================================
#
# Domain delineation follows the UNEP (1992) / IPCC AR6 (2021) Aridity Index
# (AI = P / PET) framework — NOT geographic bounding boxes.
# Primary source: Trabucco & Zomer (2019) CGIAR-CSI Global Aridity & PET
# Database v3 (https://cgiarcsi.community).
# Classification thresholds: Middleton & Thomas (1997) World Atlas of
# Desertification, 2nd ed.; Prăvălie (2016) Earth-Sci. Rev. 161:259–278.
#
# Reporting zones (SGP, NGP, etc.) are applied ONLY after AI masking for
# sub-regional result disaggregation. They do NOT define the target domain.
# ============================================================

# ── Spatial processing extent (broad CONUS; actual domain = AI mask ∩ CDL) ──
#BBOX = (-125.0, 25.0, -66.0, 50.0)     # lon_min, lat_min, lon_max, lat_max
BBOX = (-125.0, 25.0, -66.0, 54.0)   # this is the GEE coordinates downloaded
RES  = 0.09                            # ~10 km, matches SMAP L4 EASE-2 native
CRS  = "EPSG:4326"
# config.py
#SEEDS = [42, 123, 777, 2024, 31337]
# config.py
SEEDS = [42, 123, 777, 2024, 31337,
         7, 99, 512, 1234, 4096, 8191, 20250, 60613, 77777, 90210]
# ── UNEP/IPCC Aridity Index classification ──
# AI = annual precipitation (P) / potential evapotranspiration (PET)
# Data source: CGIAR-CSI ai_v3_yr.tif (stored as AI * 10000)
# Reference: UNEP 1992; Feng & Fu 2013 Nat. Clim. Change; IPCC AR6 Ch.8
ARIDITY_CLASSES = {
    "hyper_arid":   (0.00, 0.05),   # Excluded: negligible agricultural NIRv
    "arid":         (0.05, 0.20),   # Included: sparse dryland / grazing
    "semi_arid":    (0.20, 0.50),   # Core: primary RZSM-skill zone
    "dry_subhumid": (0.50, 0.65),   # Included: transitional, sensitivity buffer
    "humid":        (0.65, 9.99),   # Excluded: energy-limited, control only
}

# Overall dryland mask used in model training (excludes hyper-arid + humid)
DRYLAND_AI_MIN = 0.05
DRYLAND_AI_MAX = 0.65

# Strict semi-arid mask for primary results (mirrors Africa study definition)
STRICT_SEMI_ARID_AI_MIN = 0.20
STRICT_SEMI_ARID_AI_MAX = 0.50

# ── Agricultural system sub-masks ──
# Applied on top of AI mask. Pixels must satisfy BOTH AI range AND crop type.
USE_CROPLAND_MASK  = True   # USDA CDL 30 m → resampled to SMAP grid
USE_RAINFED_MASK   = True   # USGS IWUM irrigation fraction < 0.15
USE_IRRIGATED_MASK = True   # USGS IWUM irrigation fraction >= 0.15

AG_SYSTEMS = {
    # Primary analysis group: mirrors Africa rainfed cropland setting
    "rainfed_dryland": {
        "use_crop_mask":      True,
        "use_rainfed_mask":   True,
        "use_irrigated_mask": False,
        "ai_min": DRYLAND_AI_MIN,
        "ai_max": DRYLAND_AI_MAX,
        "note": "Rainfed cropland in drylands — primary RZSM-skill domain",
    },
    # Irrigated drylands: test irrigation deconvolution
    "irrigated_dryland": {
        "use_crop_mask":      True,
        "use_rainfed_mask":   False,
        "use_irrigated_mask": True,
        "ai_min": DRYLAND_AI_MIN,
        "ai_max": DRYLAND_AI_MAX,
        "note": "Irrigated cropland in drylands — RZSM confounding zone",
    },
    # All dryland agriculture (rainfed + irrigated combined)
    "all_ag_dryland": {
        "use_crop_mask":      True,
        "use_rainfed_mask":   False,
        "use_irrigated_mask": False,
        "ai_min": DRYLAND_AI_MIN,
        "ai_max": DRYLAND_AI_MAX,
        "note": "All cropland within drylands",
    },
    # Strict semi-arid only — for direct Africa comparison
    "strict_semi_arid_ag": {
        "use_crop_mask":      True,
        "use_rainfed_mask":   False,
        "use_irrigated_mask": False,
        "ai_min": STRICT_SEMI_ARID_AI_MIN,
        "ai_max": STRICT_SEMI_ARID_AI_MAX,
        "note": "AI 0.20–0.50 cropland — mirrors Africa study stratification",
    },
}

# ── Reporting zones (post-masking disaggregation ONLY) ──
# These are named for convenient sub-regional reporting.
# They do NOT replace the AI + CDL domain definition.
# Boundaries drawn from NOAA Climate Regions and USDA Farm Resource Regions.
REPORTING_ZONES = {
    "Southern_Great_Plains": (-105, 32, -94, 40),   # NOAA S. Plains + USDA Region 6
    "Northern_Great_Plains": (-110, 40, -95, 50),   # NOAA N. Plains + USDA Region 7
    "Desert_Southwest":      (-118, 31, -103, 38),   # NOAA SW + USDA Region 9 (partial)
    "Intermountain_West":    (-120, 37, -108, 49),   # NOAA NW (interior) + USDA Region 9
    "Central_Valley_CA":     (-123, 34.5, -118, 41), # California AG belt
}

# ── Time range ──
START_DATE = "2015-04-01"   # SMAP L4 operations began April 2015
END_DATE   = "2024-12-31"

#START_DATE = "2015-06-01"   # SMAP L4 operations began April 2015
#END_DATE   = "2015-06-30"

# ── Model parameters ──
CONTEXT_LEN = 80           # days of lookback (finding: 80+ days optimal for 40-day lead)
LEAD_DAYS   = [8, 16, 24, 32, 40]
HIDDEN_DIM  = 64
KERNEL_SIZE = 3
NUM_LAYERS  = 2
BATCH_SIZE  = 4            # spatiotemporal batches are VRAM-heavy
LR          = 1e-4
EPOCHS      = 100

# ── Paths ──
DATA_ROOT         = "data/raw"
PROC_ROOT         = "data/processed"
ARIDITY_DIR       = f"{DATA_ROOT}/aridity"
IRRIGATION_DIR    = f"{DATA_ROOT}/irrigation"
CDL_DIR           = f"{DATA_ROOT}/cdl"
SMAP_DIR          = f"{DATA_ROOT}/smap"
PRISM_DIR         = f"{DATA_ROOT}/prism"
MODIS_DIR         = f"{DATA_ROOT}/modis"
OUTPUT_MAPS       = "outputs/maps"
OUTPUT_METRICS    = "outputs/metrics"
OUTPUT_MITIGATION = "outputs/mitigation"
