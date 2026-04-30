from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .config import project_root


PROJECT_ROOT = project_root()

CATALOG_FALLBACKS = [
    "reports/imputation/PSCompPars_imputed_iterative.csv",
    "reports/imputation/PSCompPars_imputed_knn.csv",
    "reports/imputation/PSCompPars_imputed_median.csv",
    "data/PSCompPars_imputed_iterative.csv",
]


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".json":
        return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    return pd.read_csv(path, low_memory=False)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def write_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def ensure_output_tree(output_dir: str | Path) -> dict[str, Path]:
    base = resolve_repo_path(output_dir)
    tree = {
        "base": base,
        "figures": base / "figures",
        "logs": base / "logs",
    }
    for path in tree.values():
        path.mkdir(parents=True, exist_ok=True)
    return tree


def resolve_repo_path(path_like: str | Path | None) -> Path:
    if path_like is None:
        raise ValueError("Se recibio una ruta nula.")
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def repo_relative(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def safe_hostname(hostname: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", hostname.strip())
    return clean.strip("_") or "unknown_system"


def resolve_existing_path(
    candidate: str | Path | None,
    *,
    logger: logging.Logger,
    label: str,
    fallbacks: Iterable[str] | None = None,
) -> Path:
    attempted: list[str] = []
    if candidate is not None:
        raw = Path(candidate)
        options = [raw]
        if not raw.is_absolute():
            options.append(PROJECT_ROOT / raw)
        for option in options:
            attempted.append(str(option))
            if option.exists():
                return option.resolve()
        basename = raw.name
        if basename:
            matches = sorted(PROJECT_ROOT.rglob(basename))
            if matches:
                logger.warning("No se encontro %s en la ruta pedida; se usara %s.", label, repo_relative(matches[0]))
                return matches[0].resolve()
    for fallback in fallbacks or []:
        option = resolve_repo_path(fallback)
        attempted.append(str(option))
        if option.exists():
            logger.info("Usando fallback para %s: %s", label, repo_relative(option))
            return option
    raise FileNotFoundError(f"No se encontro {label}. Rutas intentadas: {attempted}")


def discover_optional_table(
    explicit: str | Path | None,
    *,
    label: str,
    patterns: list[str],
    preferred_keywords: list[str],
    logger: logging.Logger,
) -> Path | None:
    if explicit:
        try:
            return resolve_existing_path(explicit, logger=logger, label=label)
        except FileNotFoundError:
            logger.warning("No se encontro la ruta explicita de %s; se intentara autodeteccion.", label)
    candidates: list[Path] = []
    for root_name in ["outputs", "reports"]:
        root = PROJECT_ROOT / root_name
        if not root.exists():
            continue
        for pattern in patterns:
            candidates.extend(root.rglob(pattern))
    unique_candidates = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen and candidate.is_file():
            seen.add(resolved)
            unique_candidates.append(resolved)
    if not unique_candidates:
        logger.warning("No se encontro %s; el pipeline seguira con fallback nulo.", label)
        return None

    def rank(path: Path) -> tuple[int, int, int, str]:
        lower = str(path).lower()
        matched = sum(1 for keyword in preferred_keywords if keyword in lower)
        orbital_bonus = 1 if "orbital" in lower else 0
        shadow_bonus = 1 if "shadow_inputs" in lower else 0
        return (-matched, -orbital_bonus - shadow_bonus, len(path.parts), lower)

    best = sorted(unique_candidates, key=rank)[0]
    logger.info("Autodetectada %s en %s", label, repo_relative(best))
    return best


def _derive_semimajor_axis(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if "pl_orbsmax" not in out.columns:
        out["pl_orbsmax"] = np.nan
    out["pl_orbper"] = pd.to_numeric(out.get("pl_orbper"), errors="coerce")
    out["pl_orbsmax"] = pd.to_numeric(out.get("pl_orbsmax"), errors="coerce")
    out["st_mass"] = pd.to_numeric(out.get("st_mass"), errors="coerce")

    observed = out["pl_orbsmax"].notna() & np.isfinite(out["pl_orbsmax"]) & (out["pl_orbsmax"] > 0)
    derivable = (~observed) & out["pl_orbper"].notna() & (out["pl_orbper"] > 0) & out["st_mass"].notna() & (out["st_mass"] > 0)
    out["pl_orbsmax_system_module_source"] = "missing"
    out.loc[observed, "pl_orbsmax_system_module_source"] = "observed"
    out.loc[derivable, "pl_orbsmax"] = np.power(out.loc[derivable, "st_mass"] * np.power(out.loc[derivable, "pl_orbper"] / 365.25, 2.0), 1.0 / 3.0)
    out.loc[derivable, "pl_orbsmax_system_module_source"] = "derived_kepler"
    return out


def load_catalog(catalog: str | Path | None, logger: logging.Logger) -> tuple[Path, pd.DataFrame]:
    if catalog is None:
        path = resolve_existing_path(None, logger=logger, label="catalogo", fallbacks=CATALOG_FALLBACKS)
    else:
        path = resolve_existing_path(catalog, logger=logger, label="catalogo", fallbacks=CATALOG_FALLBACKS)
    frame = read_table(path)
    frame = _derive_semimajor_axis(frame)
    required = {"hostname", "pl_name", "pl_orbper"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise KeyError(f"El catalogo no contiene columnas requeridas: {missing}")
    numeric_columns = [
        "pl_orbper",
        "pl_orbsmax",
        "pl_bmasse",
        "pl_rade",
        "pl_dens",
        "st_mass",
        "st_rad",
        "disc_year",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["hostname"] = frame["hostname"].astype("string")
    frame["pl_name"] = frame["pl_name"].astype("string")
    if "discoverymethod" not in frame.columns:
        frame["discoverymethod"] = "Unknown"
    frame["discoverymethod"] = frame["discoverymethod"].astype("string").fillna("Unknown")
    if "disc_facility" not in frame.columns:
        frame["disc_facility"] = pd.NA
    logger.info("loaded catalog with %s planets from %s", len(frame), repo_relative(path))
    return path, frame
