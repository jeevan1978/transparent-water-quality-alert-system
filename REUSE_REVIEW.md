# Original Repository Reuse & Compatibility Review

**Project:** Transparent Water Quality Alert System  
**Review Target:** Original Omdena Challenge Repository (`transparent-water-quality-alert-system`)  
**Date of Review:** 2026-10-25  
**Review Status:** Completed (No legacy code or weights copied directly)

---

## 1. Executive Summary

The existing repository originates from an Omdena local chapter challenge titled *"Water Quality Monitoring Dashboard for Kutch Region"*. It was designed to pull satellite imagery from Google Earth Engine (Copernicus Sentinel-2), estimate water index metrics (NDWI), and visualize regional lakes in Gujarat, India (Hamisar, Shinai, Tappar).

While the original repository provides valuable historical domain insights regarding water monitoring workflows, **direct copying or wholesale dependency on its code and serialized assets is strongly discouraged** due to:
1. **Severe Licensing Constraints (GPL-3.0 Copyleft)**
2. **C-Dependency & Platform Incompatibilities (GDAL / OSMnx / GeoPandas on Windows)**
3. **External Service Bottlenecks (Google Earth Engine Authentication & Cloud Quotas)**
4. **Security & Reproducibility Risks with Legacy Pickled Models (`water-model1.pkl`)**

---

## 2. Inventory of Existing Files

| File / Folder | Original Purpose | Assessment for Reuse |
| :--- | :--- | :--- |
| `LICENSE` | GNU General Public License v3.0 (GPL-3.0) | **Critical Legal Implication**: Forces any derivative or combined software distributed to adopt GPL-3.0 with complete source disclosure. |
| `README.md` | Omdena contribution guidelines and task layout | Reference for task breakdown inspiration; not applicable to current architecture. |
| `app.py` | Streamlit multi-page dashboard for Kutch Region | Reference only. Uses deprecated Streamlit options (`showPyplotGlobalUse`), Heroku endpoints, and tightly couples Earth Engine with Folium. |
| `predict.py` | Prediction script combining salinity rules with pickle model | Useful reference for the 3 target categories (`Good`, `Poor`, `Needs Treatment`), but code is fragile and tied to `water-model1.pkl`. |
| `data_collection.py` | Google Earth Engine Sentinel-2 Copernicus pipeline | Requires interactive GEE credentials, geemap, and cloud authorization. High latency, unsuitable for simulated real-time telemetry. |
| `water-model1.pkl` | Pickled pre-trained classifier (450 KB) | **Do not reuse directly.** Pickle deserialization poses arbitrary code execution risks and suffers from scikit-learn version divergence. |
| `src/data/` | Satellite shapefiles and local test datasets | Geodata specific to Kutch lakes; not needed for simulated multi-station telemetry. |
| `src/tasks/` | Historical notebooks and task documents | Useful background reading for students on exploratory data analysis (EDA). |
| `src/references/` | Data dictionaries and research citations | Useful domain knowledge on water parameters (pH, salinity, dissolved oxygen). |
| `src/visualizations/` | Saved static matplotlib/seaborn plots | Historical documentation. |
| `*.png` assets | Lake photos and Omdena India logos | Branding assets; keep untouched in root, do not bundle in new core logic. |

---

## 3. License Analysis: GNU General Public License v3.0 (GPL-3.0)

The original repository is licensed under **GPL-3.0**.

### Key Implications:
- **Strong Copyleft**: If any code, function, or trained model artifact from the original repository is copied into or statically linked with the new system and subsequently distributed, the entire new project **must** be released under GPL-3.0 with its full source code made available.
- **Clean-Room Engineering Recommendation**:
  - We have adopted an isolated clean-room architecture inside `water_quality_system/`.
  - No source code or model weights have been copied from the original root files into `water_quality_system/`.
  - The project conceptually references standard environmental parameters (WHO/EPA safe ranges for pH, DO, Turbidity) and category labels (`Good`, `Needs Treatment`, `Poor`), which are factual domain concepts, avoiding copyright infringement.

---

## 4. Technical Compatibility Issues

### 4.1 Dependency Installation Friction (Especially on Windows)
The original `app.py` imports:
```python
import geopandas as gpd
import geopy
import networkx as nx
import osmnx as ox
import shapely.wkt
import geemap
import ee
```
- **C-Extension Conflicts**: Packages like `geopandas`, `shapely`, and `osmnx` rely on underlying binary C-libraries (`GDAL`, `PROJ`, `GEOS`). On Windows environments, pip installation often crashes unless precompiled wheel binaries or Conda environments are configured.
- **Streamlit Deprecations**: The original app calls `st.set_option('deprecation.showPyplotGlobalUse', False)`, which has been removed in modern Streamlit releases, generating runtime errors.

### 4.2 Google Earth Engine (GEE) Authentication Barrier
- `ee.Authenticate()` and `ee.Initialize()` require each developer to register a Google Cloud project with Earth Engine API enabled and log in via browser OAuth tokens.
- For beginners or automated CI/CD test suites, this creates an immediate barrier to getting the code running.

### 4.3 Pickle Deserialization Risk (`water-model1.pkl`)
- Python's standard `pickle.load()` reconstructs arbitrary Python objects. Loading pickled files from external or legacy sources without cryptographic signatures is a known security vulnerability (CVE risk for Remote Code Execution).
- In addition, scikit-learn models serialized across differing minor or major versions frequently emit warnings or fail to unpickle due to internal attribute schema changes.

---

## 5. Architectural Recommendations for `water_quality_system`

1. **Independent Module Space**: All new development is strictly contained within `water_quality_system/`.
2. **Synthetic Telemetry Simulation**: Replace GEE satellite extraction with an offline, high-frequency mathematical telemetry simulator (`src/data_simulator.py`) supporting diurnal cycles, sensor noise, and controllable pollution anomalies.
3. **Retrain Clean Models**: Use `scripts/train_models.py` to train new, transparent classifiers and regressors using modern `joblib` serialization, complete with training metrics.
4. **Modern UI**: Use native Streamlit and Plotly in `app.py` for interactive real-time telemetry charts, gauge indicators, and alert notifications without heavy GIS C-extensions.
5. **Decoupled Database Access**: Support Supabase PostgreSQL for cloud persistence while offering seamless automatic fallback to local SQLite for offline beginner use.
