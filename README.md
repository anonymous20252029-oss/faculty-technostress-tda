# 🧠 Topological Data Analysis (TDA) for Faculty Cognitive Load & Technostress Benchmarking

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end empirical research pipeline combining **Topological Data Analysis (Persistent Homology & Kepler-Mapper)** and **Interactive Micro-Surveying** to quantify, benchmark, and visualize higher-education faculty cognitive load under rapid AI and digital tooling adoption.

---

## 📌 Overview & Research Rationale

While classical psychometric models (e.g., linear SEM, Pearson correlations) struggle to capture non-linear phase transitions and cyclic burnout traps, this project utilizes **Topological Data Analysis (TDA)** to:
1. **Harmonize Heterogeneous Benchmarks ($N = 3,459$):** Integrates empirical data from IT industry professionals (`OSMI Tech`), university students (`Student Stress Factors`), and tertiary academic staff (`Higher Ed Wellbeing`) into a unified 3D metric state-space based on the **Job Demands-Resources (JD-R)** paradigm.
2. **Extract Topological Invariants:** Identifies multi-scale cluster coalescence ($H_0 = 682$) and recurrent cyclic feedback loops ($H_1 = 166$ persistent holes via Vietoris–Rips filtration).
3. **Real-Time Live Cohort Benchmarking:** Projects in-situ micro-survey responses ($n \approx 30$) collected via a lightweight Streamlit interface directly onto the global topological manifold.
---
## 🏗️ System Architecture

The pipeline consists of five interconnected phases:

```text
[Phase 1: Multi-Source Inputs]
   ├── OSMI Tech Industry (N=1,259)
   ├── Student Stress Factors (N=1,100)
   └── Higher Ed Wellbeing (N=1,100)
          │
          ▼
[Phase 2: Semantic Harmonization (JD-R)]
   ├── F1: Burnout Severity
   ├── F2: AI & Technostress
   └── F3: Cognitive Latency Proxy
   └── Stacking into Matrix X ∈ ℝ^(3459 × 3) on [1.0, 5.0]³
          │
          ├─────────────────────────────────────────┐
          ▼                                         ▼
[Phase 3A: Vietoris–Rips Filtration]     [Phase 3B: Kepler-Mapper Pipeline]
   └── Ripser boundary reduction            └── Nerve complex (|V|=260, |E|=484)
          │                                         │
          ▼                                         ▼
[Phase 4A: Persistence Signatures]       [Phase 4B: Topological Skeleton & Bridges]
   └── H0 (682) & H1 Loops (166)            └── Cross-domain boundary transition
          │                                         │
          └────────────────────┬────────────────────┘
                               ▼
[Phase 5: Live Streamlit Deployment & In-Situ Benchmarking]
   ├── QR-Code live onboarding
   ├── Silent latency timer (ms)
   └── Real-time out-of-sample projection & cohort centroid (★)
```
---
## 📂 Repository Structure

```text
faculty-technostress-tda/
│
├── .streamlit/
│   └── config.toml                  # Streamlit UI theme and layout settings
│
├── data/
│   ├── global_aligned_real_dataset.csv     # Harmonized multi-cohort baseline (N=3,459)
│   ├── tda_topology_metrics.csv            # Extracted Betti numbers and persistent metrics
│   └── pilot_survey_cntt_30_responses.csv  # Real-time recorded cohort responses
│
├── figures/
│   ├── empirical_multidataset_vivid_en.png # 2D PCA comparative state-space
│   ├── comparative_full_vs_mapper_en.png   # Full point cloud vs. Mapper skeleton
│   └── methodology_pipeline_spacious_en.png# End-to-end architecture diagram
│
├── app.py                           # Main interactive Streamlit application
├── requirements.txt                 # Python dependencies
└── README.md                        # Project documentation
```
## 🚀 Quickstart & Local Installation

### 1. Clone the repository
```bash
git clone https://github.com/anonymous20252029-oss/faculty-technostress-tda.git
cd faculty-technostress-tda
```

### 2. Set up Python environment
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
*(On Windows: `venv\Scripts\activate`)*

### 3. Launch the Streamlit application
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`.

---

## 📊 Live Interactive Protocol (Streamlit App)

* **Silent Cognitive Timer:** Tracks the duration (in milliseconds) spent evaluating survey items to proxy cognitive hesitation and internal friction.
* **Instant State-Space Projection:** Upon submission, the user's standardized 3D vector $[F_1, F_2, F_3]$ is projected via pre-fitted PCA transformations onto the continuous global manifold.
* **Dynamic Centroid Tracking:** Automatically recalculates the department's collective state centroid ($\star$) and updates the 95% variance confidence boundary in real time.

---

## 🔬 Citation & Academic Use

If you use this methodology, codebase, or harmonized dataset in your research, please cite:

```bibtex
@article{tda_faculty_technostress_2026,
  title={Topological Data Analysis of Academic Technostress: Multi-Domain Harmonization and Real-Time State-Space Benchmarking},
  author={Vo, Thi Kim Anh and Collaborators},
  journal={Working Paper / Research Archive},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
