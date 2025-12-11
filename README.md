# Cooperative Membership and Sustainable Development: A K-means Cluster Analysis

## Project Overview

This repository contains all the code and data files necessary to replicate the analysis presented in the paper studying the relationship between cooperative membership and sustainable development outcomes among rural Chinese households. The study uses K-means clustering combined with Multi-Criteria Decision Analysis (MCDA) indices to examine how cooperative membership affects different household clusters.

## Data Source

The raw data comes from the **China Rural Revitalization Survey (CRRS)** conducted by the Chinese Academy of Social Sciences in 2022. The survey covers household demographics, agricultural production, economic activities, health behaviors, and environmental practices.

## Folder Structure

```
project_replication/
├── README.md                 # This documentation file
├── code/                     # Python scripts for data processing and analysis
│   ├── merge_raw.py          # Step 1: Merge raw survey data files
│   ├── build_cluster_df.py   # Step 2: Build feature dataset for clustering
│   ├── run_kmeans.py         # Step 3: Run K-means clustering analysis
│   ├── analyze_clusters.py   # Step 4: Analyze clusters and calculate MCDA indices
│   └── Health_Index.py       # Alternative: Health index calculation script
├── raw_data/                 # Original survey data files
│   ├── hh_3.csv, hh_3_A.csv, hh_3_B.csv  # Household basic info & land
│   ├── hh_5_1.csv, hh_5_2.csv, hh_5_3.csv # Agricultural production
│   ├── hh_6.csv              # Input usage (fertilizer, machinery)
│   ├── hh_7_1.csv, hh_7_2.csv # Livestock production
│   ├── hh_8.csv              # Digital literacy & mobile usage
│   ├── hh_9.csv              # Financial services & credit
│   ├── hh_10.csv             # Environment & sanitation
│   ├── hh_11.csv             # Life satisfaction
│   ├── hh_17.csv             # Income data
│   ├── hh_20.csv             # Health & lifestyle
│   └── zongbiao.csv          # Master household table
└── output_data/              # Generated analysis results
    ├── raw_df.csv            # Merged raw data
    ├── cluster_df.csv        # Processed features for clustering
    ├── cluster_labels.csv    # Cluster assignments
    ├── cluster_with_labels.csv # Cluster assignments with labels
    ├── kmeans_features.csv   # Standardized features matrix
    ├── kmeans_eval_results.csv # K evaluation metrics
    ├── df_all_with_new_mcda.csv # Full dataset with MCDA indices
    ├── cluster_behavior_profile.csv # Behavior profile by cluster
    ├── cluster_member_crosstab.csv  # Cluster × member crosstab
    ├── cluster_mcda_mean.csv        # MCDA means by cluster
    ├── cluster_member_mcda_mean.csv # MCDA means by cluster × member
    ├── cluster_member_gap_mcda.csv  # Member-nonmember gap
    └── cluster_outcome_summary.csv  # Income/satisfaction outcomes
```

## Replication Instructions

### Prerequisites

Install the required Python packages:

```bash
pip install pandas numpy scikit-learn matplotlib scipy
```

### Step-by-Step Execution

**Step 1: Merge Raw Data**
```bash
cd code
python merge_raw.py
```
This script merges multiple household survey files (hh_3.csv through hh_20.csv) into a single `raw_df.csv` file using the household identifier `varID` as the merge key.

**Step 2: Build Clustering Features**
```bash
python build_cluster_df.py
```
This script:
- Cleans and preprocesses variables (handling missing values, encoding categorical variables)
- Creates clustering features including land area, mechanization, fertilizer use, digital literacy, financial access, and health/environmental behaviors
- Applies winsorization (1%-99%) to continuous variables to handle outliers
- Outputs `cluster_df.csv`

**Step 3: Run K-means Clustering**
```bash
python run_kmeans.py
```
This script:
- Standardizes continuous features using Z-score normalization
- Evaluates K values from 2-8 using inertia (elbow method) and silhouette scores
- Performs K-means clustering with K=3 (optimal value based on evaluation)
- Outputs cluster labels and evaluation results

**Step 4: Analyze Clusters**
```bash
python analyze_clusters.py
```
This script:
- Calculates three MCDA indices:
  - **CMI (Capacity and Modernization Index)**: Land resources, mechanization, irrigation, digital capability, financial access
  - **GSPI (Green and Sustainable Production Index)**: Fertilizer practices, waste management, straw utilization
  - **HLEI (Health and Living Environment Index)**: Safe water, sanitation, health checkups, health knowledge
- Computes Overall index as the mean of CMI, GSPI, and HLEI
- Analyzes cooperative membership effects within each cluster
- Generates comparison tables between members and non-members

## Key Variables

### Clustering Features
| Variable | Description | Type |
|----------|-------------|------|
| land_area | Total cultivated land (mu) | Continuous |
| tillage_mech_ratio | Proportion of mechanized tillage | Continuous |
| irrigated_area_ratio | Proportion of irrigated land | Continuous |
| chem_fert_kg_per_mu | Chemical fertilizer per mu (kg) | Continuous |
| organic_fert_total | Total organic fertilizer use | Continuous |
| mobile_hours_per_day | Daily smartphone usage (hours) | Continuous |
| can_get_info_online | Online information access ability | Binary |
| internet_training | Received internet training | Binary |
| e_commerce | Uses e-commerce | Binary |
| ag_insurance | Has agricultural insurance | Binary |
| bank_credit_user | Uses bank credit | Binary |
| applied_bank_loan | Applied for bank loan | Binary |
| trade_credit | Uses trade credit | Binary |
| safe_drinking_water | Has safe drinking water | Binary |
| garbage_sorting | Practices waste sorting | Binary |
| sanitary_toilet | Has sanitary toilet | Binary |
| health_knowledge_learning | Learns health knowledge | Binary |
| has_checkup | Had health checkup | Binary |
| straw_utilized | Utilizes crop straw | Binary |
| pesticide_pack_safe | Safe pesticide packaging disposal | Binary |

### Outcome Variables
| Variable | Source | Description |
|----------|--------|-------------|
| var5_1 | hh_5 | Cooperative membership status |
| var17_32_1 | hh_17 | Household income |
| var11_1-4 | hh_11 | Life satisfaction scores |

## Results Summary

The analysis identifies three distinct household clusters:
- **Cluster 1 (N≈994)**: Lower-capacity households with limited resources
- **Cluster 2 (N≈2113)**: Moderate-capacity households
- **Cluster 3 (N≈627)**: High-capacity households with advanced practices

Cooperative membership shows positive effects on MCDA indices across all clusters, with particularly significant improvements in CMI scores, suggesting that cooperatives primarily benefit households through enhanced capacity building and modernization.

## Contact

For questions about this analysis, please refer to the associated paper or contact the authors.

## License

This code is provided for academic replication purposes. The CRRS data is subject to the data use agreement with the Chinese Academy of Social Sciences.

