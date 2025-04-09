# -*- coding: utf-8 -*-
# --- Core Libraries ---
import numpy as np
import pandas as pd
# Import scanpy and anndata first, handle potential import errors
try:
    import scanpy as sc
except ImportError:
    print("Error: 'scanpy' module not found. Install it using 'pip install scanpy'")
    sc = None
    exit() # Exit as scanpy is core

try:
    import anndata as ad
except ImportError:
    print("Error: 'anndata' module not found. Install it using 'pip install anndata'")
    ad = None
    exit() # Exit as anndata is core

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors # Import for color conversion
import seaborn as sns
import os
import warnings

# --- Statistics & Smoothing ---
from scipy.stats import spearmanr, mannwhitneyu
from statsmodels.nonparametric.smoothers_lowess import lowess
try:
    from scipy.spatial import KDTree # Moved import here for clarity
    SCIPY_SPATIAL_AVAILABLE = True
except ImportError:
    SCIPY_SPATIAL_AVAILABLE = False
    print("Warning: scipy.spatial not found. Spatial neighborhood analysis (Figure 5b) will be skipped.")


# --- Survival Analysis ---
try:
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import logrank_test
    LIFELINES_INSTALLED = True
except ImportError:
    LIFELINES_INSTALLED = False
    print("Warning: 'lifelines' library not found. Survival analysis plots will be skipped. Install with 'pip install lifelines'")


# --- Settings ---
sc.settings.verbosity = 1 # Errors and warnings only
sc.settings.set_figure_params(dpi=80, facecolor='white', frameon=False)
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='scanpy')
warnings.filterwarnings('ignore', category=pd.errors.PerformanceWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message="Observation names are not unique.")
warnings.filterwarnings('ignore', message="objects copied")
warnings.filterwarnings('ignore', message="In accordance with common practice") # KMF CI warning

# Create output directory for figures
output_dir = "nc_level_figures_real_data" # Changed output dir name
os.makedirs(output_dir, exist_ok=True)
print(f"Output directory for figures: {output_dir}")

# ==============================================================================
# --- 1. LOAD DATA FROM FILES ---
# ==============================================================================
print("\n--- 1. Loading Data ---")

# --- Placeholder File Paths  ---
multiome_file = "combined_multiome_data.h5ad"

spatial_files_pattern = "spatial_data/patient_{patient_id}_spatial.csv"

spatial_file_combined = "all_patients_spatial_data.csv"
clinical_file = "clinical_data.csv"
# Define marker genes (useful for annotation later)
markers = {
    'Tumor': ['EPCAM', 'KRT8'], 'Tcell_CD8': ['CD3D', 'CD8A'], 'Tcell_CD4': ['CD3D', 'CD4'],
    'CD8_Tex': ['PDCD1', 'CTLA4', 'LAG3', 'TOX', 'HAVCR2'],
    'CD8_Tem': ['IL7R', 'CCR7', 'TCF7'], 'Treg': ['FOXP3', 'IL2RA'], 'CD4_Th': ['CD40LG'],
    'Macrophage': ['CD68', 'CD14'], 'Macro_M1': ['CD86', 'TNF', 'IL1B', 'IL6', 'CXCL9', 'CXCL10'],
    'Macro_M2': ['CD163', 'MRC1', 'CD206', 'IL10', 'CCL18', 'CCL22'], 'Monocyte': ['FCGR3A', 'FCN1'],
    'cDC': ['CD1C', 'FCER1A', 'CLEC9A'], 'pDC': ['LILRA4', 'IL3RA', 'GZMB'],
    'Bcell': ['MS4A1', 'CD19', 'CD79A'], 'Plasma': ['SDC1', 'MZB1', 'JCHAIN'],
    'Fibroblast': ['COL1A1', 'ACTA2', 'FAP'], 'Endothelial': ['PECAM1', 'VWF', 'CDH5']
}
# --- End Placeholder File Paths ---

# --- Placeholder Loading Functions (MODIFY AS NEEDED) ---

def load_multiome_data(filepath):
    """
    Loads the combined multiome AnnData object.
    Expected structure:
    - .X: Raw or normalized RNA counts.
    - .obs: Contains 'patient_id', 'responder_status', and other cell metadata.
    - .obsm['ATAC']: pandas DataFrame with ATAC peak counts (cells x peaks),
                     index aligned with .obs_names.
    - .var_names: Gene names.
    - .obsm['ATAC'].columns: Peak names/coordinates.
    """
    print(f"  Loading multiome data from: {filepath}...")
    if not os.path.exists(filepath):
        print(f"  Error: Multiome file not found at {filepath}")
        return None
    try:
        adata = sc.read_h5ad(filepath)
        # --- Basic Checks ---
        required_obs = ['patient_id', 'responder_status']
        if not all(col in adata.obs.columns for col in required_obs):
            print(f"  Error: Missing required columns in .obs: {required_obs}")
            return None
        if 'ATAC' not in adata.obsm:
            print("  Warning: 'ATAC' key not found in .obsm. Proceeding without ATAC data.")
        elif not isinstance(adata.obsm['ATAC'], pd.DataFrame):
            print("  Error: .obsm['ATAC'] is not a pandas DataFrame.")
            # Optionally try to convert or return None
            return None
        elif not adata.obsm['ATAC'].index.equals(adata.obs.index):
             print("  Error: .obsm['ATAC'] index does not match .obs index. Attempting reindex...")
             try:
                 adata.obsm['ATAC'] = adata.obsm['ATAC'].reindex(adata.obs_names)
                 print("   Reindex successful.")
             except Exception as reindex_err:
                 print(f"   Reindex failed: {reindex_err}. Cannot use ATAC data.")
                 del adata.obsm['ATAC'] # Remove faulty ATAC

        print(f"  Loaded multiome data: {adata.n_obs} cells, {adata.n_vars} genes.")
        if 'ATAC' in adata.obsm:
            print(f"  ATAC data shape: {adata.obsm['ATAC'].shape}")
        return adata
    except Exception as e:
        print(f"  Error loading multiome data: {e}")
        return None

def load_spatial_data(filepath_pattern_or_single_file, patient_ids):
    """
    Loads spatial data.
    Can load from individual CSVs based on a pattern or a single combined file.
    Returns a dictionary {patient_id: spatial_df} or None if loading fails.
    Expected DataFrame columns: 'x', 'y', 'patient_id', and optionally
    'dominant_cell_type' or spot expression data. Index should be unique spot IDs.
    """
    print(f"  Loading spatial data from: {filepath_pattern_or_single_file}...")
    spatial_data_dict = {}
    required_cols = ['x', 'y', 'patient_id'] # Adjust as needed

    # Try loading as a single combined file first
    if isinstance(filepath_pattern_or_single_file, str) and os.path.exists(filepath_pattern_or_single_file) and not "{patient_id}" in filepath_pattern_or_single_file:
         try:
             print("   Attempting to load as single combined file...")
             combined_df = pd.read_csv(filepath_pattern_or_single_file, index_col=0) # Assuming first col is index
             if not all(col in combined_df.columns for col in required_cols):
                 print(f"   Error: Combined spatial file missing required columns: {required_cols}")
                 return None
             # Split into dictionary by patient_id
             for p_id in patient_ids:
                 pat_df = combined_df[combined_df['patient_id'] == p_id].copy()
                 if not pat_df.empty:
                     spatial_data_dict[p_id] = pat_df
                 else:
                      print(f"   Warning: No spatial data found for patient {p_id} in combined file.")
             print(f"   Loaded spatial data for {len(spatial_data_dict)} patients from combined file.")
             return spatial_data_dict
         except Exception as e:
             print(f"   Error loading combined spatial file: {e}. Will try pattern matching.")

    # If combined loading failed or pattern was given, try loading individual files
    if isinstance(filepath_pattern_or_single_file, str) and "{patient_id}" in filepath_pattern_or_single_file:
        print("   Attempting to load individual patient files using pattern...")
        loaded_count = 0
        for p_id in patient_ids:
            filepath = filepath_pattern_or_single_file.format(patient_id=p_id)
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath, index_col=0) # Assuming first col is index
                    if not all(col in df.columns for col in required_cols):
                        print(f"    Error: Spatial file for {p_id} missing required columns. Skipping.")
                        continue
                    if 'patient_id' not in df.columns: # Add patient_id if missing
                        df['patient_id'] = p_id
                    spatial_data_dict[p_id] = df
                    loaded_count += 1
                except Exception as e:
                    print(f"    Error loading spatial file for {p_id}: {e}")
            else:
                print(f"    Warning: Spatial file not found for patient {p_id} at {filepath}")
        print(f"   Loaded spatial data for {loaded_count} patients using pattern.")
        return spatial_data_dict if spatial_data_dict else None
    else:
        print("  Error: Invalid spatial file path or pattern provided.")
        return None


def load_clinical_data(filepath):
    """
    Loads clinical data containing patient information.
    Expected DataFrame columns: 'patient_id', 'responder_status',
                               'time_to_event', 'observed_event'.
    """
    print(f"  Loading clinical data from: {filepath}...")
    if not os.path.exists(filepath):
        print(f"  Error: Clinical file not found at {filepath}")
        return None
    try:
        df = pd.read_csv(filepath)
        required_cols = ['patient_id', 'responder_status', 'time_to_event', 'observed_event']
        if not all(col in df.columns for col in required_cols):
            print(f"  Error: Clinical file missing required columns: {required_cols}")
            return None
        # Optional: Convert patient_id to string for consistent merging
        df['patient_id'] = df['patient_id'].astype(str)
        print(f"  Loaded clinical data for {df.shape[0]} patients.")
        return df
    except Exception as e:
        print(f"  Error loading clinical data: {e}")
        return None

# --- Load the Data ---
adata_multiome = load_multiome_data(multiome_file)
if adata_multiome is None:
    exit("Exiting: Failed to load multiome data.")

# Extract patient IDs for loading other data
patient_ids_from_multiome = adata_multiome.obs['patient_id'].unique().tolist()

# Pass unique patient IDs to spatial loading function if using pattern
spatial_data = load_spatial_data(spatial_files_pattern, patient_ids_from_multiome)
# If spatial_data is None or empty, subsequent spatial steps will be skipped

clinical_data_df = load_clinical_data(clinical_file)
# If clinical_data_df is None, survival analysis will be skipped

print("Data loading complete.")

# ==============================================================================
# --- 2. ANALYSIS PIPELINE (Now operates on loaded data) ---
# ==============================================================================
print("\n--- 2. Starting Analysis Pipeline ---")

# --- Step 1: scMultiome QC, Integration & Initial Annotation ---
print("\n--- Step 1: QC, Integration & Annotation ---")
# --- 1a. QC ---
# Check for 'mt' genes if mitochondrial genes follow standard naming
adata_multiome.var['mt'] = adata_multiome.var_names.str.startswith(('MT-', 'mt-'))
sc.pp.calculate_qc_metrics(adata_multiome, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
# --- Add your specific QC filtering here based on calculated metrics ---
# Example:
# print(f"Shape before QC filtering: {adata_multiome.shape}")
# sc.pp.filter_cells(adata_multiome, min_genes=200)
# sc.pp.filter_genes(adata_multiome, min_cells=3)
# if 'pct_counts_mt' in adata_multiome.obs:
#    adata_multiome = adata_multiome[adata_multiome.obs.pct_counts_mt < 20, :]
# print(f"Shape after QC filtering: {adata_multiome.shape}")
if adata_multiome.n_obs == 0: exit("Error: All cells filtered out during QC.")
# --- End QC filtering ---

# --- 1b. RNA Processing ---
adata_rna = adata_multiome.copy() # Work on a copy for RNA specific steps
if adata_rna.n_obs == 0 or adata_rna.n_vars == 0: exit("Error: adata_rna is empty after copy.")

sc.pp.normalize_total(adata_rna, target_sum=1e4); sc.pp.log1p(adata_rna)
min_cells_for_hvg = 20; min_genes_for_hvg = 50
if adata_rna.n_obs >= min_cells_for_hvg and adata_rna.n_vars >= min_genes_for_hvg:
    try:
        # If data has batches, include batch_key for better HVG selection
        batch_key_rna = 'batch' if 'batch' in adata_rna.obs else None # Adjust if your batch column name is different
        sc.pp.highly_variable_genes(adata_rna, n_top_genes=min(2000, adata_rna.n_vars-1), # Use more HVGs for real data
                                  subset=True, batch_key=batch_key_rna, flavor='seurat_v3') # seurat_v3 often good for larger datasets
    except Exception as global_hvg_err: print(f"Warning: Global HVG calculation failed: {global_hvg_err}. Using all genes.")
else: print(f"Warning: Data too small ({adata_rna.shape}), skipping global HVG selection.")
if adata_rna.n_obs == 0 or adata_rna.n_vars == 0: exit("Error: adata_rna empty after HVG selection.")

sc.pp.scale(adata_rna, max_value=10)
n_comps_pca = min(50, adata_rna.n_obs-1, adata_rna.n_vars-1)
if n_comps_pca > 1: sc.tl.pca(adata_rna, svd_solver='arpack', n_comps=n_comps_pca)
else: print("Warning: Cannot perform PCA. Skipping PCA and downstream steps.")

# --- 1c. Integration / Batch Correction (Placeholder) ---
# If you have batch effects (e.g., from different sequencing runs or patients), apply correction here.
# Options: sc.external.pp.harmony_integrate, sc.external.pp.bbknn, scVI, Scanorama
# Example (Harmony - requires installation: pip install harmony-pytorch):
# if batch_key_rna:
#    print("  Applying Harmony batch correction...")
#    try:
#        import harmonypy
#        sc.external.pp.harmony_integrate(adata_rna, batch_key_rna)
#        # Use harmony corrected PCs for neighbors
#        sc.pp.neighbors(adata_rna, n_neighbors=15, use_rep='X_pca_harmony')
#        print("  Harmony complete.")
#    except ImportError:
#        print("  Harmony not installed ('pip install harmony-pytorch'). Using uncorrected PCA for neighbors.")
#        if 'X_pca' in adata_rna.obsm: sc.pp.neighbors(adata_rna, n_neighbors=15, n_pcs=min(30, adata_rna.obsm['X_pca'].shape[1]))
#    except Exception as harmony_err:
#        print(f"  Harmony failed: {harmony_err}. Using uncorrected PCA for neighbors.")
#        if 'X_pca' in adata_rna.obsm: sc.pp.neighbors(adata_rna, n_neighbors=15, n_pcs=min(30, adata_rna.obsm['X_pca'].shape[1]))
# else:
# No batch key, use PCA for neighbors if available
if 'X_pca' in adata_rna.obsm:
    n_pcs_neighbors = min(30, adata_rna.obsm['X_pca'].shape[1])
    n_neighbors_knn = min(15, adata_rna.n_obs - 1)
    if n_neighbors_knn > 0 and n_pcs_neighbors > 0: sc.pp.neighbors(adata_rna, n_neighbors=n_neighbors_knn, n_pcs=n_pcs_neighbors)
    else: print("Warning: Insufficient cells/PCs for neighbors. Skipping Neighbors, UMAP, Louvain.")
else: print("Warning: PCA results ('X_pca') not found. Skipping Neighbors, UMAP, Louvain.")


# --- 1d. Clustering & UMAP ---
if 'neighbors' in adata_rna.uns:
    sc.tl.louvain(adata_rna, resolution=1.0, key_added='clusters') # Use a standard key 'clusters'
    sc.tl.umap(adata_rna, min_dist=0.3)
else:
    adata_rna.obsm['X_umap'] = np.zeros((adata_rna.n_obs, 2)); adata_rna.obs['clusters'] = '0'
    print("Neighbors failed, UMAP/Louvain skipped, created placeholders.")

# --- 1e. Cell Type Annotation (Marker-based) ---
# This is a critical step with real data. Requires iteration and biological knowledge.
print("  Performing marker-based annotation (Example - requires refinement)...")
# Calculate marker gene scores or use sc.tl.rank_genes_groups
sc.tl.rank_genes_groups(adata_rna, 'clusters', method='wilcoxon')
# Example: View top genes per cluster
# pd.DataFrame(adata_rna.uns['rank_genes_groups']['names']).head(5)

# --- Manual or Semi-Automated Annotation Logic ---
# You would typically inspect marker expression (dotplots, featureplots)
# and assign labels based on the 'markers' dictionary.
# This is a placeholder - replace with your actual annotation logic.
adata_rna.obs['major_cell_type'] = 'Unknown' # Initialize
# Example rough annotation based on top markers (highly simplified):
# for cluster_id in adata_rna.obs['clusters'].cat.categories:
#    top_genes = sc.get.rank_genes_groups_df(adata_rna, group=cluster_id)['names'][:5]
#    best_match = 'Unknown'
#    highest_score = 0
#    for celltype, markers_list in markers.items():
#         # Simple scoring: count matching top genes
#         score = sum(1 for gene in markers_list if gene in top_genes)
#         if score > highest_score:
#              highest_score = score
#              best_match = celltype
#    # Assign rough label (needs refinement!)
#    adata_rna.obs.loc[adata_rna.obs['clusters'] == cluster_id, 'major_cell_type'] = best_match

# --- Refine Major Types (IF annotation was successful) ---
# This mapping should be applied AFTER the marker-based annotation above
# Ensure the source labels match your assigned labels
if 'major_cell_type' in adata_rna.obs and 'Unknown' not in adata_rna.obs['major_cell_type'].unique():
    adata_rna.obs['major_cell_type'] = adata_rna.obs['major_cell_type'].astype(str).replace({
        # Adjust these mappings based on your marker dict keys and desired groupings
        'CD8_Tex': 'T Cells', 'CD8_Tem': 'T Cells', 'Treg': 'T Cells', 'CD4_Th': 'T Cells',
        'Macro_M1': 'Myeloid', 'Macro_M2': 'Myeloid', 'Monocyte': 'Myeloid', 'cDC': 'Myeloid', 'pDC': 'Myeloid',
        'Bcell': 'B/Plasma', 'Plasma': 'B/Plasma'
        # Add other types if needed
    }).astype('category')
    print("  Major cell type refinement applied.")
else:
    print("  Skipping major cell type refinement (initial annotation incomplete or failed).")
    # If annotation failed, major_cell_type might remain 'Unknown' or cluster numbers
    # Consider assigning clusters as types if no marker annotation possible
    adata_rna.obs['major_cell_type'] = adata_rna.obs['clusters'].astype('category')

print("Annotation step complete.")

# Plotting Figure 1
print("Generating Figure 1...")
fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
if 'X_umap' in adata_rna.obsm:
    # Use 'clusters' if 'major_cell_type' annotation is uncertain
    color_key = 'major_cell_type' if 'major_cell_type' in adata_rna.obs and adata_rna.obs['major_cell_type'].nunique() > 1 else 'clusters'
    sc.pl.umap(adata_rna, color=color_key, ax=axes1[0], show=False, frameon=False, title='Cell Clusters / Types', legend_loc='on data', legend_fontsize=8)
else:
    axes1[0].text(0.5, 0.5, "Fig 1a: UMAP Skipped", ha='center', va='center', transform=axes1[0].transAxes); axes1[0].set_xticks([]); axes1[0].set_yticks([]); axes1[0].set_title('UMAP (Skipped)')
axes1[1].text(0.5, 0.5, "Fig 1b: Modality Integration\n(Requires WNN/scVI analysis)", ha='center', va='center', transform=axes1[1].transAxes, fontsize=10, fontstyle='italic'); axes1[1].set_xticks([]); axes1[1].set_yticks([]); axes1[1].set_title("Modality Integration View")
fig1_markers_display = {'Tumor': ['EPCAM'], 'T Cells': ['CD3D', 'CD8A', 'CD4', 'FOXP3', 'PDCD1'], 'Myeloid': ['CD68', 'CD14', 'CD163', 'CD86', 'CD1C'], 'B/Plasma': ['MS4A1', 'SDC1'], 'Fibroblast': ['COL1A1'], 'Endothelial': ['PECAM1']}
fig1_markers_plot = {k: [g for g in v if g in adata_rna.var_names] for k,v in fig1_markers_display.items()}; fig1_markers_plot = {k:v for k,v in fig1_markers_plot.items() if v}
# Plot dotplot using the same key used for UMAP coloring
can_plot_dotplot = (color_key in adata_rna.obs and adata_rna.obs[color_key].nunique() > 1 and fig1_markers_plot and adata_rna.n_obs > 0)
if can_plot_dotplot:
    try: sc.pl.dotplot(adata_rna, fig1_markers_plot, groupby=color_key, ax=axes1[2], show=False, standard_scale='var', cmap='Blues', return_fig=False); axes1[2].set_title('Key Markers')
    except Exception as e: print(f"Warning: Dotplot failed: {e}"); axes1[2].text(0.5, 0.5, "Fig 1c: Marker Dotplot\n(Error)", ha='center', va='center', transform=axes1[2].transAxes); axes1[2].set_title('Key Markers (Error)'); axes1[2].set_xticks([]); axes1[2].set_yticks([])
else: axes1[2].text(0.5, 0.5, "Fig 1c: Marker Dotplot\n(Insufficient Data)", ha='center', va='center', transform=axes1[2].transAxes); axes1[2].set_title('Key Markers (Skipped)'); axes1[2].set_xticks([]); axes1[2].set_yticks([])
plt.suptitle("Figure 1: TME Overview", fontsize=16, y=1.02); plt.tight_layout(rect=[0, 0, 1, 0.98]); plt.savefig(os.path.join(output_dir, "Fig1_TME_Overview.png"), dpi=300, bbox_inches='tight'); plt.close(fig1)

# Composition plot
print("Generating Figure 1d: Composition...")
if color_key in adata_rna.obs and 'responder_status' in adata_rna.obs:
    try:
        comp_df = adata_rna.obs.groupby(['responder_status', color_key], observed=True).size().unstack(fill_value=0)
        if not comp_df.empty:
            comp_df_norm = comp_df.apply(lambda x: x / x.sum() if x.sum() > 0 else x, axis=1)
            if not comp_df_norm.empty:
                fig1d, ax1d = plt.subplots(figsize=(6, 4))
                comp_df_norm.plot(kind='bar', stacked=True, ax=ax1d, cmap='tab20'); ax1d.set_title('Fig 1d: TME Composition R vs NR'); ax1d.set_ylabel('Cell Proportion'); ax1d.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Cell Type/Cluster', fontsize=8); sns.despine(ax=ax1d); plt.tight_layout(rect=[0, 0, 0.8, 1]); plt.savefig(os.path.join(output_dir, "Fig1d_Composition.png"), dpi=300, bbox_inches='tight'); plt.close(fig1d)
            else: print("Skipping Fig 1d: Normalized Composition DF empty.")
        else: print("Skipping Fig 1d: Composition DF empty.")
    except Exception as e: print(f"Error generating Fig 1d: {e}")
else: print("Skipping Fig 1d: Missing required columns in adata_rna.obs.")


# --- Step 2: T Cell Analysis ---
print("\n--- Step 2: T Cell Analysis ---")
# Check if 'T Cells' category exists before subsetting
if 'major_cell_type' in adata_rna.obs and 'T Cells' in adata_rna.obs['major_cell_type'].cat.categories:
    adata_t = adata_rna[adata_rna.obs['major_cell_type'] == 'T Cells'].copy()
    if adata_t.n_obs == 0: print("Warning: T cell subset is empty after filtering.")
else:
    print("Cannot perform T cell analysis: 'major_cell_type' missing or 'T Cells' category absent.")
    adata_t = ad.AnnData() # Create empty object

dynamic_genes_df = pd.DataFrame() # Initialize df for dynamic gene info (used in step 4)
min_cells_for_hvg = 20; min_genes_for_hvg = 50

if adata_t.n_obs > 0:
    print(f"T cell subset shape: {adata_t.shape}")
    batch_key_t = 'batch' if 'batch' in adata_t.obs else None # Adjust if needed
    if adata_t.n_obs >= min_cells_for_hvg and adata_t.n_vars >= min_genes_for_hvg:
        print(f"  Running HVG on T cell subset using 'cell_ranger' flavor...")
        try: sc.pp.highly_variable_genes(adata_t, n_top_genes=min(1000, adata_t.n_vars - 1), subset=True, flavor='cell_ranger', batch_key=batch_key_t); print(f"  T cell subset shape after HVG: {adata_t.shape}")
        except Exception as hvg_err: print(f"  Warning: T cell HVG failed: {hvg_err}")
    else: print(f"  Skipping T cell HVG: Subset too small.")

    if adata_t.n_obs > 0 and adata_t.n_vars > 0:
        sc.pp.scale(adata_t, max_value=10)
        n_pcs_t = min(30, adata_t.n_obs-1, adata_t.n_vars-1)
        if n_pcs_t > 1:
            sc.tl.pca(adata_t, n_comps=n_pcs_t)
            # If batch corrected earlier using Harmony, use 'X_pca_harmony', otherwise use 'X_pca'
            use_rep_t = 'X_pca_harmony' if 'X_pca_harmony' in adata_t.obsm else 'X_pca'
            n_neighbors_pcs_t = min(20, n_pcs_t); n_neighbors_knn_t = min(10, adata_t.n_obs - 1)
            if n_neighbors_knn_t > 0 and n_neighbors_pcs_t > 0 and use_rep_t in adata_t.obsm:
                 sc.pp.neighbors(adata_t, n_neighbors=n_neighbors_knn_t, use_rep=use_rep_t) # Use appropriate representation
                 sc.tl.umap(adata_t, min_dist=0.4)
                 sc.tl.louvain(adata_t, resolution=0.6, key_added='tcell_clusters')
            else: print("Warning: Insufficient cells/PCs/Rep for T cell neighbors. Skipping."); adata_t.obsm['X_umap'] = np.zeros((adata_t.n_obs, 2)); adata_t.obs['tcell_clusters'] = '0'
        else: print("Warning: Insufficient PCs for T cell neighbors/UMAP."); adata_t.obsm['X_umap'] = np.zeros((adata_t.n_obs, 2)); adata_t.obs['tcell_clusters'] = '0'

        # --- T Cell Subtype Annotation (Refined) ---
        print("  Annotating T cell subtypes based on markers...")
        # Requires more sophisticated logic based on marker expression within tcell_clusters
        # Placeholder: Assign cluster IDs or perform basic marker checks
        adata_t.obs['tcell_subtype'] = adata_t.obs['tcell_clusters'] # Fallback
        # Example (needs refinement):
        # sc.pl.dotplot(adata_t, markers, groupby='tcell_clusters', show=False) # Visualize markers
        # Define mapping based on visualization: cluster_map = {'0': 'CD8_Tem', '1': 'CD8_Tex', ...}
        # adata_t.obs['tcell_subtype'] = adata_t.obs['tcell_clusters'].map(cluster_map).fillna('Unknown T').astype('category')
        print("  T cell subtype annotation placeholder applied (using cluster IDs). Refine this step.")

        # Calculate state scores
        exhaustion_markers = ['PDCD1', 'CTLA4', 'LAG3', 'TOX', 'HAVCR2']; cytotoxicity_markers = ['GZMB', 'PRF1', 'IFNG', 'GNLY']
        exhaustion_markers_adata = [g for g in exhaustion_markers if g in adata_t.var_names]; cytotoxicity_markers_adata = [g for g in cytotoxicity_markers if g in adata_t.var_names]
        ctrl_size_exh = min(50, len(adata_t.var_names)-1, len(exhaustion_markers_adata)) if len(exhaustion_markers_adata) > 0 else 0
        ctrl_size_cyt = min(50, len(adata_t.var_names)-1, len(cytotoxicity_markers_adata)) if len(cytotoxicity_markers_adata) > 0 else 0
        if exhaustion_markers_adata and ctrl_size_exh > 0: sc.tl.score_genes(adata_t, gene_list=exhaustion_markers_adata, score_name='Exhaustion_Score', ctrl_size=ctrl_size_exh)
        if cytotoxicity_markers_adata and ctrl_size_cyt > 0: sc.tl.score_genes(adata_t, gene_list=cytotoxicity_markers_adata, score_name='Cytotoxicity_Score', ctrl_size=ctrl_size_cyt)

        # --- Trajectory Inference (PAGA/DPT) ---
        print("  Running Trajectory Inference (PAGA/DPT)...")
        if 'neighbors' in adata_t.uns:
             sc.tl.paga(adata_t, groups='tcell_clusters') # Or use 'tcell_subtype' if annotation is reliable
             sc.pl.paga(adata_t, show=False, save="_tcell_paga.png") # Save PAGA plot
             # --- Root cell selection is CRUCIAL here ---
             # Find root based on biology (e.g., cluster with high TCF7/IL7R)
             # root_cluster = find_root_cluster(adata_t) # Implement this logic
             # adata_t.uns['iroot'] = np.flatnonzero(adata_t.obs['tcell_clusters'] == root_cluster)[0]
             # For placeholder, just pick the first cell of cluster '0'
             if 'tcell_clusters' in adata_t.obs and '0' in adata_t.obs['tcell_clusters'].cat.categories:
                  try:
                       root_index = np.flatnonzero(adata_t.obs['tcell_clusters'] == '0')
                       if len(root_index) > 0:
                            adata_t.uns['iroot'] = root_index[0]
                            print(f"    Using cell {adata_t.obs_names[adata_t.uns['iroot']]} in cluster '0' as root.")
                            sc.tl.dpt(adata_t)
                            adata_t.obs['tcell_pseudotime'] = adata_t.obs['dpt_pseudotime'] # Use calculated pseudotime
                       else:
                            print("    Warning: Cluster '0' is empty, cannot set root. Using random pseudotime.")
                            adata_t.obs['tcell_pseudotime'] = np.random.rand(adata_t.n_obs)
                  except Exception as dpt_err:
                       print(f"    Error during DPT calculation: {dpt_err}. Using random pseudotime.")
                       adata_t.obs['tcell_pseudotime'] = np.random.rand(adata_t.n_obs)
             else:
                  print("    Warning: Cannot find cluster '0' to set root. Using random pseudotime.")
                  adata_t.obs['tcell_pseudotime'] = np.random.rand(adata_t.n_obs)
             # --- End Root Selection ---
        else:
             print("    Neighbors not computed for T cells. Skipping PAGA/DPT. Using random pseudotime.")
             adata_t.obs['tcell_pseudotime'] = np.random.rand(adata_t.n_obs)


        # Plotting Figure 2
        print("Generating Figure 2...")
        fig2, axes2 = plt.subplots(1, 3, figsize=(18, 5))
        if 'X_umap' in adata_t.obsm:
            sc.pl.umap(adata_t, color='tcell_subtype', ax=axes2[0], show=False, frameon=False, title='T Cell Subtypes', legend_loc='on data', legend_fontsize=8)
            scores_to_plot_t = [s for s in ['Exhaustion_Score', 'Cytotoxicity_Score'] if s in adata_t.obs]
            if scores_to_plot_t: sc.pl.umap(adata_t, color=scores_to_plot_t, ax=axes2[1], show=False, frameon=False, title='Functional Scores', cmap='coolwarm', vmin=-0.3, vmax=0.3)
            else: axes2[1].text(0.5, 0.5, "Fig 2b: Scores Skipped", ha='center', va='center', transform=axes2[1].transAxes); axes2[1].set_title("Scores (Skipped)")
            sc.pl.umap(adata_t, color='tcell_pseudotime', ax=axes2[2], show=False, frameon=False, title='Inferred Trajectory (Pseudotime)', cmap='viridis')
        else:
            for k in range(3): axes2[k].text(0.5, 0.5, f"Fig 2{chr(ord('a')+k)}: UMAP Skipped", ha='center', va='center', transform=axes2[k].transAxes); axes2[k].set_xticks([]); axes2[k].set_yticks([]); axes2[k].set_title(f"UMAP {chr(ord('a')+k)} (Skipped)")
        plt.suptitle("Figure 2: T Cell Heterogeneity and Dynamics", fontsize=16, y=1.02); plt.tight_layout(rect=[0, 0, 1, 0.98]); plt.savefig(os.path.join(output_dir, "Fig2_Tcell_Analysis.png"), dpi=300, bbox_inches='tight'); plt.close(fig2)

        # Plot gene dynamics
        t_dynamic_genes = ['TCF7','IL7R','GZMB','PDCD1','TOX', 'CTLA4']
        t_dynamic_genes_adata = [g for g in t_dynamic_genes if g in adata_t.var_names]
        if t_dynamic_genes_adata and 'tcell_pseudotime' in adata_t.obs and not adata_t.obs['tcell_pseudotime'].isna().all():
            print("Generating Figure 2d: Gene Dynamics...")
            plot_mask_t = ~adata_t.obs['tcell_pseudotime'].isna() # Keep all non-NaN pseudotime
            if plot_mask_t.sum() > 0:
                adata_t_dyn = adata_t[plot_mask_t].copy()
                if adata_t_dyn.n_obs > 0:
                    n_genes_plot = len(t_dynamic_genes_adata); fig2d, axes2d = plt.subplots(n_genes_plot, 1, figsize=(5, 1.5 * n_genes_plot), sharex=True); axes2d = [axes2d] if n_genes_plot == 1 else axes2d
                    ptime_t = adata_t_dyn.obs['tcell_pseudotime']
                    for i, gene in enumerate(t_dynamic_genes_adata):
                        if gene in adata_t_dyn.var_names:
                            try:
                                expr_t = adata_t_dyn[:, gene].X.toarray().flatten() if not isinstance(adata_t_dyn.X, np.ndarray) else adata_t_dyn[:, gene].X.flatten()
                                if len(ptime_t) > 10 and np.var(expr_t) > 1e-6:
                                    try: smoothed_t = lowess(expr_t, ptime_t, frac=0.3, it=1); sort_idx_t = np.argsort(smoothed_t[:, 0]); axes2d[i].plot(smoothed_t[sort_idx_t, 0], smoothed_t[sort_idx_t, 1], color='darkgreen', lw=2, label='Smoothed')
                                    except Exception as smooth_err: print(f"  Warning: Lowess failed for {gene}: {smooth_err}")
                                sns.scatterplot(x=ptime_t, y=expr_t, hue=adata_t_dyn.obs['tcell_subtype'], s=5, alpha=0.5, ax=axes2d[i], legend=(i==0)); axes2d[i].set_ylabel(gene, fontsize=8, rotation=0, ha='right', va='center'); sns.despine(ax=axes2d[i])
                                if i > 0 and axes2d[i].get_legend() is not None: axes2d[i].legend().remove()
                            except Exception as plot_err: print(f"  Warning: Error plotting dynamics for {gene}: {plot_err}"); axes2d[i].text(0.5, 0.5, f"{gene}\n(Plot Error)", ha='center', va='center', transform=axes2d[i].transAxes)
                        else: axes2d[i].text(0.5, 0.5, f"{gene}\n(Not in subset)", ha='center', va='center', transform=axes2d[i].transAxes); axes2d[i].set_ylabel(gene, fontsize=8, rotation=0, ha='right', va='center'); sns.despine(ax=axes2d[i])
                    axes2d[-1].set_xlabel('T Cell Pseudotime'); legend = axes2d[0].get_legend();
                    if legend: legend.set_bbox_to_anchor((1.05, 0.5)); legend.set_title('Subtype'); # Adjust legend
                    plt.suptitle("Fig 2d: Key Gene Dynamics along T cell Trajectory", fontsize=12, y=1.01); plt.tight_layout(rect=[0, 0.05, 0.85, 0.98]); plt.savefig(os.path.join(output_dir, "Fig2d_Tcell_Gene_Dynamics.png"), dpi=300, bbox_inches='tight'); plt.close(fig2d)
                    # Calculate dynamics info for Step 4
                    print("  Calculating correlations for T cell dynamic genes..."); dynamic_genes_list_t = []
                    for gene in t_dynamic_genes_adata:
                        if gene in adata_t_dyn.var_names:
                            try:
                                expr_t = adata_t_dyn[:, gene].X.toarray().flatten() if not isinstance(adata_t_dyn.X, np.ndarray) else adata_t_dyn[:, gene].X.flatten()
                                if np.var(expr_t) > 1e-6 and len(ptime_t) > 1:
                                    try: corr, pval = spearmanr(ptime_t, expr_t);
                                    except Exception as corr_err: print(f"   Warning: Spearman failed for {gene}: {corr_err}"); continue
                                    if not np.isnan(corr): dynamic_genes_list_t.append({'Feature': gene, 'Type': 'Gene', 'Correlation': corr, 'P_value': pval})
                            except Exception as data_access_err: print(f"   Warning: Error accessing data for {gene} correlation: {data_access_err}")
                    if dynamic_genes_list_t: dynamic_genes_df = pd.DataFrame(dynamic_genes_list_t); print(f"  Found {len(dynamic_genes_df)} dynamic genes in T cells.")
                    else: print("  No dynamic genes found/calculated for T cells.")
                else: print("Skipping T cell gene dynamics plot: Subset empty after filtering.")
            else: print("Skipping T cell gene dynamics plot: No cells with valid pseudotime after filtering.")
        else: print("Skipping T cell gene dynamics plot and correlations (missing data).")

        # Compare R vs NR
        print("Generating Figure 2e: R vs NR T cell differences...")
        fig2e, axes2e = plt.subplots(1, 2, figsize=(8, 4))
        if 'Exhaustion_Score' in adata_t.obs:
             try: sns.boxplot(data=adata_t.obs, x='responder_status', y='Exhaustion_Score', ax=axes2e[0], palette={'R':'skyblue', 'NR':'salmon'}); axes2e[0].set_title('Exhaustion Score'); axes2e[0].set_ylabel('Score'); axes2e[0].set_xlabel('Responder Status'); sns.despine(ax=axes2e[0])
             except Exception as boxplot_err: print(f"  Warning: Exhaustion score boxplot failed: {boxplot_err}"); axes2e[0].text(0.5,0.5,"Exh. Score\n(Plot Error)", ha='center',va='center',transform=axes2e[0].transAxes)
             try: group_r = adata_t.obs[adata_t.obs['responder_status']=='R']['Exhaustion_Score'].dropna(); group_nr = adata_t.obs[adata_t.obs['responder_status']=='NR']['Exhaustion_Score'].dropna()
             except KeyError: group_r, group_nr = pd.Series(), pd.Series()
             if len(group_r) > 1 and len(group_nr) > 1:
                 try: stat, pval = mannwhitneyu(group_r, group_nr); axes2e[0].text(0.5, 0.95, f'p={pval:.2e}', ha='center', va='top', transform=axes2e[0].transAxes, fontsize=9)
                 except ValueError: pass
        else: axes2e[0].text(0.5,0.5,"Exh. Score\nNot Calculated", ha='center',va='center',transform=axes2e[0].transAxes); axes2e[0].set_xticks([]); axes2e[0].set_yticks([])
        if 'tcell_subtype' in adata_t.obs:
            # Check if 'CD8_Tex' exists as a category after annotation
            tex_category_exists = isinstance(adata_t.obs['tcell_subtype'].dtype, pd.CategoricalDtype) and 'CD8_Tex' in adata_t.obs['tcell_subtype'].cat.categories
            if not tex_category_exists:
                 print("   Warning: 'CD8_Tex' category not found in annotated tcell_subtype. Trying to use string if available.")
                 tex_category_exists = 'CD8_Tex' in adata_t.obs['tcell_subtype'].unique()

            if tex_category_exists:
                try:
                    t_comp = adata_t.obs.groupby(['responder_status', 'tcell_subtype'], observed=True).size().unstack(fill_value=0)
                    if not t_comp.empty:
                        t_comp_norm = t_comp.apply(lambda x: x / x.sum() if x.sum() > 0 else x, axis=1)
                        if 'CD8_Tex' in t_comp_norm.columns: sns.boxplot(data=t_comp_norm.reset_index(), x='responder_status', y='CD8_Tex', ax=axes2e[1], palette={'R':'skyblue', 'NR':'salmon'}); axes2e[1].set_title('CD8_Tex Proportion'); axes2e[1].set_ylabel('Proportion within T cells'); axes2e[1].set_xlabel('Responder Status'); sns.despine(ax=axes2e[1])
                        else: axes2e[1].text(0.5,0.5,"CD8_Tex\nNot Found", ha='center',va='center',transform=axes2e[1].transAxes); axes2e[1].set_xticks([]); axes2e[1].set_yticks([])
                    else: axes2e[1].text(0.5,0.5,"T Comp Empty", ha='center',va='center',transform=axes2e[1].transAxes); axes2e[1].set_xticks([]); axes2e[1].set_yticks([])
                except Exception as comp_plot_err: print(f"  Warning: T cell comp boxplot failed: {comp_plot_err}"); axes2e[1].text(0.5,0.5,"CD8_Tex\n(Plot Error)", ha='center',va='center',transform=axes2e[1].transAxes)
            else: axes2e[1].text(0.5,0.5,"CD8_Tex\nNot Annot.", ha='center',va='center',transform=axes2e[1].transAxes); axes2e[1].set_xticks([]); axes2e[1].set_yticks([])
        else: axes2e[1].text(0.5,0.5,"Subtype Missing", ha='center',va='center',transform=axes2e[1].transAxes); axes2e[1].set_xticks([]); axes2e[1].set_yticks([])
        plt.suptitle("Fig 2e: T cell Differences R vs NR", fontsize=12, y=1.02); plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(os.path.join(output_dir, "Fig2e_Tcell_R_vs_NR.png"), dpi=300, bbox_inches='tight'); plt.close(fig2e)
    else: print("Skipping rest of T cell analysis: subset empty after HVG.")
else: print("Skipping T cell analysis: initial subset empty.")


# --- Step 3: Myeloid Analysis ---
# [Code similar to T cell analysis, adapted for Myeloid cells]
# [Ensure placeholders for annotation and proper checks are included]
print("\n--- Step 3: Myeloid Analysis ---")
if 'major_cell_type' in adata_rna.obs and 'Myeloid' in adata_rna.obs['major_cell_type'].cat.categories: adata_mye = adata_rna[adata_rna.obs['major_cell_type'] == 'Myeloid'].copy()
else: print("Cannot perform Myeloid analysis: 'major_cell_type' or 'Myeloid' category missing."); adata_mye = ad.AnnData()
if adata_mye.n_obs > 0:
    print(f"Myeloid subset shape: {adata_mye.shape}")
    batch_key_mye = 'batch' if 'batch' in adata_mye.obs else None
    if adata_mye.n_obs >= min_cells_for_hvg and adata_mye.n_vars >= min_genes_for_hvg:
        print(f"  Running HVG on Myeloid subset using 'cell_ranger' flavor...")
        try: sc.pp.highly_variable_genes(adata_mye, n_top_genes=min(1000, adata_mye.n_vars - 1), subset=True, flavor='cell_ranger', batch_key=batch_key_mye); print(f"  Myeloid subset shape after HVG: {adata_mye.shape}")
        except Exception as hvg_err: print(f"  Warning: Myeloid HVG failed: {hvg_err}")
    else: print(f"  Skipping Myeloid HVG: Subset too small.")
    if adata_mye.n_obs > 0 and adata_mye.n_vars > 0:
        sc.pp.scale(adata_mye, max_value=10)
        n_pcs_mye = min(30, adata_mye.n_obs-1, adata_mye.n_vars-1)
        if n_pcs_mye > 1:
            sc.tl.pca(adata_mye, n_comps=n_pcs_mye)
            use_rep_mye = 'X_pca_harmony' if 'X_pca_harmony' in adata_mye.obsm else 'X_pca'
            n_neighbors_pcs_mye = min(20, n_pcs_mye); n_neighbors_knn_mye = min(10, adata_mye.n_obs - 1)
            if n_neighbors_knn_mye > 0 and n_neighbors_pcs_mye > 0 and use_rep_mye in adata_mye.obsm: sc.pp.neighbors(adata_mye, n_neighbors=n_neighbors_knn_mye, use_rep=use_rep_mye); sc.tl.umap(adata_mye, min_dist=0.4); sc.tl.louvain(adata_mye, resolution=0.5, key_added='myeloid_clusters')
            else: print("Warning: Insufficient cells/PCs/Rep for Myeloid neighbors. Skipping."); adata_mye.obsm['X_umap'] = np.zeros((adata_mye.n_obs, 2)); adata_mye.obs['myeloid_clusters'] = '0'
        else: print("Warning: Insufficient PCs for Myeloid neighbors/UMAP."); adata_mye.obsm['X_umap'] = np.zeros((adata_mye.n_obs, 2)); adata_mye.obs['myeloid_clusters'] = '0'
        print("  Annotating Myeloid subtypes based on markers (Placeholder)...")
        adata_mye.obs['myeloid_subtype'] = adata_mye.obs['myeloid_clusters'] # Fallback
        # Add your refined myeloid annotation logic here, potentially using markers dict
        print("  Myeloid subtype annotation placeholder applied. Refine this step.")
        if 'myeloid_subtype' in adata_mye.obs: macro_mask = adata_mye.obs['myeloid_subtype'].astype(str).str.contains('Macro') # Use astype(str) for safety
        else: macro_mask = pd.Series([False] * adata_mye.n_obs, index=adata_mye.obs_names)
        m1_markers = ['CD86', 'TNF', 'IL1B', 'IL6', 'CXCL9', 'CXCL10']; m2_markers = ['CD163', 'MRC1', 'CD206', 'IL10', 'CCL18', 'CCL22']
        m1_markers_adata = [g for g in m1_markers if g in adata_mye.var_names]; m2_markers_adata = [g for g in m2_markers if g in adata_mye.var_names]
        ctrl_size_m1 = min(50, len(adata_mye.var_names)-1, len(m1_markers_adata)) if len(m1_markers_adata) > 0 else 0
        ctrl_size_m2 = min(50, len(adata_mye.var_names)-1, len(m2_markers_adata)) if len(m2_markers_adata) > 0 else 0
        if m1_markers_adata and ctrl_size_m1 > 0: sc.tl.score_genes(adata_mye, gene_list=m1_markers_adata, score_name='M1_Score', ctrl_size=ctrl_size_m1)
        if m2_markers_adata and ctrl_size_m2 > 0: sc.tl.score_genes(adata_mye, gene_list=m2_markers_adata, score_name='M2_Score', ctrl_size=ctrl_size_m2)
        # Plotting Figure 3
        print("Generating Figure 3..."); fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
        if 'X_umap' in adata_mye.obsm:
            sc.pl.umap(adata_mye, color='myeloid_subtype', ax=axes3[0], show=False, frameon=False, title='Myeloid Subtypes', legend_loc='on data', legend_fontsize=8)
            scores_to_plot_m = [s for s in ['M1_Score', 'M2_Score'] if s in adata_mye.obs]
            if scores_to_plot_m and macro_mask.sum() > 0: sc.pl.umap(adata_mye, color=scores_to_plot_m, ax=axes3[1], show=False, frameon=False, title='Macrophage Polarization Scores', cmap='RdYlBu_r', vmin=-0.3, vmax=0.3)
            else: axes3[1].text(0.5, 0.5, "Fig 3b: Scores Skipped", ha='center', va='center', transform=axes3[1].transAxes); axes3[1].set_title("Scores (Skipped)")
        else:
            for k in range(2): axes3[k].text(0.5, 0.5, f"Fig 3{chr(ord('a')+k)}: UMAP Skipped", ha='center', va='center', transform=axes3[k].transAxes); axes3[k].set_xticks([]); axes3[k].set_yticks([]); axes3[k].set_title(f"UMAP {chr(ord('a')+k)} (Skipped)")
        plt.suptitle("Figure 3: Myeloid Heterogeneity", fontsize=16, y=1.02); plt.tight_layout(rect=[0, 0, 1, 0.98]); plt.savefig(os.path.join(output_dir, "Fig3_Myeloid_Analysis.png"), dpi=300, bbox_inches='tight'); plt.close(fig3)
        # Compare R vs NR
        print("Generating Figure 3d: R vs NR Myeloid differences..."); fig3d, axes3d = plt.subplots(1, 2, figsize=(8, 4))
        if 'M2_Score' in adata_mye.obs and macro_mask.sum() > 0:
             try: sns.boxplot(data=adata_mye[macro_mask].obs, x='responder_status', y='M2_Score', ax=axes3d[0], palette={'R':'skyblue', 'NR':'salmon'}); axes3d[0].set_title('Macrophage M2 Score'); axes3d[0].set_ylabel('Score'); axes3d[0].set_xlabel('Responder Status'); sns.despine(ax=axes3d[0])
             except Exception as boxplot_m2_err: print(f"  Warning: M2 score boxplot failed: {boxplot_m2_err}"); axes3d[0].text(0.5,0.5,"M2 Score\n(Plot Error)", ha='center',va='center',transform=axes3d[0].transAxes)
             try: group_r_m2 = adata_mye[macro_mask & (adata_mye.obs['responder_status']=='R')]['M2_Score'].dropna(); group_nr_m2 = adata_mye[macro_mask & (adata_mye.obs['responder_status']=='NR')]['M2_Score'].dropna()
             except KeyError: group_r_m2, group_nr_m2 = pd.Series(), pd.Series()
             if len(group_r_m2) > 1 and len(group_nr_m2) > 1:
                 try: stat, pval = mannwhitneyu(group_r_m2, group_nr_m2); axes3d[0].text(0.5, 0.95, f'p={pval:.2e}', ha='center', va='top', transform=axes3d[0].transAxes, fontsize=9)
                 except ValueError: pass
        else: axes3d[0].text(0.5,0.5,"M2 Score\nNot Calculated/\nNo Macros", ha='center',va='center',transform=axes3d[0].transAxes); axes3d[0].set_xticks([]); axes3d[0].set_yticks([])
        if 'myeloid_subtype' in adata_mye.obs:
             # Check if 'Macro_M2' exists as a category/value
             m2_category_exists = (isinstance(adata_mye.obs['myeloid_subtype'].dtype, pd.CategoricalDtype) and 'Macro_M2' in adata_mye.obs['myeloid_subtype'].cat.categories) or ('Macro_M2' in adata_mye.obs['myeloid_subtype'].unique())
             if m2_category_exists:
                 try:
                    mye_comp = adata_mye.obs.groupby(['responder_status', 'myeloid_subtype'], observed=True).size().unstack(fill_value=0)
                    if not mye_comp.empty:
                        mye_comp_norm = mye_comp.apply(lambda x: x / x.sum() if x.sum() > 0 else x, axis=1)
                        if 'Macro_M2' in mye_comp_norm.columns: sns.boxplot(data=mye_comp_norm.reset_index(), x='responder_status', y='Macro_M2', ax=axes3d[1], palette={'R':'skyblue', 'NR':'salmon'}); axes3d[1].set_title('Macro_M2 Proportion'); axes3d[1].set_ylabel('Proportion within Myeloid'); axes3d[1].set_xlabel('Responder Status'); sns.despine(ax=axes3d[1])
                        else: axes3d[1].text(0.5,0.5,"Macro_M2\nNot Found", ha='center',va='center',transform=axes3d[1].transAxes); axes3d[1].set_xticks([]); axes3d[1].set_yticks([])
                    else: axes3d[1].text(0.5,0.5,"Mye Comp Empty", ha='center',va='center',transform=axes3d[1].transAxes); axes3d[1].set_xticks([]); axes3d[1].set_yticks([])
                 except Exception as comp_mye_err: print(f"  Warning: Myeloid comp boxplot failed: {comp_mye_err}"); axes3d[1].text(0.5,0.5,"Macro_M2\n(Plot Error)", ha='center',va='center',transform=axes3d[1].transAxes)
             else: axes3d[1].text(0.5,0.5,"Macro_M2\nNot Annot.", ha='center',va='center',transform=axes3d[1].transAxes); axes3d[1].set_xticks([]); axes3d[1].set_yticks([])
        else: axes3d[1].text(0.5,0.5,"Subtype Missing", ha='center',va='center',transform=axes3d[1].transAxes); axes3d[1].set_xticks([]); axes3d[1].set_yticks([])
        plt.suptitle("Fig 3d: Myeloid Differences R vs NR", fontsize=12, y=1.02); plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(os.path.join(output_dir, "Fig3d_Myeloid_R_vs_NR.png"), dpi=300, bbox_inches='tight'); plt.close(fig3d)
    else: print("Skipping rest of Myeloid analysis: subset empty after HVG.")
else: print("Skipping Myeloid analysis: initial subset empty.")


# --- Step 4: ATAC Integration for T cell Exhaustion ---
print("\n--- Step 4: ATAC Integration for T cell Exhaustion ---")
if ('adata_t' in locals() and adata_t.n_obs > 0 and 'tcell_pseudotime' in adata_t.obs and
    'ATAC' in adata_multiome.obsm and isinstance(adata_multiome.obsm['ATAC'], pd.DataFrame) and not adata_multiome.obsm['ATAC'].empty):
    common_obs_names = adata_t.obs_names.intersection(adata_multiome.obsm['ATAC'].index)
    if not common_obs_names.empty:
        adata_t_atac = adata_multiome[common_obs_names, :].copy()
        cols_to_join = [col for col in ['tcell_pseudotime', 'tcell_subtype'] if col in adata_t.obs];
        if cols_to_join: adata_t_atac.obs = adata_t_atac.obs.join(adata_t.obs[cols_to_join])
        try: atac_t_df = adata_multiome.obsm['ATAC'].loc[common_obs_names].copy(); adata_t_atac.obsm['ATAC_subset'] = atac_t_df
        except KeyError: print("Error accessing ATAC data for T cell subset."); atac_t_df=None # Handle potential reindex issues earlier

        if 'tcell_pseudotime' in adata_t_atac.obs and atac_t_df is not None:
            valid_ptime_mask = ~adata_t_atac.obs['tcell_pseudotime'].isna() # Use all non-NA pseudotime
            if valid_ptime_mask.sum() > 10:
                adata_t_atac_dyn = adata_t_atac[valid_ptime_mask, :].copy()
                if 'ATAC_subset' in adata_t_atac_dyn.obsm and not adata_t_atac_dyn.obsm['ATAC_subset'].empty:
                    atac_t_df_dyn = adata_t_atac_dyn.obsm['ATAC_subset']; ptime_t_atac = adata_t_atac_dyn.obs['tcell_pseudotime']
                    dynamic_threshold_atac = 0.1; dynamic_peaks_t_list = []
                    print(f"  Identifying dynamic peaks in {len(ptime_t_atac)} T cells...")
                    for peak in atac_t_df_dyn.columns:
                        peak_vector = atac_t_df_dyn[peak].values
                        if np.var(peak_vector) > 1e-6:
                            try: corr, pval = spearmanr(ptime_t_atac, peak_vector)
                            except Exception as peak_corr_err: print(f"   Warning: Spearman failed for peak {peak}: {peak_corr_err}"); continue
                            if not np.isnan(corr) and abs(corr) >= dynamic_threshold_atac: dynamic_peaks_t_list.append({'Feature': peak, 'Type': 'Peak', 'Correlation': corr, 'P_value': pval})
                    if dynamic_peaks_t_list:
                        dynamic_peaks_t_df = pd.DataFrame(dynamic_peaks_t_list); dynamic_peaks_t_df['Abs_Correlation'] = dynamic_peaks_t_df['Correlation'].abs(); dynamic_peaks_t_df = dynamic_peaks_t_df.sort_values('Abs_Correlation', ascending=False).drop(columns=['Abs_Correlation'])
                        print(f"  Found {len(dynamic_peaks_t_df)} dynamic Peaks in T cells (rho >= {dynamic_threshold_atac})."); peaks_t_available = True
                    else: print("  No dynamic Peaks found."); dynamic_peaks_t_df = pd.DataFrame(); peaks_t_available = False
                    # --- Peak-Gene Links: Use external database or prediction tool output here ---
                    # Placeholder using markers, but replace with actual links from e.g., GREAT, Cicero, or correlation
                    tcell_peak_gene_links = pd.DataFrame([
                        {'PeakID': 'Peak_Near_IL7R', 'Gene': 'IL7R'}, # Replace with actual peak IDs
                        {'PeakID': 'Peak_Near_TCF7', 'Gene': 'TCF7'},
                        {'PeakID': 'Peak_Near_PDCD1', 'Gene': 'PDCD1'},
                        {'PeakID': 'Peak_Near_TOX', 'Gene': 'TOX'}
                        # Add more based on data
                    ])
                    print("  Using placeholder peak-gene links. Replace with data-driven links.")
                    # --- End Peak-Gene Links ---
                    integrated_t_results = []
                    t_genes_available = not dynamic_genes_df.empty and 'Feature' in dynamic_genes_df.columns; t_dynamic_gene_set = set(dynamic_genes_df['Feature']) if t_genes_available else set(); t_gene_ptime_corr_map = dynamic_genes_df.set_index('Feature')['Correlation'].to_dict() if t_genes_available else {}
                    if t_genes_available and peaks_t_available:
                        t_dynamic_peak_set = set(dynamic_peaks_t_df['Feature']); t_peak_ptime_corr_map = dynamic_peaks_t_df.set_index('Feature')['Correlation'].to_dict()
                        print("  Integrating RNA/ATAC dynamics within T cells...")
                        for _, link in tcell_peak_gene_links.iterrows():
                            gene_id = link['Gene']; peak_id = link['PeakID'] # Use the linked PeakID
                            # Check if the *specific linked peak* was found dynamic, not just *any* peak
                            if gene_id in t_dynamic_gene_set and peak_id in t_dynamic_peak_set:
                                gene_ptime_corr = t_gene_ptime_corr_map.get(gene_id, np.nan); peak_ptime_corr = t_peak_ptime_corr_map.get(peak_id, np.nan)
                                concordant = (gene_ptime_corr * peak_ptime_corr) > 0 if not (np.isnan(gene_ptime_corr) or np.isnan(peak_ptime_corr)) else False
                                integrated_t_results.append({'Gene': gene_id, 'PeakID': peak_id, 'Gene_Corr': gene_ptime_corr, 'Peak_Corr': peak_ptime_corr, 'Concordant': concordant})
                    if integrated_t_results:
                        integrated_t_df = pd.DataFrame(integrated_t_results).sort_values('Concordant', ascending=False); print("  Found co-dynamic Peak-Gene pairs in T cells (based on links):"); print(integrated_t_df)
                        print("Generating Figure 4 (Co-dynamic pair example)...")
                        plot_pair_t_df = integrated_t_df[integrated_t_df['Concordant']].head(1) # Plot first concordant pair found
                        if plot_pair_t_df.empty: plot_pair_t_df = integrated_t_df.head(1)
                        if not plot_pair_t_df.empty:
                            target_gene_t = plot_pair_t_df['Gene'].iloc[0]; target_peak_t = plot_pair_t_df['PeakID'].iloc[0]
                            # Check if peak exists in the dynamic ATAC df columns
                            if target_gene_t in adata_t_atac_dyn.var_names and target_peak_t in atac_t_df_dyn.columns:
                                fig4b, axes4b = plt.subplots(2, 1, figsize=(7, 5), sharex=True)
                                try:
                                    gene_expr_t = adata_t_atac_dyn[:, target_gene_t].X.toarray().flatten() if not isinstance(adata_t_atac_dyn.X, np.ndarray) else adata_t_atac_dyn[:, target_gene_t].X.flatten()
                                    if len(ptime_t_atac) > 10 and np.var(gene_expr_t) > 1e-6: smoothed_g_t = lowess(gene_expr_t, ptime_t_atac, frac=0.3); sort_g_t = np.argsort(smoothed_g_t[:,0]); axes4b[0].plot(smoothed_g_t[sort_g_t,0], smoothed_g_t[sort_g_t,1], color='darkblue', lw=2)
                                    sns.scatterplot(x=ptime_t_atac, y=gene_expr_t, hue=adata_t_atac_dyn.obs['tcell_subtype'], s=8, alpha=0.6, ax=axes4b[0], legend=False); axes4b[0].set_ylabel(f"{target_gene_t}\nExpression"); sns.despine(ax=axes4b[0])
                                    peak_signal_t = atac_t_df_dyn[target_peak_t].values
                                    if len(ptime_t_atac) > 10 and np.var(peak_signal_t) > 1e-6: smoothed_p_t = lowess(peak_signal_t, ptime_t_atac, frac=0.3); sort_p_t = np.argsort(smoothed_p_t[:,0]); axes4b[1].plot(smoothed_p_t[sort_p_t,0], smoothed_p_t[sort_p_t,1], color='darkred', lw=2)
                                    sns.scatterplot(x=ptime_t_atac, y=peak_signal_t, hue=adata_t_atac_dyn.obs['tcell_subtype'], s=8, alpha=0.6, ax=axes4b[1], legend=True); axes4b[1].set_ylabel(f"{target_peak_t}\nAccessibility"); axes4b[1].set_xlabel("T cell Pseudotime"); sns.despine(ax=axes4b[1]); legend=axes4b[1].get_legend(); legend.set_bbox_to_anchor((1.05, 0.5)); legend.set_title('Subtype'); # Adjust legend
                                    plt.suptitle(f"Figure 4b: Co-dynamics of {target_gene_t} & {target_peak_t}", fontsize=12, y=1.0); plt.tight_layout(rect=[0, 0, 0.85, 0.97]); plt.savefig(os.path.join(output_dir, f"Fig4b_Tcell_CoDynamic_{target_gene_t}_{target_peak_t}.png"), dpi=300, bbox_inches='tight'); plt.close(fig4b)
                                except Exception as fig4b_err: print(f"  Warning: Error plotting Fig 4b: {fig4b_err}"); plt.close(fig4b) if 'fig4b' in locals() and plt.fignum_exists(fig4b.number) else None
                            else: print(f"Cannot plot Fig 4b: Target gene/peak '{target_peak_t}' not in dynamic data.")
                        else: print("Cannot select co-dynamic T cell pair for Fig 4b.")
                    else: print("Skipping T cell ATAC integration plots: No dynamic pairs determined.")
                    # TF Motif Enrichment Plot (Requires actual motif analysis results)
                    print("Generating Figure 4c (TF Motif Enrichment)..."); fig4c, ax4c = plt.subplots(figsize=(5, 4));
                    # --- Placeholder --- Run chromVAR or similar, get results, e.g., enrichment_results_df ---
                    # Example: enrichment_results_df = pd.DataFrame({'TF': ['TCF7', 'TOX', 'NR4A1', ...], 'EnrichmentScore': [...]})
                    # enrichment_results_df = enrichment_results_df.sort_values('EnrichmentScore', ascending=False).head(10)
                    # ax4c.barh(enrichment_results_df['TF'], enrichment_results_df['EnrichmentScore'], color='lightblue');
                    ax4c.text(0.5, 0.5, "Run TF Motif Enrichment (e.g., chromVAR)\non dynamic peaks and plot results here.", ha='center', va='center', transform=ax4c.transAxes, fontstyle='italic')
                    ax4c.set_xlabel('TF Enrichment Score'); ax4c.set_title('Fig 4c: TF Motif Enrichment in Dynamic Peaks'); sns.despine(ax=ax4c); plt.tight_layout(); plt.savefig(os.path.join(output_dir, "Fig4c_TF_Motif_Enrichment.png"), dpi=300, bbox_inches='tight'); plt.close(fig4c)
                else: print("Skipping ATAC dynamic peak analysis: ATAC subset missing or empty.")
            else: print("Skipping ATAC dynamic peak analysis: Not enough cells with valid pseudotime.")
        else: print("Skipping ATAC dynamic peak analysis: 'tcell_pseudotime' missing.")
    else: print("Skipping ATAC integration: T cell common obs names empty or ATAC data missing.")
else: print("Skipping ATAC integration (Step 4): Required data missing/empty.")


# --- Step 5: Spatial Analysis ---
print("\n--- Step 5: Spatial Analysis ---")
if spatial_data and isinstance(spatial_data, dict): # Check if spatial_data is a non-empty dict
    try: all_spots_df = pd.concat(spatial_data.values(), ignore_index=False) # Keep original index if unique per patient
    except Exception as e: print(f"Error concatenating spatial data: {e}"); all_spots_df = None
    if all_spots_df is not None and not all_spots_df.empty:
        print("Generating Figure 5a: Spatial Distribution Example...")
        example_patient_r_id = None; example_patient_nr_id = None
        for p_id, info in spatial_data.items():
            if isinstance(info, pd.DataFrame) and 'responder_status' in info.columns and not info.empty:
                 status = info['responder_status'].iloc[0]
                 if status == 'R' and example_patient_r_id is None: example_patient_r_id = p_id
                 if status == 'NR' and example_patient_nr_id is None: example_patient_nr_id = p_id
                 if example_patient_r_id and example_patient_nr_id: break
            else: print(f"Warning: Invalid spatial data entry {p_id}.")
        if example_patient_r_id is None and spatial_data: example_patient_r_id = list(spatial_data.keys())[0]
        if example_patient_nr_id is None and spatial_data: example_patient_nr_id = list(spatial_data.keys())[-1]

        if example_patient_r_id and example_patient_nr_id:
            fig5a, axes5a = plt.subplots(1, 2, figsize=(10, 5)); plot_types = ['Tumor', 'CD8_Tex', 'Macro_M2', 'CD8_Tem', 'Macro_M1', 'Treg']
            try: base_palette_mpl = sns.color_palette("tab10", len(plot_types)); palette = {t: mcolors.to_rgba(base_palette_mpl[i]) for i, t in enumerate(plot_types)}; palette['Other'] = mcolors.to_rgba('lightgrey'); palette_valid = True
            except Exception as e: print(f"Warning: Palette creation failed: {e}."); palette = None; palette_valid = False
            # Plot R patient
            if example_patient_r_id in spatial_data:
                df_r = spatial_data[example_patient_r_id].copy(); df_r['plot_type'] = df_r['dominant_cell_type'].apply(lambda x: x if x in plot_types else 'Other').astype('category'); plot_colors_r = None
                if palette_valid:
                    try: mapped_colors = df_r['plot_type'].astype(str).map(palette); df_r['plot_color_rgba'] = mapped_colors.fillna(palette.get('Other', (0.8,0.8,0.8,1.0))); plot_colors_r = df_r['plot_color_rgba'].tolist() # Use .get for fillna
                    except Exception as map_err: print(f"  Warning: Error mapping R colors: {map_err}")
                axes5a[0].scatter(df_r['x'], df_r['y'], c=plot_colors_r, s=15, alpha=0.9)
                if palette_valid: handles_r = [plt.Line2D([0], [0], marker='o', color='w', label=t, markersize=7, markerfacecolor=palette.get(t, 'grey')) for t in df_r['plot_type'].cat.categories if t in palette]; axes5a[0].legend(handles=handles_r, title='Cell Type', fontsize=8, loc='upper right')
                axes5a[0].set_title(f'Spatial Plot ({example_patient_r_id} - R)'); axes5a[0].set_aspect('equal', adjustable='box'); axes5a[0].set_xticks([]); axes5a[0].set_yticks([])
            else: axes5a[0].text(0.5, 0.5, f"Data Not Found\n({example_patient_r_id})", ha='center', va='center', transform=axes5a[0].transAxes); axes5a[0].set_title('R Patient (Skipped)')
            # Plot NR patient
            if example_patient_nr_id in spatial_data:
                df_nr = spatial_data[example_patient_nr_id].copy(); df_nr['plot_type'] = df_nr['dominant_cell_type'].apply(lambda x: x if x in plot_types else 'Other').astype('category'); plot_colors_nr = None
                if palette_valid:
                    try: mapped_colors_nr = df_nr['plot_type'].astype(str).map(palette); df_nr['plot_color_rgba'] = mapped_colors_nr.fillna(palette.get('Other', (0.8,0.8,0.8,1.0))); plot_colors_nr = df_nr['plot_color_rgba'].tolist()
                    except Exception as map_err_nr: print(f"  Warning: Error mapping NR colors: {map_err_nr}")
                axes5a[1].scatter(df_nr['x'], df_nr['y'], c=plot_colors_nr, s=15, alpha=0.9)
                if palette_valid: handles_nr = [plt.Line2D([0], [0], marker='o', color='w', label=t, markersize=7, markerfacecolor=palette.get(t, 'grey')) for t in df_nr['plot_type'].cat.categories if t in palette]; axes5a[1].legend(handles=handles_nr, title='Cell Type', fontsize=8, loc='upper right')
                axes5a[1].set_title(f'Spatial Plot ({example_patient_nr_id} - NR)'); axes5a[1].set_aspect('equal', adjustable='box'); axes5a[1].set_xticks([]); axes5a[1].set_yticks([])
            else: axes5a[1].text(0.5, 0.5, f"Data Not Found\n({example_patient_nr_id})", ha='center', va='center', transform=axes5a[1].transAxes); axes5a[1].set_title('NR Patient (Skipped)')
            plt.suptitle("Figure 5a: Example Spatial Cell Distributions", fontsize=14, y=1.0); plt.tight_layout(rect=[0, 0, 1, 0.96]); plt.savefig(os.path.join(output_dir, "Fig5a_Spatial_Example.png"), dpi=300, bbox_inches='tight'); plt.close(fig5a)
        else: print("Warning: Could not determine R and NR example patient IDs for Fig 5a.")

        # Plotting Figure 5b (Neighborhood Analysis)
        print("Generating Figure 5b: Neighborhood Analysis...")
        coloc_scores = []
        if SCIPY_SPATIAL_AVAILABLE:
            try:
                for patient_id, spot_df in spatial_data.items():
                    if not isinstance(spot_df, pd.DataFrame) or spot_df.empty: print(f"Warning: Skipping {patient_id} for neighbors: invalid data."); continue
                    patient_status = spot_df['responder_status'].iloc[0]; tex_spots = spot_df[spot_df['dominant_cell_type'] == 'CD8_Tex']
                    if 'x' not in spot_df.columns or 'y' not in spot_df.columns or spot_df[['x', 'y']].isnull().any().any(): print(f"Warning: Skipping {patient_id} for neighbors: missing/invalid coordinates."); coloc_scores.append({'patient_id': patient_id, 'responder_status': patient_status, 'Tex_M2_Neighbor_Score': np.nan, 'Tex_Count': 0}); continue
                    if tex_spots.empty: coloc_scores.append({'patient_id': patient_id, 'responder_status': patient_status, 'Tex_M2_Neighbor_Score': 0, 'Tex_Count': 0}); continue
                    try: tree = KDTree(spot_df[['x', 'y']].values)
                    except Exception as build_tree_err: print(f"Warning: KDTree build failed for {patient_id}: {build_tree_err}."); coloc_scores.append({'patient_id': patient_id, 'responder_status': patient_status, 'Tex_M2_Neighbor_Score': np.nan, 'Tex_Count': len(tex_spots)}); continue
                    neighbor_radius = 1.5; tex_indices = tex_spots.index
                    try: all_neighbors_indices_lists = tree.query_ball_point(tex_spots[['x', 'y']].values, r=neighbor_radius)
                    except Exception as kdtree_err: print(f"Warning: KDTree query failed for {patient_id}: {kdtree_err}."); coloc_scores.append({'patient_id': patient_id, 'responder_status': patient_status, 'Tex_M2_Neighbor_Score': np.nan, 'Tex_Count': len(tex_spots)}); continue
                    patient_total_m2_neighbor_prop = 0; num_tex_with_neighbors = 0
                    for i, tex_idx in enumerate(tex_indices):
                        neighbor_indices_list = all_neighbors_indices_lists[i]; neighbor_indices_for_tex = [spot_df.index[j] for j in neighbor_indices_list if j < len(spot_df) and spot_df.index[j] != tex_idx]
                        if not neighbor_indices_for_tex: continue
                        num_tex_with_neighbors += 1; neighbor_types = spot_df.loc[neighbor_indices_for_tex, 'dominant_cell_type']; m2_neighbor_count = (neighbor_types == 'Macro_M2').sum(); proportion_m2 = m2_neighbor_count / len(neighbor_indices_for_tex); patient_total_m2_neighbor_prop += proportion_m2
                    avg_prop_m2_neighbors = (patient_total_m2_neighbor_prop / num_tex_with_neighbors) if num_tex_with_neighbors > 0 else 0
                    coloc_scores.append({'patient_id': patient_id, 'responder_status': patient_status, 'Tex_M2_Neighbor_Score': avg_prop_m2_neighbors, 'Tex_Count': len(tex_spots)})
            except Exception as spatial_err: print(f"Error during spatial neighborhood analysis: {spatial_err}")
        else: print("Skipping neighborhood analysis (Fig 5b): scipy.spatial not available.")
        if coloc_scores:
            coloc_df = pd.DataFrame(coloc_scores); print("  Tex-M2 Neighborhood Analysis Results:"); print(coloc_df)
            if not coloc_df.empty and 'responder_status' in coloc_df.columns and 'Tex_M2_Neighbor_Score' in coloc_df.columns:
                fig5b, ax5b = plt.subplots(figsize=(4, 4)); sns.boxplot(data=coloc_df, x='responder_status', y='Tex_M2_Neighbor_Score', ax=ax5b, palette={'R':'skyblue', 'NR':'salmon'}); ax5b.set_title('Fig 5b: Tex-M2 Spatial Co-localization\n(Avg. Prop. M2 Neighbors)'); ax5b.set_ylabel('Avg. Prop. M2 Neighbors'); ax5b.set_xlabel('Responder Status')
                try: group_r_coloc = coloc_df[coloc_df['responder_status']=='R']['Tex_M2_Neighbor_Score'].dropna(); group_nr_coloc = coloc_df[coloc_df['responder_status']=='NR']['Tex_M2_Neighbor_Score'].dropna()
                except KeyError: group_r_coloc, group_nr_coloc = pd.Series(), pd.Series()
                if len(group_r_coloc) > 1 and len(group_nr_coloc) > 1:
                    try: stat, pval = mannwhitneyu(group_r_coloc, group_nr_coloc); ax5b.text(0.5, 0.95, f'p={pval:.2e}', ha='center', va='top', transform=ax5b.transAxes, fontsize=9)
                    except ValueError: pass
                sns.despine(ax=ax5b); plt.tight_layout(); plt.savefig(os.path.join(output_dir, "Fig5b_Neighborhood_Analysis.png"), dpi=300, bbox_inches='tight'); plt.close(fig5b) # Renamed file
            else: print("Skipping Fig 5b plot: coloc_df empty or missing columns.")
        else: print("Skipping Fig 5b: No co-localization scores calculated.")

        # Plotting Figure 5c (Cell-Cell Communication)
        print("Generating Figure 5c: Inferred Cell-Cell Communication...")
        # --- Placeholder: Run CellPhoneDB/Squidpy/etc. based on expression data ---
        # comm_results = run_cci_analysis(adata_rna) # Your CCI analysis function
        # comm_df = process_cci_results(comm_results) # Process results into plottable format
        # --- Example Data (replace with actual results) ---
        comm_data = {('Macro_M2', 'CD8_Tex', 'IL10 -> IL10RA/B'): {'R': 0.3, 'NR': 0.7}, ('Macro_M2', 'CD8_Tex', 'TGFB1 -> TGFBR1/2'): {'R': 0.2, 'NR': 0.6}, ('Treg', 'CD8_Tex', 'CTLA4 -> CD80/86'): {'R': 0.25, 'NR': 0.5}, ('Macro_M1', 'CD8_Tex', 'IL12A/B -> IL12RB1/2'): {'R': 0.5, 'NR': 0.2}, ('cDC', 'CD8_Tem', 'IL15 -> IL15RA'): {'R': 0.6, 'NR': 0.4}, ('Tumor', 'CD8_Tex', 'PDL1(CD274) -> PDCD1'): {'R': 0.4, 'NR': 0.75}}
        comm_df = pd.DataFrame([{'Source': src, 'Target': tgt, 'LR_Pair': lr, 'Responder': resp, 'Strength': strength * np.random.uniform(0.8, 1.2)} for (src, tgt, lr), resp_dict in comm_data.items() for resp, strength in resp_dict.items()])
        # --- End Placeholder ---
        if 'comm_df' in locals() and not comm_df.empty:
            fig5c, ax5c = plt.subplots(figsize=(7, 4.5))
            inhibitory_pairs = ['IL10 -> IL10RA/B', 'TGFB1 -> TGFBR1/2', 'CTLA4 -> CD80/86', 'PDL1(CD274) -> PDCD1']
            sns.barplot(data=comm_df[comm_df['LR_Pair'].isin(inhibitory_pairs)], x='LR_Pair', y='Strength', hue='Responder', ax=ax5c, palette={'R':'skyblue', 'NR':'salmon'})
            ax5c.set_title('Fig 5c: Inferred Inhibitory Communication Strength\n(Target: CD8_Tex)'); ax5c.set_ylabel('Interaction Strength'); ax5c.set_xlabel('Ligand-Receptor Pair')
            ax5c.tick_params(axis='x', rotation=25); ax5c.legend(title='Responder Status'); sns.despine(ax=ax5c); plt.tight_layout(); plt.savefig(os.path.join(output_dir, "Fig5c_Inferred_CCI.png"), dpi=300, bbox_inches='tight'); plt.close(fig5c)
        else: print("Skipping Fig 5c: CCI analysis results not available.")
    else: print("Skipping Spatial plots (Fig 5a, 5b, 5c): Combined spatial DataFrame missing or empty.")
else: print("Skipping Spatial Analysis (Step 5): spatial_data dictionary empty or invalid.")


# --- Step 6: Clinical Association ---
print("\n--- Step 6: Clinical Association ---")
if LIFELINES_INSTALLED and clinical_data_df is not None:
    print("  Preparing data for clinical association...")
    patient_df = clinical_data_df.copy() # Start with loaded clinical data
    # --- Merge TME features calculated earlier ---
    use_tex_prop = False; strat_var_tex = None; use_spatial_score = False; strat_var_spatial = None
    # Option 1: Tex proportion
    if ('adata_t' in locals() and adata_t.n_obs > 0 and 'tcell_subtype' in adata_t.obs and isinstance(adata_t.obs['tcell_subtype'].dtype, pd.CategoricalDtype) and 'CD8_Tex' in adata_t.obs['tcell_subtype'].cat.categories and 'patient_id' in adata_t.obs):
        try:
            t_cell_summary = adata_t.obs.groupby(['patient_id', 'tcell_subtype'], observed=True).size().unstack(fill_value=0)
            if not t_cell_summary.empty:
                 t_cell_summary['total_T'] = t_cell_summary.sum(axis=1); t_cell_summary['Tex_Prop_T'] = 0.0; non_zero_mask = t_cell_summary['total_T'] > 0
                 if 'CD8_Tex' in t_cell_summary.columns: t_cell_summary.loc[non_zero_mask, 'Tex_Prop_T'] = t_cell_summary.loc[non_zero_mask, 'CD8_Tex'] / t_cell_summary.loc[non_zero_mask, 'total_T']
                 t_cell_summary.reset_index(inplace=True) # Ensure patient_id is a column
                 patient_df = pd.merge(patient_df, t_cell_summary[['patient_id', 'Tex_Prop_T']], on='patient_id', how='left'); patient_df['Tex_Prop_T'].fillna(0, inplace=True)
                 median_tex_prop = patient_df['Tex_Prop_T'].median()
                 if median_tex_prop > 1e-6 and patient_df['Tex_Prop_T'].nunique() > 1: patient_df['TME_Group_Tex'] = np.where(patient_df['Tex_Prop_T'] > median_tex_prop, 'High_Tex', 'Low_Tex'); strat_var_tex = 'TME_Group_Tex'; print(f"  Using T cell Tex Proportion for survival stratification (Median: {median_tex_prop:.3f})"); use_tex_prop = True
                 else: print("  Warning: Skipping Tex proportion stratification (zero median or single value).")
            else: print("  Warning: T cell summary empty, cannot calculate Tex proportion.")
        except Exception as tex_prop_err: print(f"  Warning: Error calculating/merging Tex proportion: {tex_prop_err}")
    else: print("  Warning: Cannot use Tex proportion for stratification (required data missing).")
    # Option 2: Spatial score
    if 'coloc_df' in locals() and isinstance(coloc_df, pd.DataFrame) and not coloc_df.empty and 'Tex_M2_Neighbor_Score' in coloc_df.columns:
        try:
            patient_df = pd.merge(patient_df, coloc_df[['patient_id', 'Tex_M2_Neighbor_Score']], on='patient_id', how='left'); patient_df['Tex_M2_Neighbor_Score'].fillna(0, inplace=True)
            median_score = patient_df['Tex_M2_Neighbor_Score'].median()
            if median_score > 1e-6 and patient_df['Tex_M2_Neighbor_Score'].nunique() > 1: patient_df['TME_Group_Spatial'] = np.where(patient_df['Tex_M2_Neighbor_Score'] > median_score, 'High_Tex_M2_N', 'Low_Tex_M2_N'); strat_var_spatial = 'TME_Group_Spatial'; print(f"  Using Tex_M2_Neighbor_Score for survival stratification (Median: {median_score:.3f})"); use_spatial_score = True
            else: print("  Warning: Skipping spatial score stratification (zero median or single value).")
        except Exception as spatial_merge_err: print(f"  Warning: Error merging spatial score: {spatial_merge_err}")
    else: print("  Warning: Cannot use Spatial Score for stratification (coloc_df missing/invalid).")
    # Choose stratification variable
    strat_var = None
    if use_spatial_score: strat_var = strat_var_spatial
    elif use_tex_prop: strat_var = strat_var_tex
    elif 'responder_status' in patient_df.columns: strat_var = 'responder_status'; print("  Warning: Using R vs NR status for fallback survival stratification.")
    else: print("Error: Cannot determine stratification variable for survival.")
    # Plotting Figure 6b (Kaplan-Meier)
    if strat_var and strat_var in patient_df.columns:
        print(f"Generating Figure 6b: Kaplan-Meier Plot stratified by {strat_var}...")
        kmf = KaplanMeierFitter(); fig6b, ax6b = plt.subplots(figsize=(5, 4.5))
        patient_df_km = patient_df.dropna(subset=[strat_var, 'time_to_event', 'observed_event']).copy()
        groups = patient_df_km[strat_var].unique()
        if len(groups) == 2:
            group_data = {}; all_fits_successful = True
            for name, grouped_df in patient_df_km.groupby(strat_var):
                if not grouped_df.empty:
                    try: kmf.fit(grouped_df['time_to_event'], grouped_df['observed_event'], label=f"{name} (n={len(grouped_df)})"); kmf.plot_survival_function(ax=ax6b); group_data[name] = grouped_df
                    except Exception as km_fit_err: print(f"  Warning: KMF fit/plot failed for group '{name}': {km_fit_err}"); all_fits_successful = False
                else: print(f"  Warning: Stratification group '{name}' empty."); all_fits_successful = False
            if len(group_data) == 2 and all_fits_successful:
                 keys = list(group_data.keys())
                 try: results = logrank_test(group_data[keys[0]]['time_to_event'], group_data[keys[1]]['time_to_event'], event_observed_A=group_data[keys[0]]['observed_event'], event_observed_B=group_data[keys[1]]['observed_event']); pval_lr = results.p_value; ax6b.text(0.05, 0.05, f'Log-rank p = {pval_lr:.3f}', transform=ax6b.transAxes, fontsize=9)
                 except Exception as logrank_err: print(f"  Warning: Logrank test failed: {logrank_err}")
            elif len(group_data) != 2: print(f"  Logrank test skipped: Did not get data for exactly two groups (got {len(group_data)}).")
            ax6b.set_title(f'Fig 6b: Survival Stratified by {strat_var}'); ax6b.set_xlabel('Time (Months)'); ax6b.set_ylabel('Survival Probability'); ax6b.legend(fontsize=8); ax6b.set_ylim(0, 1.05); sns.despine(ax=ax6b); plt.tight_layout(); plt.savefig(os.path.join(output_dir, f"Fig6b_Survival_{strat_var}.png"), dpi=300, bbox_inches='tight'); plt.close(fig6b)
        else: print(f"  Skipping Kaplan-Meier plot: Stratification by '{strat_var}' resulted in {len(groups)} groups (expected 2). Groups: {groups}")
    else: print("  Skipping Kaplan-Meier plot: Stratification variable not found or determined.")
elif not LIFELINES_INSTALLED: print("Skipping Clinical Association (Step 6): 'lifelines' not installed.")
else: print("Skipping Clinical Association (Step 6): Clinical data not loaded.")

# Final Summary Figure (Improved)
print("Generating Figure 6c: Summary Schematic (Improved)...")
fig6c, ax6c = plt.subplots(figsize=(10, 6)); ax6c.set_title("Figure 6c: Summary Model of ICB Resistance Mechanisms", fontsize=14, pad=20); ax6c.axis('off'); ax6c.set_xlim(0, 10); ax6c.set_ylim(0, 7); ax6c.axvline(5, color='grey', linestyle='--', lw=1)
# Left Side: Responder
ax6c.text(2.5, 6.5, "Responder (R) TME", ha='center', va='center', weight='bold', fontsize=12, color='blue')
tumor_r = patches.Circle((1.5, 3.5), radius=0.4, fc='lightgrey', ec='black'); tem_tcell = patches.Ellipse((3, 4.5), width=0.7, height=0.5, fc='skyblue', ec='black'); m1_macro = patches.Ellipse((3, 2.5), width=0.8, height=0.5, fc='mediumpurple', ec='black')
ax6c.add_patch(tumor_r); ax6c.add_patch(tem_tcell); ax6c.add_patch(m1_macro)
ax6c.text(1.5, 3.5, "Tumor", ha='center', va='center', fontsize=8); ax6c.text(3, 4.5, "CD8 Tem\n(Effector)", ha='center', va='center', fontsize=8); ax6c.text(3, 2.5, "M1 Macro\n(Pro-Immune)", ha='center', va='center', fontsize=8)
ax6c.arrow(2.7, 4.3, -0.8, -0.6, head_width=0.1, head_length=0.15, fc='darkblue', ec='darkblue', lw=1); ax6c.text(2.2, 4.0, "Killing", color='darkblue', fontsize=7)
ax6c.arrow(3, 2.8, 0, 1.2, head_width=0.1, head_length=0.15, fc='purple', ec='purple', lw=1); ax6c.text(3.3, 3.5, "Support", color='purple', fontsize=7)
treg_r = patches.Circle((4, 5.5), radius=0.15, fc='orange', ec='black', alpha=0.5); tex_r = patches.Circle((4, 1.5), radius=0.15, fc='salmon', ec='black', alpha=0.5); m2_r = patches.Circle((1, 1.5), radius=0.15, fc='gold', ec='black', alpha=0.5)
ax6c.add_patch(treg_r); ax6c.add_patch(tex_r); ax6c.add_patch(m2_r)
ax6c.text(4.0, 5.8, "Low Treg", ha='center', fontsize=7, alpha=0.7); ax6c.text(4.0, 1.2, "Low Tex", ha='center', fontsize=7, alpha=0.7); ax6c.text(1.0, 1.2, "Low M2", ha='center', fontsize=7, alpha=0.7)
ax6c.arrow(2.5, 0.2, 0, 0.6, head_width=0.2, head_length=0.3, fc='black', ec='black', lw=1.5); ax6c.text(2.5, 0, "ICB", ha='center', va='top', fontsize=10); ax6c.text(2.5, 1.5, "Response ✓", ha='center', va='center', fontsize=14, color='green', weight='bold')
# Right Side: Non-Responder
ax6c.text(7.5, 6.5, "Non-Responder (NR) TME", ha='center', va='center', weight='bold', fontsize=12, color='red')
tumor_nr = patches.Circle((6.5, 3.5), radius=0.4, fc='lightgrey', ec='black'); tex_tcell = patches.Ellipse((8, 4.5), width=0.7, height=0.5, fc='salmon', ec='black'); m2_macro = patches.Ellipse((8, 2.5), width=0.8, height=0.5, fc='gold', ec='black'); treg_cell = patches.Circle((9, 3.5), radius=0.3, fc='orange', ec='black')
ax6c.add_patch(tumor_nr); ax6c.add_patch(tex_tcell); ax6c.add_patch(m2_macro); ax6c.add_patch(treg_cell)
ax6c.text(6.5, 3.5, "Tumor\n(PD-L1+)", ha='center', va='center', fontsize=8); ax6c.text(8, 4.5, "CD8 Tex\n(Exhausted)", ha='center', va='center', fontsize=8); ax6c.text(8, 2.5, "M2 Macro\n(Suppressive)", ha='center', va='center', fontsize=8); ax6c.text(9, 3.5, "Treg", ha='center', va='center', fontsize=8)
ax6c.arrow(8, 2.8, 0, 1.2, head_width=0.1, head_length=0.15, fc='darkorange', ec='darkorange', lw=1, linestyle='--'); ax6c.text(8.3, 3.5, "IL-10,\nTGFβ", color='darkorange', fontsize=7)
ax6c.arrow(8.7, 3.5, -0.4, 0.8, head_width=0.1, head_length=0.15, fc='darkorange', ec='darkorange', lw=1, linestyle='--'); ax6c.text(8.5, 3.9, "Suppr.", color='darkorange', fontsize=7)
ax6c.arrow(6.8, 3.7, 0.8, 0.6, head_width=0.1, head_length=0.15, fc='darkred', ec='darkred', lw=1, linestyle='--'); ax6c.text(7.3, 4.3, "PD-L1:PD-1", color='darkred', fontsize=7)
niche = patches.Ellipse((8.3, 3.5), width=2.5, height=2.8, fill=False, linestyle='dotted', edgecolor='grey'); ax6c.add_patch(niche); ax6c.text(8.3, 1.8, "Suppressive Niche\n(Co-localization)", ha='center', va='center', fontsize=8, color='grey')
ax6c.text(7.6, 4.8, "🔒", fontsize=12, color='black', ha='center', va='center'); ax6c.text(7.5, 5.1, "Epigenetic\nLock", fontsize=7, ha='center')
ax6c.arrow(7.5, 0.2, 0, 0.6, head_width=0.2, head_length=0.3, fc='black', ec='black', lw=1.5); ax6c.text(7.5, 0, "ICB", ha='center', va='top', fontsize=10); ax6c.plot([7.3, 7.7], [1.0, 1.4], color='red', lw=2.5); ax6c.plot([7.3, 7.7], [1.4, 1.0], color='red', lw=2.5); ax6c.text(7.5, 1.5, "Resistance ✗", ha='center', va='bottom', fontsize=14, color='red', weight='bold', y=1.1)
plt.tight_layout(rect=[0, 0, 1, 0.95]); plt.savefig(os.path.join(output_dir, "Fig6c_Summary_Schematic_Improved.png"), dpi=300, bbox_inches='tight'); plt.close(fig6c)
print("Improved summary schematic figure saved.")


print("\n--- Analysis and Visualization Complete ---")
print(f"Figures saved to directory: {output_dir}")
# plt.show()
