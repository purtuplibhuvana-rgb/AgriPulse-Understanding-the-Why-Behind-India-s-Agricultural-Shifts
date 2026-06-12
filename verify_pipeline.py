import os
import json
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)

def verify_and_fix(workspace_dir):
    cps_path = os.path.join(workspace_dir, 'significant_change_points.json')
    shap_path = os.path.join(workspace_dir, 'shap_results.json')
    # Load data
    logger.info('Loading change points and SHAP results...')
    change_points = load_json(cps_path)
    shap_results = load_json(shap_path)

    # Align lengths
    if len(change_points) != len(shap_results):
        logger.warning(f'Length mismatch: {len(change_points)} vs {len(shap_results)}. Trimming to shortest.')
        min_len = min(len(change_points), len(shap_results))
        change_points = change_points[:min_len]
        shap_results = shap_results[:min_len]

    # Ensure metadata consistency and clean NaNs
    fixed_shap = []
    for idx, (cp, shp) in enumerate(zip(change_points, shap_results)):
        for field in ['state', 'crop', 'break_year']:
            if cp.get(field) != shp.get(field):
                logger.warning(f'Row {idx} field {field} mismatch – correcting.')
                shp[field] = cp.get(field)
        # Replace NaN values in numeric lists
        for arr_key in ['shap_diffs', 'mean_shap_before', 'mean_shap_after',
                        'mean_val_before', 'mean_val_after']:
            arr = shp.get(arr_key, [])
            shp[arr_key] = [0.0 if isinstance(v, float) and np.isnan(v) else v for v in arr]
        fixed_shap.append(shp)

    cleaned_path = os.path.join(workspace_dir, 'shap_results_verified.json')
    save_json(fixed_shap, cleaned_path)
    logger.info(f'Cleaned SHAP results saved to {cleaned_path}')

    # Produce top‑5 summary per state‑crop based on absolute shift_std
    summary = []
    groups = {}
    for cp in change_points:
        key = (cp['state'], cp['crop'])
        groups.setdefault(key, []).append(cp)
    for (state, crop), cps in groups.items():
        top5 = sorted(cps, key=lambda x: abs(x.get('shift_std', 0)), reverse=True)[:5]
        for entry in top5:
            summary.append({
                'state': state,
                'crop': crop,
                'break_year': entry['break_year'],
                'break_season': entry.get('break_season'),
                'shift_std': entry.get('shift_std'),
                'shift_pct': entry.get('shift_pct'),
                'explanation': entry.get('explanation')
            })
    top5_path = os.path.join(workspace_dir, 'top5_summary.json')
    save_json(summary, top5_path)
    logger.info(f'Top‑5 summary saved to {top5_path}')
    return cleaned_path, top5_path

if __name__ == '__main__':
    workspace = r"d:\\perso_pro\\Crop health Analysis"
    verify_and_fix(workspace)
