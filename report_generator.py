import os
import json
import logging
import pandas as pd
from ndvi_context import NDVIContextLayer

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_report(workspace_dir):
    """
    Combine results from previous stages to produce a final anomaly report JSON.
    """
    cps_json = os.path.join(workspace_dir, "significant_change_points.json")
    shap_json = os.path.join(workspace_dir, "shap_results.json")
    ndvi_csv = os.path.join(workspace_dir, "NDVI Prediction.csv")
    output_json = os.path.join(workspace_dir, "anomaly_report.json")
    
    if not all(os.path.exists(p) for p in [cps_json, shap_json, ndvi_csv]):
        logger.error("Missing intermediate files (change points, SHAP results, or NDVI dataset)!")
        return
        
    logger.info("Loading change points and SHAP results...")
    with open(cps_json, 'r') as f:
        change_points = json.load(f)
    with open(shap_json, 'r') as f:
        shap_results = json.load(f)
        
    # Build a lookup dictionary for SHAP results
    shap_lookup = {}
    for r in shap_results:
        key = (r['state'], r['crop'], r['break_year'])
        shap_lookup[key] = r
        
    # Initialize the NDVI Context Layer
    logger.info("Initializing NDVI Context Layer...")
    ndvi_layer = NDVIContextLayer(ndvi_csv)
    
    anomaly_cards = []
    logger.info("Generating anomaly cards...")
    
    for idx, cp in enumerate(change_points):
        state = cp['state']
        crop = cp['crop']
        break_year = cp['break_year']
        
        # Get matching SHAP info
        key = (state, crop, break_year)
        shap_info = shap_lookup.get(key)
        
        if not shap_info:
            logger.warning(f"No SHAP attribution found for {state} - {crop} in {break_year}. Skipping.")
            continue
            
        # Get average annual rainfall in the window (feature Annual_Rainfall is at index 1)
        mean_rainfall_before = shap_info['mean_val_before'][1]
        mean_rainfall_after = shap_info['mean_val_after'][1]
        avg_rainfall_win = (mean_rainfall_before + mean_rainfall_after) / 2.0
        
        # Query NDVI context
        ndvi_context = ndvi_layer.get_ndvi_context(avg_rainfall_win)
        
        # Build anomaly card
        card = {
            'state': state,
            'crop': crop,
            'break_year': break_year,
            'break_season': cp['break_season'],
            'yield_before': cp['mean_before'],
            'yield_after': cp['mean_after'],
            'yield_shift': cp['shift_magnitude'],
            'yield_pct_change': cp['shift_pct'],
            'p_value': cp['p_value'],
            'shift_in_std': cp['shift_std'],
            'top_feature': shap_info['top_feature'],
            'explanation': shap_info['explanation'],
            'fallback_type': shap_info['fallback_type'],
            'ndvi_context': ndvi_context,
            'shap_details': {
                'feature_names': shap_info['feature_names'],
                'shap_diffs': shap_info['shap_diffs'],
                'mean_shap_before': shap_info['mean_shap_before'],
                'mean_shap_after': shap_info['mean_shap_after'],
                'mean_val_before': shap_info['mean_val_before'],
                'mean_val_after': shap_info['mean_val_after']
            }
        }
        anomaly_cards.append(card)
        
    # Save the combined anomaly report
    with open(output_json, 'w') as f:
        json.dump(anomaly_cards, f, indent=4)
        
    logger.info(f"Anomaly report generated successfully. Saved {len(anomaly_cards)} cards to {output_json}")

if __name__ == "__main__":
    workspace_dir = r"d:\perso_pro\Crop health Analysis"
    generate_report(workspace_dir)
