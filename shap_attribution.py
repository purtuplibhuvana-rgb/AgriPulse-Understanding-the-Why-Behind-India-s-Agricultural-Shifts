import os
import json
import logging
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import shap

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_explanation(feature, val_before, val_after, shap_diff, yield_before, yield_after):
    """
    Generate a natural language explanation of a change point based on the top SHAP feature,
    the direction of the feature value shift, and the direction of the yield shift.
    """
    yield_direction = "recovery" if yield_after > yield_before else "decline"
    yield_verb = "increase" if yield_after > yield_before else "decrease"
    
    feat_direction = "increase" if val_after > val_before else "decrease"
    feat_direction_past = "increased" if val_after > val_before else "decreased"
    feat_direction_adj = "higher" if val_after > val_before else "lower"
    
    # Map raw feature names to descriptive terms
    feature_readable = {
        'Area': 'cultivation area',
        'Annual_Rainfall': 'annual rainfall',
        'Fertilizer': 'fertilizer usage',
        'Pesticide': 'pesticide application',
        'Season_encoded': 'seasonal cropping patterns',
        'Crop_Year': 'long-term temporal trends'
    }.get(feature, feature)
    
    # Match explanations based on feature and change patterns
    if feature == 'Annual_Rainfall':
        if feat_direction == "increase" and yield_direction == "recovery":
            return f"Increased annual rainfall post-break was a major driver of the yield recovery."
        elif feat_direction == "decrease" and yield_direction == "decline":
            return f"A significant reduction in annual rainfall post-break was the primary driver of the yield decline."
        else:
            return f"Shifts in annual rainfall patterns ({feat_direction_past}) played a key role in the post-break yield transition."
            
    elif feature in ['Fertilizer', 'Pesticide']:
        input_name = "fertilizer usage" if feature == 'Fertilizer' else "pesticide application"
        if feat_direction == "increase" and yield_direction == "recovery":
            return f"An increase in {input_name} post-break supported a strong yield recovery."
        elif feat_direction == "decrease" and yield_direction == "decline":
            return f"Reduced {input_name} post-break was a key contributor to the yield decline."
        else:
            return f"Changes in {input_name} post-break marked a significant shift in crop input efficiency and yield contribution."
            
    elif feature == 'Area':
        if feat_direction == "decrease" and yield_direction == "recovery":
            return f"A reduction in cultivation area post-break suggests consolidation toward more productive lands, driving yield recovery."
        elif feat_direction == "increase" and yield_direction == "decline":
            return f"Cultivation area expansion was associated with a decrease in average yield post-break, possibly due to marginal land use."
        else:
            return f"A post-break change in cultivation area ({feat_direction_past}) was a major factor in the yield transition."
            
    elif feature == 'Season_encoded':
        return f"Shifts in seasonal cropping patterns post-break contributed to a change in overall yield, indicating a transition in seasonal output."
        
    elif feature == 'Crop_Year':
        if yield_direction == "recovery":
            return f"Long-term technology improvements and temporal trends drove the yield recovery post-break."
        else:
            return f"Long-term temporal decline factors post-break contributed to the yield drop."
            
    # Fallback explanation
    return f"{feature_readable.capitalize()} had the most significant shift in yield contribution post-break, acting as a key factor in the yield {yield_direction}."


def run_shap_attribution(workspace_dir):
    """
    Run local model training and SHAP attribution for each significant change point.
    """
    yield_csv = os.path.join(workspace_dir, "crop_yield.csv")
    cps_json = os.path.join(workspace_dir, "significant_change_points.json")
    output_json = os.path.join(workspace_dir, "shap_results.json")
    
    if not os.path.exists(yield_csv) or not os.path.exists(cps_json):
        logger.error("Required yield data or change point JSON not found!")
        return
        
    df = pd.read_csv(yield_csv)
    with open(cps_json, 'r') as f:
        change_points = json.load(f)
        
    logger.info(f"Loaded {len(change_points)} significant change points.")
    
    # Establish a stable global mapping for Season column
    seasons = df['Season'].unique()
    season_mapping = {season: idx for idx, season in enumerate(seasons)}
    
    # Prepare list for SHAP results
    shap_results = []
    
    features = ['Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide', 'Season_encoded', 'Crop_Year']
    
    for idx, cp in enumerate(change_points):
        state = cp['state']
        crop = cp['crop']
        break_year = cp['break_year']
        
        # Filter target series
        group_df = df[(df['State'] == state) & (df['Crop'] == crop)].sort_values(['Crop_Year', 'Season']).copy()
        group_df = group_df.reset_index(drop=True)
        break_idx = cp['break_index']
        
        # 1. Windowing and Fallback Extraction
        fallback_type = "none"
        window_start = break_year - 3
        window_end = break_year + 2
        
        win_df = group_df[(group_df['Crop_Year'] >= window_start) & (group_df['Crop_Year'] <= window_end)].copy()
        
        before_mask = win_df.index < break_idx
        after_mask = win_df.index >= break_idx
        
        # Fallback 1: Expand window if before/after sections are too sparse (< 2 samples in either)
        if before_mask.sum() < 2 or after_mask.sum() < 2:
            fallback_type = "expanded_window"
            window_start = break_year - 5
            window_end = break_year + 4
            win_df = group_df[(group_df['Crop_Year'] >= window_start) & (group_df['Crop_Year'] <= window_end)].copy()
            before_mask = win_df.index < break_idx
            after_mask = win_df.index >= break_idx
            
        # Fallback 2: Pool data from the same crop in other states if still too sparse
        if before_mask.sum() < 2 or after_mask.sum() < 2:
            fallback_type = "pooled_states"
            window_start = break_year - 3
            window_end = break_year + 2
            
            # Extract target state crop data
            target_win_df = group_df[(group_df['Crop_Year'] >= window_start) & (group_df['Crop_Year'] <= window_end)].copy()
            target_win_df['is_before'] = target_win_df.index < break_idx
            
            # Extract same crop across all other states
            other_states_df = df[(df['State'] != state) & (df['Crop'] == crop)].sort_values(['Crop_Year', 'Season']).copy()
            other_win_df = other_states_df[(other_states_df['Crop_Year'] >= window_start) & (other_states_df['Crop_Year'] <= window_end)].copy()
            other_win_df['is_before'] = other_win_df['Crop_Year'] < break_year
            
            # Concatenate to form a larger training dataset
            win_df = pd.concat([target_win_df, other_win_df], ignore_index=True)
            
            # Mask specifically for the target state rows
            before_mask = (win_df['State'] == state) & win_df['is_before']
            after_mask = (win_df['State'] == state) & (~win_df['is_before'])
            
        # 2. Preprocess features
        win_df['Season_encoded'] = win_df['Season'].map(season_mapping)
        X = win_df[features]
        y = win_df['Yield']
        
        # Handle cases where even after pooling there is insufficient data (should be rare)
        if len(win_df) < 3:
            logger.warning(f"Skipping change point for {state} - {crop} in {break_year} due to extreme data sparsity.")
            continue
            
        # 3. Train Local Regression Model
        model = RandomForestRegressor(n_estimators=50, max_depth=3, random_state=42)
        model.fit(X, y)
        
        # 4. Compute SHAP Values
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
            
        # 5. Analyze SHAP Shifts for Target State Rows
        if fallback_type == "pooled_states":
            target_before_mask = before_mask.values
            target_after_mask = after_mask.values
            
            # If target state has no rows in before segment, fall back to pooled before rows
            if np.sum(target_before_mask) == 0:
                target_before_mask = win_df['is_before'].values
            # If target state has no rows in after segment, fall back to pooled after rows
            if np.sum(target_after_mask) == 0:
                target_after_mask = (~win_df['is_before']).values
        else:
            target_before_mask = (win_df.index < break_idx)
            target_after_mask = (win_df.index >= break_idx)
        
        # Extract target SHAP and feature values using standard numpy boolean masking
        shap_before = shap_values[target_before_mask]
        shap_after = shap_values[target_after_mask]
        
        X_before = X.values[target_before_mask]
        X_after = X.values[target_after_mask]
        
        # Calculate mean SHAP and feature values
        mean_shap_before = np.mean(shap_before, axis=0)
        mean_shap_after = np.mean(shap_after, axis=0)
        
        mean_val_before = np.mean(X_before, axis=0)
        mean_val_after = np.mean(X_after, axis=0)
        
        # Calculate SHAP difference
        shap_diffs = mean_shap_after - mean_shap_before
        abs_shap_diffs = np.abs(shap_diffs)
        
        # Identify top feature causing the break transition
        top_feat_idx = np.argmax(abs_shap_diffs)
        top_feature = features[top_feat_idx]
        top_shap_diff = shap_diffs[top_feat_idx]
        
        top_val_before = mean_val_before[top_feat_idx]
        top_val_after = mean_val_after[top_feat_idx]
        
        # Yield averages for target state
        yield_before = cp['mean_before']
        yield_after = cp['mean_after']
        
        # Generate Natural Language Explanation
        explanation = generate_explanation(
            top_feature, 
            top_val_before, 
            top_val_after, 
            top_shap_diff, 
            yield_before, 
            yield_after
        )
        
        # Prepare card data
        shap_card = {
            'state': state,
            'crop': crop,
            'break_year': break_year,
            'fallback_type': fallback_type,
            'top_feature': top_feature,
            'explanation': explanation,
            'feature_names': features,
            'shap_diffs': [float(d) for d in shap_diffs],
            'mean_shap_before': [float(v) for v in mean_shap_before],
            'mean_shap_after': [float(v) for v in mean_shap_after],
            'mean_val_before': [float(v) for v in mean_val_before],
            'mean_val_after': [float(v) for v in mean_val_after]
        }
        shap_results.append(shap_card)
        
        # Periodically log progress
        if (idx + 1) % 100 == 0:
            logger.info(f"Processed SHAP attribution for {idx + 1}/{len(change_points)} change points.")
            
    # Save SHAP results
    with open(output_json, 'w') as f:
        json.dump(shap_results, f, indent=4)
        
    logger.info(f"SHAP attribution complete. Saved to {output_json}")
    
if __name__ == "__main__":
    workspace_dir = r"d:\perso_pro\Crop health Analysis"
    run_shap_attribution(workspace_dir)
