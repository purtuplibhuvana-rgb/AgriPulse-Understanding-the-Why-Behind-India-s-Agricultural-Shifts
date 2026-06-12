import os
import json
import logging
import pandas as pd
import numpy as np
import ruptures as rpt
from scipy import stats

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def detect_change_points(yield_path, output_path):
    """
    Load the crop yield dataset, run change point detection on eligible State-Crop combinations,
    filter for statistically significant breaks, and save the results.
    """
    if not os.path.exists(yield_path):
        logger.error(f"Input file not found at {yield_path}")
        return 0
    
    logger.info(f"Loading yield dataset from {yield_path}...")
    df = pd.read_csv(yield_path)
    
    # 1. Identify combinations with at least 8 unique years of data
    year_counts = df.groupby(['State', 'Crop'])['Crop_Year'].nunique().reset_index(name='unique_years')
    valid_groups = year_counts[year_counts['unique_years'] >= 8]
    logger.info(f"Total State-Crop combinations: {len(year_counts)}")
    logger.info(f"Combinations with >= 8 unique years: {len(valid_groups)}")
    
    significant_change_points = []
    total_detected = 0
    
    # 2. Run change point detection for each valid group
    for idx, row in valid_groups.iterrows():
        state = row['State']
        crop = row['Crop']
        
        # Filter and sort stably by Crop_Year and Season
        group_df = df[(df['State'] == state) & (df['Crop'] == crop)].sort_values(['Crop_Year', 'Season']).copy()
        
        # Get raw yield values
        yield_values = group_df['Yield'].values
        years = group_df['Crop_Year'].values
        seasons = group_df['Season'].values
        
        # Handle zero-variance edge cases
        series_mean = np.mean(yield_values)
        series_std = np.std(yield_values)
        if series_std < 1e-8:
            logger.debug(f"Skipping {state} - {crop} due to zero variance in Yield.")
            continue
            
        # Standardize for PELT scale-invariance
        yield_scaled = (yield_values - series_mean) / series_std
        
        # Run ruptures PELT
        # - model='l2': detects shifts in the mean of the standardized yield
        # - min_size=3: ensures before/after segments are at least 3 points for statistical tests
        # - pen=3.0: penalty term tuned to catch distinct structural shifts
        try:
            algo = rpt.Pelt(model='l2', min_size=3).fit(yield_scaled)
            result = algo.predict(pen=3.0)
            
            # Change points are indices where segments end (excluding the last element which is series length)
            change_point_indices = result[:-1]
            
            for cp_idx in change_point_indices:
                # The year of the break is the year of the first row in the after segment
                break_year = int(years[cp_idx])
                break_season = seasons[cp_idx]
                
                # Split segments
                y_before = yield_values[:cp_idx]
                y_after = yield_values[cp_idx:]
                
                # Compute statistics
                mean_before = float(np.mean(y_before))
                mean_after = float(np.mean(y_after))
                shift = mean_after - mean_before
                pct_change = float((shift / (mean_before + 1e-8)) * 100)
                shift_in_std = float(abs(shift) / series_std)
                
                # Two-sample Welch's t-test (unequal variances assumed)
                # Ensure we have enough data points to compute stats
                p_val = 1.0
                if len(y_before) >= 2 and len(y_after) >= 2:
                    _, p_val = stats.ttest_ind(y_before, y_after, equal_var=False)
                    if np.isnan(p_val):
                        p_val = 1.0
                p_val = float(p_val)
                
                total_detected += 1
                
                # 3. Filter for significance
                # Shift exceeds 1 standard deviation OR p-value < 0.05
                is_sig = (shift_in_std > 1.0) or (p_val < 0.05)
                
                if is_sig:
                    cp_record = {
                        'state': state,
                        'crop': crop,
                        'break_year': break_year,
                        'break_season': break_season,
                        'break_index': int(cp_idx),
                        'mean_before': mean_before,
                        'mean_after': mean_after,
                        'shift_magnitude': shift,
                        'shift_pct': pct_change,
                        'shift_std': shift_in_std,
                        'p_value': p_val,
                        'series_std': float(series_std)
                    }
                    significant_change_points.append(cp_record)
                    
        except Exception as e:
            logger.warning(f"Error running change point detection for {state} - {crop}: {e}")
            continue
            
    # Save significant change points to JSON
    with open(output_path, 'w') as f:
        json.dump(significant_change_points, f, indent=4)
        
    logger.info(f"Change point detection completed.")
    logger.info(f"Total raw change points detected: {total_detected}")
    logger.info(f"Significant change points (filtered): {len(significant_change_points)}")
    
    return len(significant_change_points)

if __name__ == "__main__":
    workspace_dir = r"d:\perso_pro\Crop health Analysis"
    yield_csv = os.path.join(workspace_dir, "crop_yield.csv")
    output_json = os.path.join(workspace_dir, "significant_change_points.json")
    
    detect_change_points(yield_csv, output_json)
