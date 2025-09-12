
import pandas as pd
import numpy as np

def replace_not_reported(df: pd.DataFrame, default_code: int = 99, custom_mappings: dict = None) -> pd.DataFrame:
    """
    Replace 'Not Reported', empty strings, and NaN in the DataFrame with specified numeric codes.

    Parameters:
    - df (pd.DataFrame): Input DataFrame
    - default_code (int): Default numeric code for Not Reported
    - custom_mappings (dict): Optional dict specifying codes per column, e.g., {'column1': 9, 'column2': 99}

    Returns:
    - pd.DataFrame: Cleaned DataFrame
    """
    df_cleaned = df.copy()
    for col in df_cleaned.columns:
        # Determine the replacement code for this column
        replacement_code = custom_mappings.get(col, default_code) if custom_mappings else default_code
        
        # Replace empty strings, 'Not Reported', and NaN with the replacement code
        df_cleaned[col] = df_cleaned[col].replace(['', 'Not Reported'], np.nan)
        df_cleaned[col] = df_cleaned[col].fillna(replacement_code)
    
    return df_cleaned

# Example usage
if __name__ == "__main__":
    # Example DataFrame
    data = {
        'building_type': ['1', '2', '', 'Not Reported', '4'],
        'roof_material': ['3', '', '2', 'Not Reported', ''],
    }
    df = pd.DataFrame(data)
    
    # Define custom codes if needed
    custom_codes = {'building_type': 0, 'roof_material': 9}
    
    # Replace 'Not Reported' values
    df_cleaned = replace_not_reported(df, default_code=99, custom_mappings=custom_codes)
    
    print(df_cleaned)
