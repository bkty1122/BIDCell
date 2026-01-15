import pandas as pd
import os

file_path = r'D:\2512-BROCK-CODING\BIDCell\data\previous-results-ref\amtl-min-max.csv'

try:
    # Read csv
    df = pd.read_csv(file_path)
    
    # Identify the value column (3rd column)
    val_col = df.columns[2]
    print(f"Aggregating values in column: {val_col}")

    # Create BaseMethod column
    # Split by ' - ' and take the first part, effectively merging all suffixes
    df['BaseMethod'] = df['Method'].apply(lambda x: x.split(' - ')[0].strip())

    # Group by Metric and BaseMethod, then calculate mean
    grouped_df = df.groupby(['Metric Category', 'BaseMethod'], as_index=False)[val_col].mean()

    # Rename BaseMethod back to Method to match original format
    grouped_df.rename(columns={'BaseMethod': 'Method'}, inplace=True)

    # Reorder columns to match original: Metric Category, Method, Value
    grouped_df = grouped_df[['Metric Category', 'Method', val_col]]

    # Sort for better readability (optional, but good for checking)
    grouped_df.sort_values(by=['Metric Category', 'Method'], inplace=True)

    # Save
    grouped_df.to_csv(file_path, index=False)
    print("Successfully aggregated means and saved to file.")
    print(grouped_df)

except Exception as e:
    print(f"An error occurred: {e}")
