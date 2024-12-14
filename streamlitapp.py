

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.experimental import enable_iterative_imputer  # Required to enable IterativeImputer
from sklearn.impute import IterativeImputer

# App Title
st.title("Eco Soil Insights AKL - Soil Data Cleaning Dashboard")
st.write("Upload your soil dataset and perform cleaning, validation, and visualization.")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (Excel format)", type=["xlsx"])

if uploaded_file:
    # Load the dataset
    df = pd.read_excel(uploaded_file)
    st.write("### Raw Data Preview")
    st.dataframe(df.head())

    # Validation 1: Check for essential columns
    essential_columns = ['Site Num', 'Year', 'pH', 'TC %', 'TN %', 'Olsen P', 'AMN', 'BD']
    missing_columns = [col for col in essential_columns if col not in df.columns]
    if missing_columns:
        st.error(f"The uploaded dataset is missing essential columns: {missing_columns}")
        st.stop()
    else:
        st.success("All essential columns are present.")

    # Display dataset info
    st.write("### Dataset Info")
    st.write("Number of rows and columns:", df.shape)
    st.write("### Missing Values in Each Column")
    st.write(df.isnull().sum())

    # Step 1: Handle Missing Values in Critical Columns
    critical_columns = ['pH', 'TC %', 'TN %', 'Olsen P', 'AMN', 'BD']
    df_cleaned = df.dropna(subset=critical_columns, how='any')
    rows_removed = len(df) - len(df_cleaned)
    st.write(f"### Removed {rows_removed} rows with missing values in critical columns.")

    # Step 2: Remove Duplicate Rows
    initial_rows = len(df_cleaned)
    df_cleaned.drop_duplicates(inplace=True)
    st.write(f"### Removed {initial_rows - len(df_cleaned)} duplicate rows.")

    # Step 3: Extract Sample Count
    if 'Site No.1' in df_cleaned.columns:
        df_cleaned['Sample Count'] = df_cleaned['Site No.1'].str.extract(r'-(\d{2})$').astype(float)
        st.write("### Extracted Sample Count")
        st.dataframe(df_cleaned[['Site No.1', 'Sample Count']].head())

    # Step 4: Assign Periods Based on Year
    conditions = [
        (df_cleaned['Year'] >= 1995) & (df_cleaned['Year'] <= 2000),
        (df_cleaned['Year'] >= 2008) & (df_cleaned['Year'] <= 2012),
        (df_cleaned['Year'] >= 2013) & (df_cleaned['Year'] <= 2017),
        (df_cleaned['Year'] >= 2018) & (df_cleaned['Year'] <= 2023)
    ]
    period_labels = ['1995-2000', '2008-2012', '2013-2017', '2018-2023']
    df_cleaned['Period'] = np.select(conditions, period_labels, default='Unknown')
    st.write("### Assigned Periods")
    st.dataframe(df_cleaned[['Year', 'Period']].head())

    # Validation 2: Check unassigned periods
    unassigned_periods = df_cleaned[df_cleaned['Period'] == 'Unknown']
    if not unassigned_periods.empty:
        st.warning(f"{len(unassigned_periods)} rows have an unassigned period. Please verify the year range.")
        st.dataframe(unassigned_periods)

    # Step 5: Handle "<" Values in Trace Element Columns
    trace_elements = ['As', 'Cd', 'Cr', 'Cu', 'Ni', 'Pb', 'Zn']
    for column in trace_elements:
        if column in df_cleaned.columns:
            df_cleaned[column] = df_cleaned[column].apply(
                lambda x: float(x[1:]) / 2 if isinstance(x, str) and x.startswith('<') else x
            )
    st.write("### Updated Columns After Handling '<' Values")
    st.dataframe(df_cleaned[trace_elements].head())

    # Step 6: Impute Missing Values Using IterativeImputer
    st.write("### Imputation Using IterativeImputer")
    non_predictive_columns = ['Site No.1', 'Site Num', 'Year', 'Sample Count', 'Period']
    df_for_imputation = df_cleaned.drop(columns=non_predictive_columns, errors="ignore")

    # Separate categorical and numerical columns
    categorical_columns = df_for_imputation.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_columns = df_for_imputation.select_dtypes(include=['number']).columns.tolist()

    # One-hot encode categorical variables
    df_encoded = pd.get_dummies(df_for_imputation, columns=categorical_columns, drop_first=False)

    # Apply IterativeImputer for missing value imputation
    imputer = IterativeImputer(max_iter=5, random_state=0)
    imputed_data = imputer.fit_transform(df_encoded)
    df_imputed = pd.DataFrame(imputed_data, columns=df_encoded.columns)

    # Map one-hot-encoded columns back to original categorical columns
    for col in categorical_columns:
        encoded_columns = [c for c in df_encoded.columns if c.startswith(f"{col}_")]
        df_imputed[col] = df_imputed[encoded_columns].idxmax(axis=1).str[len(col) + 1:]
        df_imputed = df_imputed.drop(columns=encoded_columns)

    # Reattach non-predictive columns to the imputed dataset
    df_final = pd.concat([df_cleaned[non_predictive_columns].reset_index(drop=True), df_imputed], axis=1)
    st.write("### Dataset After Imputation")
    st.dataframe(df_final.head())

    # Step 7: Perform KS Test
    st.write("### Kolmogorov-Smirnov Test Results")
    ks_results = {}
    for column in trace_elements:
        if column in df_cleaned.columns and column in df_final.columns:
            ks_stat, p_value = ks_2samp(df_cleaned[column].dropna(), df_final[column].dropna())
            ks_results[column] = {'KS Statistic': ks_stat, 'p-value': p_value}
    ks_results_df = pd.DataFrame(ks_results).T
    st.write(ks_results_df)

    # Validation 3: Highlight significant KS test results
    significant_differences = [col for col, result in ks_results.items() if result['p-value'] < 0.05]
    if significant_differences:
        st.warning(f"Significant distribution differences detected in: {significant_differences}")

    # Step 8: Download Cleaned Dataset
    st.write("### Download Cleaned Dataset")
    cleaned_file = df_final.to_excel(index=False, engine="openpyxl")
    st.download_button(
        label="Download Cleaned Dataset",
        data=cleaned_file,
        file_name="cleaned_soil_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.write("Please upload a dataset to start the cleaning process.")
