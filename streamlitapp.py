
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

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

    # Step 5: Handle "<" Values in Trace Element Columns
    trace_elements = ['As', 'Cd', 'Cr', 'Cu', 'Ni', 'Pb', 'Zn']
    for column in trace_elements:
        if column in df_cleaned.columns:
            df_cleaned[column] = df_cleaned[column].apply(
                lambda x: float(x[1:]) / 2 if isinstance(x, str) and x.startswith('<') else x
            )
    st.write("### Updated Columns After Handling '<' Values")
    st.dataframe(df_cleaned[trace_elements].head())

    # Step 6: Impute Missing Values Using IterativeImputer with Random Forest
    st.write("### Imputation Using IterativeImputer with Random Forest")
    non_predictive_columns = ['Site No.1', 'Site Num', 'Year', 'Sample Count', 'Period']
    df_for_imputation = df_cleaned.drop(columns=non_predictive_columns, errors="ignore")

    # Use Random Forest as the estimator in IterativeImputer
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=50, random_state=0),
        max_iter=5,
        random_state=0
    )

    # Impute missing values
    numeric_columns = df_for_imputation.select_dtypes(include=['number']).columns
    imputed_data = imputer.fit_transform(df_for_imputation[numeric_columns])
    df_imputed = pd.DataFrame(imputed_data, columns=numeric_columns)
    df_final = pd.concat([df_cleaned[non_predictive_columns].reset_index(drop=True), df_imputed], axis=1)
    st.write("### Dataset After Imputation")
    st.dataframe(df_final.head())

    # Step 7: Perform KS Test
    st.write("### Kolmogorov-Smirnov Test Results")
    ks_results = {}
    for column in trace_elements:
        if column in df_cleaned.columns and column in df_final.columns:
            original = df_cleaned[column].dropna().astype(float)
            imputed = df_final[column].dropna().astype(float)
            if not original.empty and not imputed.empty:
                ks_stat, p_value = ks_2samp(original, imputed)
                ks_results[column] = {'KS Statistic': ks_stat, 'p-value': p_value}
    ks_results_df = pd.DataFrame(ks_results).T
    st.write(ks_results_df)

    # Step 8: Download Cleaned Dataset
    st.write("### Download Cleaned Dataset")
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_final.to_excel(writer, index=False, sheet_name="Cleaned Data")
        writer.save()
    output.seek(0)
    st.download_button(
        label="Download Cleaned Dataset",
        data=output,
        file_name="cleaned_soil_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.write("Please upload a dataset to start the cleaning process.")

