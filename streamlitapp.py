

import streamlit as st
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
from sklearn.experimental import enable_iterative_imputer  # Enable IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor
from io import BytesIO

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

    # Step 1: Check for Essential Columns
    essential_columns = ['Site Num', 'Year', 'pH', 'TC %', 'TN %', 'Olsen P', 'AMN', 'BD']
    missing_columns = [col for col in essential_columns if col not in df.columns]
    if missing_columns:
        st.error(f"The uploaded dataset is missing essential columns: {missing_columns}")
        st.stop()

    # Step 2: Display Basic Information
    st.write("### Dataset Info")
    st.write("Shape:", df.shape)
    st.write("Missing Values:")
    st.write(df.isnull().sum())

    # Step 3: Handle Missing Values in Critical Columns
    critical_columns = ['pH', 'TC %', 'TN %', 'Olsen P', 'AMN', 'BD']
    df_cleaned = df.dropna(subset=critical_columns, how='any')
    rows_removed = len(df) - len(df_cleaned)
    st.write(f"### Removed {rows_removed} rows with missing values in critical columns.")

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

    # Step 6: Impute Missing Values Using IterativeImputer
    st.write("### Imputation Using IterativeImputer")
    numeric_columns = df_cleaned.select_dtypes(include=['number']).columns.tolist()
    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=10, random_state=0),
        max_iter=5,
        random_state=0
    )
    df_cleaned[numeric_columns] = imputer.fit_transform(df_cleaned[numeric_columns])
    st.write("### Imputed Data")
    st.dataframe(df_cleaned.head())

    # Step 7: Perform KS Test
    st.write("### Kolmogorov-Smirnov Test Results")
    ks_results = {}
    for column in trace_elements:
        if column in df_cleaned.columns:
            original = df[column].dropna() if column in df else []
            imputed = df_cleaned[column]
            if len(original) > 0:
                ks_stat, p_value = ks_2samp(original, imputed)
                ks_results[column] = {'KS Statistic': ks_stat, 'p-value': p_value}
    ks_results_df = pd.DataFrame(ks_results).T
    st.write(ks_results_df)

    # Step 8: Download Cleaned Dataset
    st.write("### Download Cleaned Dataset")
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_cleaned.to_excel(writer, index=False, sheet_name="Cleaned Data")
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

