import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
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
    df_cleaned = df_cleaned.drop_duplicates()
    st.write(f"### Removed {initial_rows - len(df_cleaned)} duplicate rows.")

    # Step 3: Impute Missing Values Using IterativeImputer with Random Forest
    st.write("### Imputation Using IterativeImputer with Random Forest")
    non_predictive_columns = ['Site Num', 'Year']
    df_for_imputation = df_cleaned.drop(columns=non_predictive_columns, errors="ignore")

    # Convert non-numeric columns to NaN for imputation
    df_for_imputation = df_for_imputation.apply(pd.to_numeric, errors='coerce')

    imputer = IterativeImputer(
        estimator=RandomForestRegressor(n_estimators=10, random_state=0),
        max_iter=10,
        random_state=0
    )
    imputed_data = imputer.fit_transform(df_for_imputation)
    df_imputed = pd.DataFrame(imputed_data, columns=df_for_imputation.columns)

    # Reattach non-predictive columns
    df_final = pd.concat([df_cleaned[non_predictive_columns].reset_index(drop=True), df_imputed], axis=1)
    st.write("### Dataset After Imputation")
    st.dataframe(df_final.head())

    # Step 4: Download Cleaned Dataset
    st.write("### Download Cleaned Dataset")
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        df_final.to_excel(writer, index=False, sheet_name="Cleaned Data")
        writer.close()
    output.seek(0)
    st.download_button(
        label="Download Cleaned Dataset",
        data=output,
        file_name="cleaned_soil_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )

else:
    st.write("Please upload a dataset to start the cleaning process.")

