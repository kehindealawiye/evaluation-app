import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, f_oneway
import statsmodels.formula.api as smf
import io

st.set_page_config(page_title="All-in-One Survey Analyzer", layout="wide")
st.title("Final All-in-One Survey Analysis App (Improved)")

uploaded_file = st.file_uploader("Upload Excel or CSV file", type=["csv", "xlsx"])

def detect_type(series):
    try:
        pd.to_datetime(series)
        return "Datetime"
    except:
        if pd.api.types.is_numeric_dtype(series):
            return "Numeric"
        elif pd.api.types.is_string_dtype(series):
            if series.nunique() < 10:
                return "Categorical"
            elif series.nunique() < 30:
                return "Categorical"
            else:
                return "Text"
        return "Unknown"

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)

    st.subheader("Raw Data Preview")
    st.dataframe(df.head())

    column_types = {col: detect_type(df[col]) for col in df.columns}

    st.markdown("## Multi-Chart Dashboard (with Static/Interactive Toggle)")
    display_mode = st.radio("Choose chart mode", ["Interactive", "Static"], horizontal=True)

    for col in df.columns:
        col_type = column_types[col]
        st.markdown(f"### {col} ({col_type})")

        try:
            if col_type == "Categorical":
                data = df[col].value_counts().reset_index()
                data.columns = [col, "Count"]
                if display_mode == "Interactive":
                    st.plotly_chart(px.bar(data, x=col, y="Count", color=col, title=f"Bar Chart - {col}"))
                    st.plotly_chart(px.pie(data, names=col, values="Count", title=f"Pie Chart - {col}"))
                else:
                    fig, ax = plt.subplots()
                    sns.barplot(data=data, x=col, y="Count", ax=ax)
                    ax.set_title(f"Bar Chart - {col}")
                    st.pyplot(fig)

            elif col_type == "Numeric":
                if display_mode == "Interactive":
                    st.plotly_chart(px.histogram(df, x=col, nbins=20, title=f"Histogram - {col}"))
                    st.plotly_chart(px.box(df, y=col, title=f"Boxplot - {col}"))
                else:
                    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
                    sns.histplot(df[col], kde=True, ax=axs[0])
                    axs[0].set_title(f"Histogram - {col}")
                    sns.boxplot(y=df[col], ax=axs[1])
                    axs[1].set_title(f"Boxplot - {col}")
                    st.pyplot(fig)

            elif col_type == "Text":
                text = " ".join(df[col].dropna().astype(str))
                if len(text.split()) > 10:
                    wordcloud = WordCloud(width=800, height=400).generate(text)
                    fig, ax = plt.subplots()
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.info("Not enough text for word cloud.")

        except Exception as e:
            st.warning(f"Could not chart {col}: {e}")

    st.markdown("## Statistical Test Builder (Safe & Smart)")

    dep = st.selectbox("Select Dependent Variable", df.columns)
    dep_type = column_types[dep]

    if dep_type == "Numeric":
        indep_options = [col for col in df.columns if col != dep and column_types[col] in ["Categorical", "Numeric"]]
    elif dep_type == "Categorical":
        indep_options = [col for col in df.columns if col != dep and column_types[col] == "Categorical"]
    else:
        indep_options = []

    if indep_options:
        indep = st.selectbox("Select Independent Variable", indep_options)
        indep_type = column_types[indep]
        df_test = df[[dep, indep]].dropna()

        if st.button("Run Test"):
            try:
                if len(df_test) < 10:
                    st.warning("Too few rows for meaningful statistical test (min 10).")
                else:
                    if dep_type == "Categorical" and indep_type == "Categorical":
                        table = pd.crosstab(df_test[dep], df_test[indep])
                        chi2, p, _, _ = chi2_contingency(table)
                        st.write(f"Chi-square p = {p:.4f}")
                        st.success("Significant!" if p < 0.05 else "Not significant.")

                    elif dep_type == "Numeric" and indep_type == "Categorical":
                        groups = [group[dep] for _, group in df_test.groupby(indep)]
                        f_stat, p_val = f_oneway(*groups)
                        st.write(f"ANOVA p = {p_val:.4f}")
                        st.success("Significant!" if p_val < 0.05 else "Not significant.")

                    elif dep_type == "Numeric" and indep_type == "Numeric":
                        model = smf.ols(f"{dep} ~ {indep}", data=df_test).fit()
                        st.text(model.summary())
                        p_val = model.pvalues[1]
                        r2 = model.rsquared
                        st.success(f"Significant (R² = {r2:.2f})" if p_val < 0.05 else f"Not significant (R² = {r2:.2f})")
            except Exception as e:
                st.error(f"Test error: {e}")
    else:
        st.warning("No valid independent variables available for the selected dependent variable.")

    st.markdown("## Export Section")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Full Dataset (CSV)", csv, "survey_data.csv", "text/csv")
