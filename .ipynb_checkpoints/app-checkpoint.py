import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from scipy.stats import chi2_contingency, f_oneway
import statsmodels.formula.api as smf
import io

st.set_page_config(page_title="Survey Analyzer (Upgraded)", layout="wide")
st.title("Advanced Evaluation Study Analysis App")

uploaded_file = st.file_uploader("Upload Excel or CSV", type=["csv", "xlsx"])

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
    st.subheader("Raw Data")
    st.dataframe(df.head())

    st.markdown("## X–Y Chart Explorer (with Chart Mode Option)")

    x_col = st.selectbox("X variable", df.columns)
    y_col = st.selectbox("Y variable", [c for c in df.columns if c != x_col])

    x_type = detect_type(df[x_col])
    y_type = detect_type(df[y_col])
    chart_mode = st.radio("Chart Type", ["Interactive (Plotly)", "Static (PNG)"])

    cleaned_df = df[[x_col, y_col]].dropna()

    try:
        if x_type == "Numeric" and y_type == "Numeric":
            if chart_mode == "Interactive (Plotly)":
                fig = px.scatter(cleaned_df, x=x_col, y=y_col, title=f"Scatter: {x_col} vs {y_col}")
                st.plotly_chart(fig)
            else:
                fig, ax = plt.subplots()
                sns.scatterplot(data=cleaned_df, x=x_col, y=y_col, ax=ax)
                ax.set_title(f"Scatter: {x_col} vs {y_col}")
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                st.pyplot(fig)
                st.download_button("Download Chart", buf.getvalue(), f"{x_col}_vs_{y_col}_scatter.png", "image/png")

        elif x_type == "Categorical" and y_type == "Numeric":
            if chart_mode == "Interactive (Plotly)":
                fig = px.box(cleaned_df, x=x_col, y=y_col, title=f"Boxplot: {y_col} by {x_col}")
                st.plotly_chart(fig)
            else:
                fig, ax = plt.subplots()
                sns.boxplot(data=cleaned_df, x=x_col, y=y_col, ax=ax)
                ax.set_title(f"Boxplot: {y_col} by {x_col}")
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                st.pyplot(fig)
                st.download_button("Download Chart", buf.getvalue(), f"{y_col}_by_{x_col}_boxplot.png", "image/png")

        elif x_type == "Datetime" and y_type == "Numeric":
            cleaned_df = cleaned_df.sort_values(by=x_col)
            if chart_mode == "Interactive (Plotly)":
                fig = px.line(cleaned_df, x=x_col, y=y_col, title=f"Line: {y_col} over {x_col}")
                st.plotly_chart(fig)
            else:
                fig, ax = plt.subplots()
                ax.plot(cleaned_df[x_col], cleaned_df[y_col])
                ax.set_title(f"Line: {y_col} over {x_col}")
                buf = io.BytesIO()
                fig.savefig(buf, format="png")
                st.pyplot(fig)
                st.download_button("Download Chart", buf.getvalue(), f"{y_col}_over_{x_col}_line.png", "image/png")

        else:
            st.warning("Unsupported variable types for this chart.")
    except Exception as e:
        st.error(f"Chart error: {str(e)}")

    st.markdown("## Statistical Testing with Interpretation")

    dep = st.selectbox("Dependent Variable", df.columns)
    indep = st.selectbox("Independent Variable", [c for c in df.columns if c != dep])

    dep_type = detect_type(df[dep])
    indep_type = detect_type(df[indep])

    test_data = df[[dep, indep]].dropna()

    if st.button("Run Statistical Test"):
        try:
            if len(test_data) < 10:
                st.warning("Not enough data for reliable testing (minimum 10 rows).")
            else:
                if dep_type == "Categorical" and indep_type == "Categorical":
                    table = pd.crosstab(test_data[dep], test_data[indep])
                    chi2, p, _, _ = chi2_contingency(table)
                    st.write(f"Chi-square p-value: {p:.4f}")
                    st.success("Significant relationship." if p < 0.05 else "No significant relationship.")

                elif dep_type == "Numeric" and indep_type == "Categorical":
                    groups = [group[dep] for _, group in test_data.groupby(indep)]
                    f_stat, p_val = f_oneway(*groups)
                    st.write(f"ANOVA p-value: {p_val:.4f}")
                    st.success("Significant difference." if p_val < 0.05 else "No significant difference.")

                elif dep_type == "Numeric" and indep_type == "Numeric":
                    model = smf.ols(f"{dep} ~ {indep}", data=test_data).fit()
                    st.text(model.summary())
                    p_val = model.pvalues[1]
                    r2 = model.rsquared
                    if p_val < 0.05:
                        st.success(f"Significant linear relationship (R² = {r2:.2f})")
                    else:
                        st.info(f"No significant relationship (R² = {r2:.2f})")
                else:
                    st.warning("Unsupported variable types for statistical test.")
        except Exception as e:
            st.error(f"Test error: {str(e)}")
