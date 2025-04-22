
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from scipy.stats import chi2_contingency, f_oneway
import statsmodels.formula.api as smf
from wordcloud import WordCloud
import io

st.set_page_config(page_title="Complete Survey Analysis Tool", layout="wide")
st.title("Survey Analysis Dashboard – Final Complete Version")

uploaded_file = st.file_uploader("Upload Excel or CSV", type=["csv", "xlsx"])

def detect_type(series):
    try:
        pd.to_datetime(series)
        return "Datetime"
    except:
        if pd.api.types.is_numeric_dtype(series):
            return "Numeric"
        elif pd.api.types.is_string_dtype(series):
            unique_vals = series.nunique()
            if unique_vals < 10:
                return "Categorical"
            elif unique_vals < 30:
                return "Categorical"
            else:
                return "Text"
        return "Unknown"

if uploaded_file:
    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith("csv") else pd.read_excel(uploaded_file)

    # Type detection
    column_types = {col: detect_type(df[col]) for col in df.columns}

    # Show variable summary
    st.markdown("## Detected Variable Types")
    var_df = pd.DataFrame.from_dict(column_types, orient="index", columns=["Type"])
    st.dataframe(var_df)

    chart_mode = st.radio("Chart display mode", ["Interactive", "Static"], horizontal=True)

    st.markdown("## Multi-Chart Dashboard")
    for col in df.columns:
        col_type = column_types[col]
        st.markdown(f"### {col} ({col_type})")
        try:
            if col_type == "Categorical":
                counts = df[col].value_counts().reset_index()
                counts.columns = [col, "Count"]
                if chart_mode == "Interactive":
                    st.plotly_chart(px.bar(counts, x=col, y="Count", color=col))
                    st.plotly_chart(px.pie(counts, names=col, values="Count"))
                else:
                    fig, ax = plt.subplots()
                    sns.barplot(data=counts, x=col, y="Count", ax=ax)
                    st.pyplot(fig)

            elif col_type == "Numeric":
                if chart_mode == "Interactive":
                    st.plotly_chart(px.histogram(df, x=col))
                    st.plotly_chart(px.box(df, y=col))
                else:
                    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
                    sns.histplot(df[col], kde=True, ax=axs[0])
                    axs[0].set_title("Histogram")
                    sns.boxplot(y=df[col], ax=axs[1])
                    axs[1].set_title("Boxplot")
                    st.pyplot(fig)

            elif col_type == "Text":
                text = " ".join(df[col].dropna().astype(str))
                if len(text.split()) > 10:
                    wc = WordCloud(width=800, height=400).generate(text)
                    fig, ax = plt.subplots()
                    ax.imshow(wc, interpolation="bilinear")
                    ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.info("Not enough content for word cloud.")
        except Exception as e:
            st.warning(f"Chart error: {e}")

    st.markdown("## X–Y Chart Explorer")
    x_options = [col for col in df.columns if column_types[col] in ["Numeric", "Categorical", "Datetime"]]
    x_col = st.selectbox("X Variable", x_options)
    x_type = column_types[x_col]

    if x_type == "Numeric":
        y_options = [col for col in df.columns if col != x_col and column_types[col] == "Numeric"]
    elif x_type in ["Categorical", "Datetime"]:
        y_options = [col for col in df.columns if col != x_col and column_types[col] == "Numeric"]
    else:
        y_options = []

    if not y_options:
        st.warning("No compatible Y variables found. Try selecting a different X (e.g., 'price', 'revenue').")
    else:
        y_col = st.selectbox("Y Variable", y_options)
        df_chart = df[[x_col, y_col]].dropna()
        try:
            if chart_mode == "Interactive":
                if x_type == "Numeric":
                    st.plotly_chart(px.scatter(df_chart, x=x_col, y=y_col))
                elif x_type == "Categorical":
                    st.plotly_chart(px.box(df_chart, x=x_col, y=y_col))
                elif x_type == "Datetime":
                    st.plotly_chart(px.line(df_chart.sort_values(x_col), x=x_col, y=y_col))
            else:
                fig, ax = plt.subplots()
                if x_type == "Numeric":
                    sns.scatterplot(data=df_chart, x=x_col, y=y_col, ax=ax)
                elif x_type == "Categorical":
                    sns.boxplot(data=df_chart, x=x_col, y=y_col, ax=ax)
                elif x_type == "Datetime":
                    ax.plot(df_chart[x_col], df_chart[y_col])
                st.pyplot(fig)
        except Exception as e:
            st.warning(f"Rendering failed: {e}")

    st.markdown("## Statistical Test Preview and Execution")
    dep = st.selectbox("Dependent Variable", df.columns)
    dep_type = column_types[dep]

    test_hint = ""
    if dep_type == "Numeric":
        indep_options = [col for col in df.columns if col != dep and column_types[col] in ["Categorical", "Numeric"]]
        test_hint = "Test options: ANOVA (if group is categorical), Regression (if numeric)"
    elif dep_type == "Categorical":
        indep_options = [col for col in df.columns if col != dep and column_types[col] == "Categorical"]
        test_hint = "Test option: Chi-square (cat–cat)"
    else:
        indep_options = []

    st.info(test_hint)
    if indep_options:
        indep = st.selectbox("Independent Variable", indep_options)
        df_test = df[[dep, indep]].dropna()

        if st.button("Run Test"):
            try:
                if len(df_test) < 10:
                    st.warning("Too few rows for reliable testing.")
                else:
                    if dep_type == "Categorical" and column_types[indep] == "Categorical":
                        chi_table = pd.crosstab(df_test[dep], df_test[indep])
                        chi2, p, _, _ = chi2_contingency(chi_table)
                        st.write(f"Chi-square p = {p:.4f}")
                        st.success("Significant!" if p < 0.05 else "Not significant.")
                    elif dep_type == "Numeric" and column_types[indep] == "Categorical":
                        groups = [g[dep] for _, g in df_test.groupby(indep)]
                        f_stat, p_val = f_oneway(*groups)
                        st.write(f"ANOVA p = {p_val:.4f}")
                        st.success("Significant!" if p_val < 0.05 else "Not significant.")
                    elif dep_type == "Numeric" and column_types[indep] == "Numeric":
                        model = smf.ols(f"{dep} ~ {indep}", data=df_test).fit()
                        st.text(model.summary())
                        st.success(f"Significant (R² = {model.rsquared:.2f})" if model.pvalues[1] < 0.05 else f"Not significant (R² = {model.rsquared:.2f})")
            except Exception as e:
                st.error(f"Test error: {e}")
    else:
        st.warning("No compatible independent variables available.")

    st.markdown("## Descriptive Statistics")
    num_cols = df.select_dtypes(include=np.number).columns
    if not num_cols.empty:
        st.dataframe(df[num_cols].describe().T)

    st.markdown("## Correlation Matrix")
    if len(num_cols) >= 2:
        corr = df[num_cols].corr()
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        st.pyplot(fig)

    st.markdown("## Export")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Dataset (CSV)", csv, "survey_data.csv", "text/csv")
