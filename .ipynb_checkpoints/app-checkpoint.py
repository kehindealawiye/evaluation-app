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

st.set_page_config(page_title="Full Survey Analysis Suite", layout="wide")
st.title("Unified Evaluation Study Survey Analysis Platform")

uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["csv", "xlsx"])

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
    st.subheader("Raw Data")
    st.dataframe(df.head())

    st.markdown("## Dashboard Insights (Multi-Charts Per Column)")
    for col in df.columns:
        try:
            df[col] = pd.to_datetime(df[col])
            col_type = "Datetime"
        except:
            if pd.api.types.is_numeric_dtype(df[col]):
                col_type = "Numeric"
            elif pd.api.types.is_string_dtype(df[col]):
                unique_vals = df[col].nunique()
                col_type = "Categorical" if unique_vals < 30 else "Text"
            else:
                col_type = "Unknown"

        st.markdown(f"### Charts for: {col} ({col_type})")
        try:
            if col_type == "Categorical":
                counts = df[col].value_counts().reset_index()
                counts.columns = [col, "Count"]
                st.plotly_chart(px.bar(counts, x=col, y="Count", color=col, title=f"Bar Chart - {col}"), use_container_width=True)
                st.plotly_chart(px.pie(counts, names=col, values="Count", title=f"Pie Chart - {col}"), use_container_width=True)
            elif col_type == "Numeric":
                st.plotly_chart(px.histogram(df, x=col, nbins=20, title=f"Histogram - {col}"), use_container_width=True)
                st.plotly_chart(px.box(df, y=col, title=f"Boxplot - {col}"), use_container_width=True)
            elif col_type == "Datetime":
                df_sorted = df.sort_values(by=col)
                for num_col in df.select_dtypes(include="number").columns:
                    st.plotly_chart(px.line(df_sorted, x=col, y=num_col, title=f"Trend of {num_col} over {col}"), use_container_width=True)
            elif col_type == "Text":
                text = " ".join(df[col].dropna().astype(str))
                if len(text.split()) > 10:
                    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
                    fig, ax = plt.subplots()
                    ax.imshow(wordcloud, interpolation='bilinear')
                    ax.axis("off")
                    st.pyplot(fig)
                else:
                    st.info("Not enough text for word cloud.")
        except Exception as e:
            st.warning(f"Could not display chart for {col}: {str(e)}")

    st.markdown("## X–Y Variable Chart Explorer")
    x_col = st.selectbox("X variable", df.columns, key="x")
    y_col = st.selectbox("Y variable", [col for col in df.columns if col != x_col], key="y")
    try:
        x_type = detect_type(df[x_col])
        y_type = detect_type(df[y_col])
        if x_type == "Numeric" and y_type == "Numeric":
            st.plotly_chart(px.scatter(df, x=x_col, y=y_col, title=f"Scatter: {x_col} vs {y_col}"), use_container_width=True)
        elif x_type == "Categorical" and y_type == "Numeric":
            st.plotly_chart(px.box(df, x=x_col, y=y_col, title=f"Boxplot: {y_col} by {x_col}"), use_container_width=True)
        elif x_type == "Datetime" and y_type == "Numeric":
            st.plotly_chart(px.line(df.sort_values(x_col), x=x_col, y=y_col, title=f"Trend: {y_col} over {x_col}"), use_container_width=True)
        else:
            st.warning("Unsupported variable combination for visualization.")
    except Exception as e:
        st.warning(f"Chart error: {str(e)}")

    st.markdown("## Statistical Tests & Interpretation")
    dep = st.selectbox("Dependent Variable", df.columns)
    indep = st.selectbox("Independent Variable", [col for col in df.columns if col != dep])
    dep_type = detect_type(df[dep])
    indep_type = detect_type(df[indep])
    st.markdown(f"**H₀:** No relationship between {indep} and {dep}")
    st.markdown(f"**H₁:** Significant relationship between {indep} and {dep}")
    if st.button("Run Test"):
        try:
            if dep_type == "Categorical" and indep_type == "Categorical":
                table = pd.crosstab(df[dep], df[indep])
                chi2, p, _, _ = chi2_contingency(table)
                st.write(f"Chi-Square p-value: {p:.4f}")
                st.success("Significant!" if p < 0.05 else "Not significant.")
            elif dep_type == "Numeric" and indep_type == "Categorical":
                groups = [g[dep].dropna() for _, g in df.groupby(indep)]
                f_stat, p_val = f_oneway(*groups)
                st.write(f"ANOVA p-value: {p_val:.4f}")
                st.success("Significant difference." if p_val < 0.05 else "No significant difference.")
            elif dep_type == "Numeric" and indep_type == "Numeric":
                model = smf.ols(f"{dep} ~ {indep}", data=df).fit()
                p_val = model.pvalues[1]
                st.text(model.summary())
                st.success(f"Significant relationship (R²={model.rsquared:.2f})" if p_val < 0.05 else f"No significant relationship (R²={model.rsquared:.2f})")
            else:
                st.warning("Unsupported variable types.")
        except Exception as e:
            st.error(f"Test failed: {str(e)}")

    st.markdown("## Descriptive Statistics & Correlation")
    numeric_cols = df.select_dtypes(include=np.number).columns
    if not numeric_cols.empty:
        st.dataframe(df[numeric_cols].describe().T)
        if len(numeric_cols) >= 2:
            corr = df[numeric_cols].corr()
            fig, ax = plt.subplots(figsize=(10, 5))
            sns.heatmap(corr, annot=True, cmap="Blues", fmt=".2f", ax=ax)
            st.pyplot(fig)

    st.markdown("## Export Tools")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Data as CSV", csv, file_name="survey_data.csv", mime="text/csv")

    if not numeric_cols.empty:
        chart_col = st.selectbox("Choose numeric column to export histogram", numeric_cols, key="hist_chart")
        fig, ax = plt.subplots()
        sns.histplot(df[chart_col], kde=True, ax=ax)
        ax.set_title(f"Histogram of {chart_col}")
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        st.pyplot(fig)
        st.download_button("Download Chart as PNG", buf.getvalue(), file_name=f"{chart_col}_chart.png", mime="image/png")
        buf.close()
