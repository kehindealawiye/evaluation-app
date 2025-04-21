
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

st.set_page_config(page_title="All-in-One Survey Analyzer", layout="wide")
st.title("Final All-in-One Survey Analysis Platform")

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

    st.subheader("Raw Data")
    st.dataframe(df.head())

    column_types = {col: detect_type(df[col]) for col in df.columns}

    st.markdown("## Multi-Chart Dashboard Per Column")
    for col in df.columns:
        try:
            col_type = column_types[col]
            st.markdown(f"### {col} ({col_type})")
            if col_type == "Categorical":
                counts = df[col].value_counts().reset_index()
                counts.columns = [col, "Count"]
                st.plotly_chart(px.bar(counts, x=col, y="Count", color=col, title=f"Bar Chart - {col}"))
                st.plotly_chart(px.pie(counts, names=col, values="Count", title=f"Pie Chart - {col}"))
            elif col_type == "Numeric":
                st.plotly_chart(px.histogram(df, x=col, nbins=20, title=f"Histogram - {col}"))
                st.plotly_chart(px.box(df, y=col, title=f"Boxplot - {col}"))
            elif col_type == "Datetime":
                for num_col in df.select_dtypes(include=np.number).columns:
                    fig = px.line(df.sort_values(by=col), x=col, y=num_col, title=f"{num_col} over {col}")
                    st.plotly_chart(fig)
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
            st.warning(f"Could not chart {col}: {str(e)}")

    st.markdown("## X–Y Chart Explorer with Smart Filtering")
    x_options = [c for c, t in column_types.items() if t in ["Numeric", "Categorical", "Datetime"]]
    x_col = st.selectbox("Select X variable", x_options)
    x_type = column_types[x_col]

    if x_type == "Numeric":
        y_options = [c for c in df.columns if c != x_col and column_types[c] == "Numeric"]
    elif x_type in ["Categorical", "Datetime"]:
        y_options = [c for c in df.columns if c != x_col and column_types[c] == "Numeric"]
    else:
        y_options = []

    y_col = st.selectbox("Select Y variable", y_options)
    chart_mode = st.radio("Chart Mode", ["Interactive", "Static"])

    try:
        df_chart = df[[x_col, y_col]].dropna()
        if chart_mode == "Interactive":
            if x_type == "Numeric" and column_types[y_col] == "Numeric":
                st.plotly_chart(px.scatter(df_chart, x=x_col, y=y_col, title=f"{x_col} vs {y_col}"))
            elif x_type == "Categorical":
                st.plotly_chart(px.box(df_chart, x=x_col, y=y_col, title=f"{y_col} by {x_col}"))
            elif x_type == "Datetime":
                df_chart = df_chart.sort_values(x_col)
                st.plotly_chart(px.line(df_chart, x=x_col, y=y_col, title=f"{y_col} over {x_col}"))
        else:
            fig, ax = plt.subplots()
            if x_type == "Numeric":
                sns.scatterplot(data=df_chart, x=x_col, y=y_col, ax=ax)
            elif x_type == "Categorical":
                sns.boxplot(data=df_chart, x=x_col, y=y_col, ax=ax)
            elif x_type == "Datetime":
                ax.plot(df_chart[x_col], df_chart[y_col])
            ax.set_title(f"{y_col} vs {x_col}")
            st.pyplot(fig)
            buf = io.BytesIO()
            fig.savefig(buf, format="png")
            st.download_button("Download Chart", buf.getvalue(), f"{y_col}_vs_{x_col}.png", "image/png")
    except Exception as e:
        st.warning(f"Chart rendering issue: {e}")

    st.markdown("## Statistical Tests with Auto-Clean & Interpret")
    dep = st.selectbox("Dependent Variable", df.columns)
    dep_type = column_types[dep]
    if dep_type == "Numeric":
        indep_options = [c for c in df.columns if c != dep and column_types[c] in ["Categorical", "Numeric"]]
    elif dep_type == "Categorical":
        indep_options = [c for c in df.columns if c != dep and column_types[c] == "Categorical"]
    else:
        indep_options = []

    indep = st.selectbox("Independent Variable", indep_options)
    indep_type = column_types[indep]

    df_test = df[[dep, indep]].dropna()
    if st.button("Run Test"):
        try:
            if dep_type == "Categorical" and indep_type == "Categorical":
                table = pd.crosstab(df_test[dep], df_test[indep])
                chi2, p, _, _ = chi2_contingency(table)
                st.write(f"Chi-square p = {p:.4f}")
                st.success("Significant!" if p < 0.05 else "Not significant.")
            elif dep_type == "Numeric" and indep_type == "Categorical":
                groups = [group[dep] for _, group in df_test.groupby(indep)]
                f_stat, p_val = f_oneway(*groups)
                st.write(f"ANOVA p = {p_val:.4f}")
                st.success("Significant difference!" if p_val < 0.05 else "No significant difference.")
            elif dep_type == "Numeric" and indep_type == "Numeric":
                model = smf.ols(f"{dep} ~ {indep}", data=df_test).fit()
                st.text(model.summary())
                st.success(f"Significant (R² = {model.rsquared:.2f})" if model.pvalues[1] < 0.05 else f"Not significant (R² = {model.rsquared:.2f})")
        except Exception as e:
            st.error(f"Test error: {e}")

    st.markdown("## Export Tools")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Data as CSV", csv, "survey_data.csv", "text/csv")
