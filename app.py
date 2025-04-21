
import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
from scipy.stats import chi2_contingency, f_oneway
import statsmodels.formula.api as smf
import nltk
from nltk.corpus import stopwords
from collections import Counter
from textblob import TextBlob

nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(page_title="Survey Dashboard", layout="wide")
st.markdown("<h1 style='text-align: center; color: #4b5fb5;'>Evaluation Study Survey Dashboard</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload your Excel or CSV file", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith("csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    st.markdown("## Step 1: Review & Confirm Column Types")
    user_column_types = {}
    type_options = ["Numeric", "Categorical", "Text", "Ordinal"]

    for col in df.columns:
        inferred_type = "Numeric" if pd.api.types.is_numeric_dtype(df[col]) else (
            "Text" if df[col].nunique() > 30 else "Categorical"
        )
        user_column_types[col] = st.selectbox(f"{col} - Detected: {inferred_type}", type_options, index=type_options.index(inferred_type))

    st.markdown("## Step 2: Dashboard Insights")

    selected_column = st.selectbox("Select a column to visualize", df.columns)
    col_type = user_column_types[selected_column]

    if col_type == "Categorical":
        fig = px.bar(df[selected_column].value_counts().reset_index(), x="index", y=selected_column,
                     labels={"index": selected_column, selected_column: "Count"},
                     title=f"Distribution of {selected_column}", color="index")
        st.plotly_chart(fig, use_container_width=True)

    elif col_type == "Numeric":
        fig = px.histogram(df, x=selected_column, nbins=20, title=f"Distribution of {selected_column}", marginal="box")
        st.plotly_chart(fig, use_container_width=True)

    elif col_type == "Text":
        st.subheader("Word Cloud")
        text = " ".join(df[selected_column].dropna().astype(str))
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        st.image(wordcloud.to_array(), use_column_width=True)

        st.subheader("Summary")
        if st.button("Generate Text Summary"):
            words = nltk.word_tokenize(text.lower())
            stop_words = set(stopwords.words('english'))
            words = [word for word in words if word.isalpha() and word not in stop_words]
            most_common = Counter(words).most_common(10)
            keywords = ", ".join([w for w, _ in most_common])
            blob = TextBlob(text)
            sentiment = blob.sentiment.polarity
            sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

            st.markdown(f"**Most Mentioned Keywords:** {keywords}")
            st.markdown(f"**Overall Sentiment:** {sentiment_label}")
            summary_sentences = blob.sentences[:3]
            summary_text = " ".join(str(s) for s in summary_sentences)
            st.markdown(f"**Auto-Generated Summary:** {summary_text}")

    st.markdown("## Step 3: Statistical Tests")

    cat_cols = [col for col, typ in user_column_types.items() if typ == "Categorical"]
    num_cols = [col for col, typ in user_column_types.items() if typ == "Numeric"]

    if len(cat_cols) >= 2:
        st.subheader("Chi-Square Test")
        chi1 = st.selectbox("Categorical Variable 1", cat_cols, key="chi1")
        chi2 = st.selectbox("Categorical Variable 2", cat_cols, key="chi2")
        if st.button("Run Chi-Square"):
            chi_table = pd.crosstab(df[chi1], df[chi2])
            chi2_stat, p_val, _, _ = chi2_contingency(chi_table)
            st.write(f"Chi-Square Test p-value: {p_val:.4f}")
            st.dataframe(chi_table)

    if len(cat_cols) >= 1 and len(num_cols) >= 1:
        st.subheader("ANOVA Test")
        group_col = st.selectbox("Group (Categorical)", cat_cols, key="anova_group")
        value_col = st.selectbox("Value (Numeric)", num_cols, key="anova_val")
        if st.button("Run ANOVA"):
            grouped = [group[value_col].dropna() for name, group in df.groupby(group_col)]
            f_stat, p_val = f_oneway(*grouped)
            st.write(f"ANOVA Test p-value: {p_val:.4f}")

    if len(num_cols) >= 2:
        st.subheader("Linear Regression")
        y = st.selectbox("Dependent Variable (Numeric)", num_cols, key="reg_y")
        x = st.selectbox("Independent Variable", df.columns, key="reg_x")
        if st.button("Run Regression"):
            x_type = user_column_types[x]
            formula = f"{y} ~ C({x})" if x_type == "Categorical" else f"{y} ~ {x}"
            model = smf.ols(formula=formula, data=df).fit()
            st.text(model.summary())
