import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob

st.set_page_config(layout="wide")
@st.cache_data
def load_data():
    # Financial impact dataset
    fin = pd.read_csv(
        './Data/financial_impact_final.csv',
        parse_dates=['published_date']
    )
    # NYT coverage dataset
    nyt = pd.read_csv(
        './Data/processed_nyt_data_20250417_v3.csv',
        parse_dates=['published_date']
    )

    metrics = pd.read_csv(
        './Data/gdacs_temp2_geotext.csv',
        parse_dates=['pubdate']
    )
    return  fin, nyt, metrics

fin, nyt, metrics = load_data()   


# Sidebar tabs
tab = st.sidebar.radio("Navigation", [ "GDACS metrics" , "NYT Financial Analysis", "NYT Coverage"])

def top_words(text_series, n=10):
    vec = CountVectorizer(stop_words='english', max_features=1000)
    X = vec.fit_transform(text_series.dropna())
    freqs = zip(vec.get_feature_names_out(), X.sum(axis=0).tolist()[0])
    return pd.DataFrame(sorted(freqs, key=lambda x: x[1], reverse=True)[:n], columns=['word','count'])

@st.cache_data
def compute_sentiments(text_series):
    return text_series.dropna().apply(lambda t: TextBlob(t).sentiment.polarity)

if tab == "Financial Analysis":
    st.title("📉 Financial Impact Dashboard")
    disaster = st.sidebar.selectbox('Disaster Type', sorted(fin['disaster_type'].dropna().unique()))
    df_fin = fin[fin['disaster_type'] == disaster]
    metrics = ['%Change Prev→Event','%Change Event→Next','%Change Prev→Next','%Change Prev→NextWeek']
    corr = df_fin[metrics].corr()
    ts = df_fin.groupby(df_fin['published_date'].dt.to_period('M'))[metrics].mean().reset_index()
    ts['date'] = ts['published_date'].dt.to_timestamp()

    fig1 = px.line(ts, x='date', y=metrics, title='Average Stock Impact Metrics Over Time')
    fig2 = px.imshow(corr, text_auto=True, title='Correlation Matrix of Stock Impact Metrics')
    fig3 = px.histogram(df_fin, x='%Change Event→Next', nbins=20, title='Distribution: Event to Next Day % Change')
    top_ticks = df_fin.groupby('Ticker')['%Change Prev→Next'].mean().nlargest(10).reset_index()
    fig4 = px.bar(top_ticks, x='Ticker', y='%Change Prev→Next', title='Top 10 Tickers by % Change (Prev to Next)')

elif tab == "NYT Coverage":
    st.title("🗞️ NYT Disaster Coverage Dashboard")
    dtype = st.sidebar.selectbox('Disaster Type', sorted(nyt['DisasterType'].dropna().unique()))
    df_nyt = nyt[nyt['DisasterType'] == dtype].dropna(subset=['summary'])
    top_sum = top_words(df_nyt['summary'].astype(str))
    sentiments = compute_sentiments(df_nyt['summary'].astype(str))
    n_month = df_nyt.groupby(df_nyt['published_date'].dt.to_period('M')).size().reset_index(name='article_count')
    n_month['date'] = n_month['published_date'].dt.to_timestamp()

    fig1 = px.line(n_month, x='date', y='article_count', title='NYT Article Count Over Time')
    fig2 = px.bar(top_sum, x='word', y='count', title='Top Keywords in NYT Summaries')
    fig3 = px.histogram(x=sentiments, nbins=20, labels={'x':'Sentiment Polarity'}, title='Sentiment Distribution in NYT Summaries')

    # Generate word cloud
    # wordcloud = WordCloud(width=800, height=400, background_color='white').generate(' '.join(df_nyt['summary'].dropna()))
    # fig4, ax = plt.subplots(figsize=(10, 5))
    # ax.imshow(wordcloud, interpolation='bilinear')
    # ax.set_title('Word Cloud of NYT Summaries')
    # ax.axis('off')
    
    # fig4 = px.treemap(top_sum, path=['word'], values='count', 
    #               color='count', title='Treemap of Keywords in NYT Summaries',
    #               color_continuous_scale='Blues')
    
    import spacy
    from collections import Counter

    nlp = spacy.load("en_core_web_sm")

    def extract_entities(text_series, entity_type='GPE'):
        entities = []
        for doc in nlp.pipe(text_series.dropna().astype(str), disable=["tagger", "parser"]):
            entities += [ent.text for ent in doc.ents if ent.label_ == entity_type]
        return pd.DataFrame(Counter(entities).most_common(10), columns=['Entity','Count'])

    top_places = extract_entities(df_nyt['summary'], entity_type='GPE')
    fig4 = px.bar(top_places, x='Entity', y='Count', title='Top Mentioned Locations in NYT Summaries')


else:  # Event Metrics
    st.title("📊 GDACS Event Metrics Dashboard")
    # Filter by DisasterType
    dtype = st.sidebar.selectbox('Disaster Type', sorted(metrics['DisasterType'].dropna().unique()))
    dfm = metrics[metrics['DisasterType']==dtype]
    # Chart 1: Event count over time
    dfm_month = dfm.groupby(dfm['pubdate'].dt.to_period('M')).size().reset_index(name='count')
    dfm_month['date']=dfm_month['pubdate'].dt.to_timestamp()
    fig1 = px.line(dfm_month, x='date', y='count', title='Monthly Event Count')
    # Chart 2: Severity vs Duration scatter
    fig2 = px.scatter(
        dfm, x='Duration_days', y='SeverityScore_pca',
        size='NumPlacesMentioned', color='ClusterID',
        title='Severity vs Duration (size=Places Mentioned)'
    )
    # Chart 3: Distribution of normalized severity
    fig3 = px.histogram(dfm, x='SeverityScore_pca', nbins=30, title='Severity Score Distribution')
    # Chart 4: Bar of top normalized features
    norm_cols = [c for c in dfm.columns if c.endswith('_norm')]
    avg_norm = dfm[norm_cols].mean().reset_index()
    avg_norm.columns=['metric','avg_value']
    fig4 = px.bar(avg_norm, x='metric', y='avg_value', title='Average Normalized Metrics')

# Display the 2x2 layout in a wide format using columns
cols = st.columns(2)
with cols[0]:
    st.plotly_chart(fig1, use_container_width=True)
    st.plotly_chart(fig3, use_container_width=True)
with cols[1]:
    st.plotly_chart(fig2, use_container_width=True)
    # if tab == "NYT Coverage":
    #     st.pyplot(fig4, use_container_width=True)
    # else:
    #     st.plotly_chart(fig4, use_container_width=True)
    st.plotly_chart(fig4, use_container_width=True)
