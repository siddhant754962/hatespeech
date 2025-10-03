# -------------------------------
# Hate Speech Recognition App
# Streamlit + TF-IDF + ML Model
# -------------------------------

import streamlit as st
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import pandas as pd
import math
from datetime import datetime
import altair as alt

# --- Page Configuration (MUST be the first Streamlit command) ---
st.set_page_config(page_title="Hate Speech Recognition", layout="wide", initial_sidebar_state="collapsed")

# --- Initialize Session State ---
if 'result' not in st.session_state:
    st.session_state.result = None
if 'user_input' not in st.session_state:
    st.session_state.user_input = ""
if 'example_selector' not in st.session_state:
    st.session_state.example_selector = "Select an example..."


# --- Download NLTK resources (done once) ---
@st.cache_resource
def load_nltk_data():
    try:
        stopwords.words("english")
    except LookupError:
        nltk.download('stopwords')
    try:
        WordNetLemmatizer().lemmatize("test")
    except LookupError:
        nltk.download('wordnet')
load_nltk_data()

# --- Load Model and Vectorizer ---
@st.cache_resource
def load_model_and_vectorizer():
    try:
        # IMPORTANT: Make sure these paths are correct for your system
        model = joblib.load("best_model.pkl")
        vectorizer = joblib.load("tfidf_vectorizer.pkl")
        return model, vectorizer
    except FileNotFoundError:
        st.error("Model or vectorizer file not found. Please check the file paths in the script.")
        st.stop()
model, vectorizer = load_model_and_vectorizer()

# --- FEATURE FUNCTIONS ---

# --- Preprocessing & Keyword Highlighting ---
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
TOXIC_KEYWORDS = [
    'idiot', 'stupid', 'loser', 'garbage', 'hell', 'pathetic', 'worthless',
    'clown', 'trash', 'parasites', 'disease', 'criminals', 'crap', 'bitch',
    'asshole', 'fuck', 'shit', 'kill', 'die', 'hate'
]
LEMMATIZED_KEYWORDS = {lemmatizer.lemmatize(word) for word in TOXIC_KEYWORDS}

@st.cache_data
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return " ".join(tokens)

def highlight_words(text, prediction):
    if prediction in [0, 1]: # Only highlight for Hate or Offensive
        highlight_color = "#ef4444" if prediction == 0 else "#f97316"
        def replacer(match):
            word = match.group(0)
            clean_word = re.sub(r'[^\w\s]', '', word).lower()
            if lemmatizer.lemmatize(clean_word) in LEMMATIZED_KEYWORDS:
                return f"<span class='highlight' style='--highlight-color: {highlight_color};'>{word}</span>"
            return word
        pattern = re.compile(r'\b\w+\b', re.IGNORECASE)
        return pattern.sub(replacer, text)
    return text

# --- Text Metrics ---
@st.cache_data
def calculate_readability(text):
    words = text.split()
    word_count = len(words)
    if word_count == 0: return 0, 0
    sentence_count = max(1, len(re.split(r'[.!?]+', text)))
    syllable_count = 0
    for word in words:
        word = word.lower()
        count = len(re.findall(r'[aeiouy]+', word))
        if word.endswith('e'):
            count -= 1
        if word.endswith('le') and len(word) > 2 and word[-3] not in 'aeiouy':
            count += 1
        if count == 0:
            count = 1
        syllable_count += count
    
    if word_count > 0 and sentence_count > 0:
        score = 206.835 - 1.015 * (word_count / sentence_count) - 84.6 * (syllable_count / word_count)
        grade = round(0.39 * (word_count / sentence_count) + 11.8 * (syllable_count / word_count) - 15.59)
        return max(0, score), max(1, grade)
    return 0, 0


# --- Simulated API Calls for Advanced Analysis ---
@st.cache_data
def simulate_advanced_analysis(probabilities):
    hate_prob, off_prob, neu_prob = probabilities[0], probabilities[1], probabilities[2]
    
    if hate_prob > 0.6 or off_prob > 0.6:
        sentiment = "Negative"
        sentiment_score = max(hate_prob, off_prob)
    elif neu_prob > 0.6:
        sentiment = "Neutral"
        sentiment_score = neu_prob
    else:
        sentiment = "Mixed"
        sentiment_score = 1 - neu_prob
        
    emotions = {
        'Anger': hate_prob + off_prob * 0.5,
        'Disgust': hate_prob * 0.6 + off_prob * 0.4,
        'Fear': hate_prob * 0.4,
        'Joy': neu_prob * 0.7,
        'Sadness': off_prob * 0.3,
        'Surprise': max(0, (1 - sum(probabilities)) * 0.5)
    }
    total_emotion = sum(emotions.values())
    if total_emotion > 0:
        emotions = {k: v / total_emotion for k, v in emotions.items()}
    
    return sentiment, sentiment_score, emotions

# --- Report Generation ---
def generate_report(analysis_data):
    report = f"""
HATE SPEECH ANALYSIS REPORT
===========================
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

--- INPUT TEXT ---
{analysis_data['user_input']}

--- PRIMARY CLASSIFICATION ---
Category: {analysis_data['label']}
Confidence: {analysis_data['confidence']:.1%}

--- DETAILED ANALYSIS ---
Overall Sentiment: {analysis_data['sentiment']} ({analysis_data['sentiment_score']:.1%})
Readability (Flesch Score): {analysis_data['readability_score']:.1f}
Readability (Grade Level): {analysis_data['readability_grade']}

--- TOXICITY BREAKDOWN ---
Hate Speech: {analysis_data['probabilities'][0]:.1%}
Offensive:   {analysis_data['probabilities'][1]:.1%}
Neutral:     {analysis_data['probabilities'][2]:.1%}

--- EMOTIONAL TONE ---
"""
    for emotion, value in analysis_data['emotions'].items():
        report += f"{emotion:<10}: {value:.1%}\n"
    
    return report

# --- CUSTOM CSS ---
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;900&display=swap" rel="stylesheet">
    <style>
        /* --- General & Background --- */
        .stApp {
            background-color: #0d1117;
            background-image: url('https://www.transparenttextures.com/patterns/cubes.png');
            color: #c9d1d9; font-family: 'Poppins', sans-serif;
        }
        .stApp::before {
            content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
            background: radial-gradient(circle at 20% 20%, #1a2a40, transparent 40%),
                        radial-gradient(circle at 80% 80%, #2a1a40, transparent 40%);
            animation: bg-pan 20s linear infinite alternate; z-index: -1;
        }
        @keyframes bg-pan { 0% { background-position: 0% 50%; } 100% { background-position: 100% 50%; } }

        .block-container { max-width: 1600px; padding: 2rem 3rem; }
        
        /* --- Titles & Headers --- */
        h1 {
            font-weight: 900; text-align: center; font-size: 3.5rem; letter-spacing: -2px;
            background: -webkit-linear-gradient(45deg, #38bdf8, #a78bfa, #f472b6);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            margin-bottom: 0;
        }
        .subtitle { text-align: center; color: #8b949e; font-size: 1.1rem; max-width: 650px; margin: 0 auto 2.5rem auto; }
        
        /* --- Cards & Containers --- */
        .main-card {
            background: rgba(22, 27, 34, 0.85); backdrop-filter: blur(12px);
            border: 1px solid #30363d; padding: 2rem 2.5rem; border-radius: 1rem; height: 100%;
        }
        
        /* --- Input Elements --- */
        @keyframes animated-gradient {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .stTextArea {
            border-radius: 0.75rem;
            padding: 3px;
            background: linear-gradient(90deg, #38bdf8, #a78bfa, #f472b6, #38bdf8);
            background-size: 200% 200%;
            animation: animated-gradient 3s ease-in-out infinite;
            transition: all 0.3s ease;
        }

        .stTextArea:focus-within {
            box-shadow: 0 0 15px #a78bfa88;
        }

        .stTextArea textarea {
            background-color: #0d1117;
            border: none;
            color: #c9d1d9;
            border-radius: 0.5rem;
            font-size: 1.1rem;
            min-height: 280px;
            transition: all 0.2s ease;
        }
        
        .stButton>button {
            width: 100%; font-weight: 700; background: #238636; color: white;
            border-radius: 0.5rem; border: 1px solid #33a347; padding: 0.75rem 1.5rem; font-size: 1.1rem;
            transition: all 0.2s ease;
        }
        .stButton>button:hover { background: #2ea043; border-color: #40c459; transform: translateY(-2px); }
        .stDownloadButton>button { width: 100%; border-color: #2f81f7; }

        /* --- Result & Metric Styles --- */
        .result-card { border-radius: 0.75rem; padding: 2rem; color: white; text-align: center; border: 2px solid; margin-bottom: 1.5rem;}
        .result-card-HateSpeech { background-color: #7f1d1d; border-color: #ef4444; }
        .result-card-Offensive { background-color: #7c2d12; border-color: #f97316; }
        .result-card-Neutral { background-color: #14532d; border-color: #22c55e; }
        .result-icon { font-size: 3rem; line-height: 1; }
        .result-label { font-size: 1.8rem; font-weight: 700; margin-top: 0.5rem; }

        .metric-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; }
        .metric-card { background: #0d1117; border: 1px solid #30363d; padding: 1.25rem; border-radius: 0.5rem; text-align: center; }
        .metric-label { font-size: 1.8rem; font-weight: 700; color: #f0f6fc; line-height: 1.2; }
        .metric-value { font-size: 0.9rem; color: #8b949e; }
        
        .highlight { border-radius: 0.25rem; padding: 0.1em 0.3em; background-color: var(--highlight-color); color: white; font-weight: 600;}

        /* --- Tab Styling --- */
        button[data-baseweb="tab"] {
            background-color: transparent !important; font-size: 1rem; font-weight: 600; color: #8b949e;
            border-bottom: 2px solid transparent !important; padding: 0.8rem 1rem !important;
        }
        button[data-baseweb="tab"][aria-selected="true"] {
            color: #c9d1d9 !important; border-bottom: 2px solid #2f81f7 !important;
        }

        /* --- Placeholder Styling --- */
        @keyframes pulse { 0% { opacity: 0.4; } 50% { opacity: 1; } 100% { opacity: 0.4; } }
        .placeholder-content { text-align: center; padding: 4rem 2rem; border: 2px dashed #30363d; border-radius: 1rem; animation: pulse 2.5s infinite ease-in-out;}
        .placeholder-icon { font-size: 4rem; margin-bottom: 1rem; }
        
        /* --- Footer --- */
        footer { text-align: center; margin-top: 3rem; color: #484f58; }
    </style>
""", unsafe_allow_html=True)

# --- UI LAYOUT ---
st.title("Hate Speech Recognition")
st.markdown("<p class='subtitle'>Analyze text to detect hate speech and offensive language using Machine Learning.</p>", unsafe_allow_html=True)

col1, col2 = st.columns([0.5, 0.5], gap="large")

# --- LEFT COLUMN: INPUT ---
with col1:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.markdown("#### Input Text for Analysis")
    user_input = st.text_area("Input Text", st.session_state.user_input, key="text_input", label_visibility="collapsed")
    analyze_button = st.button("🚀 Run Full Analysis")
    
    st.markdown("<hr style='margin: 1.5rem 0; border-color: #30363d;'>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        example_options = {
            "Select an example...": "",
            "Hate Speech": "All immigrants are criminals and parasites, they are a disease and should be kicked out of the country.",
            "Offensive Language": "What the hell is wrong with you, you stupid idiot? You are such a pathetic, worthless loser. Your opinion is complete garbage.",
            "Neutral Statement": "The new park downtown is scheduled to open next month and will feature a playground and several walking trails."
        }
        def set_example_text():
            st.session_state.user_input = example_options[st.session_state.example_selector]
        st.selectbox("Load an Example:", options=example_options.keys(), key="example_selector", on_change=set_example_text)
    
    with c2:
        with st.expander("ℹ️ About this Tool"):
            st.markdown("""
            This application uses a Machine Learning model to classify content into three categories:
            - **Hate Speech:** Content that attacks or demeans a group based on attributes like race, religion, or origin.
            - **Offensive Language:** Vulgar, profane, or insulting content that is not targeted at a protected group.
            - **Neutral:** Content that is neither hateful nor offensive.

            Additional analysis provides insights into sentiment, emotional tone, and readability.
            """)

    st.markdown('</div>', unsafe_allow_html=True)

# --- ANALYSIS LOGIC ---
if analyze_button:
    st.session_state.user_input = user_input
    if user_input.strip() == "":
        st.toast("⚠️ Please provide text for analysis.", icon="✍️")
        st.session_state.result = None
    else:
        with st.spinner('Performing multi-vector analysis...'):
            cleaned_input = clean_text(user_input)
            vectorized_input = vectorizer.transform([cleaned_input])
            probabilities = model.predict_proba(vectorized_input)[0]
            st.session_state.result = probabilities
            st.rerun()

# --- RIGHT COLUMN: RESULTS ---
with col2:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    if st.session_state.result is not None:
        probabilities = st.session_state.result
        prediction = probabilities.argmax()
        confidence = probabilities[prediction]
        labels = {0: "Hate Speech", 1: "Offensive", 2: "Neutral"}
        icons = {0: "🔥", 1: "⚡", 2: "✅"}
        
        # Run advanced analysis
        readability_score, readability_grade = calculate_readability(st.session_state.user_input)
        sentiment, sentiment_score, emotions = simulate_advanced_analysis(probabilities)
        
        # --- Create Tabs ---
        tab1, tab2, tab3, tab4 = st.tabs(["Summary", "Tone & Emotion", "Text Insights", "Export"])

        with tab1:
            # --- Display Primary Result ---
            result_label = labels[prediction]
            st.markdown(f"""
            <div class="result-card result-card-{result_label.replace(' ', '')}">
                <div class="result-icon">{icons[prediction]}</div>
                <div class="result-label">{result_label}</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(confidence, text=f"{confidence:.0%} Confidence")

            # --- Detailed Metrics Grid ---
            st.markdown("<div class='metric-grid'>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><div class='metric-label'>{sentiment}</div><div class='metric-value'>Sentiment</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><div class='metric-label'>Grade {readability_grade}</div><div class='metric-value'>Readability</div></div>", unsafe_allow_html=True)
            word_count = len(st.session_state.user_input.split())
            read_time = math.ceil(word_count / 200)
            st.markdown(f"<div class='metric-card'><div class='metric-label'>{word_count}</div><div class='metric-value'>Word Count</div></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='metric-card'><div class='metric-label'>~{read_time} min</div><div class='metric-value'>Read Time</div></div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with tab2:
             # --- Emotion Chart ---
            st.markdown("<h5 style='text-align:center; margin-top: 1rem;'>Emotional Tone</h5>", unsafe_allow_html=True)
            emotion_df = pd.DataFrame(emotions.items(), columns=['Emotion', 'Score']).sort_values('Emotion')
            
            chart = alt.Chart(emotion_df).mark_bar(
                cornerRadiusTopLeft=5,
                cornerRadiusTopRight=5
            ).encode(
                x=alt.X('Emotion:N', sort=None, title=None, axis=alt.Axis(labelAngle=0)),
                y=alt.Y('Score:Q', title=None, axis=alt.Axis(format='%')),
                color=alt.Color('Emotion:N', scale=alt.Scale(scheme='spectral'), legend=None),
                tooltip=[alt.Tooltip('Emotion:N'), alt.Tooltip('Score:Q', format='.1%')]
            ).properties(
                height=300
            )
            st.altair_chart(chart, use_container_width=True)


        with tab3:
            # --- Highlighted Text ---
            highlighted_output = highlight_words(st.session_state.user_input, prediction)
            st.markdown("<h5 style='margin-top: 1rem;'>Toxicity Highlights</h5>", unsafe_allow_html=True)
            st.markdown(f"<div style='background:#0d1117; padding: 1rem; border-radius:0.5rem; border:1px solid #30363d; min-height: 150px; max-height: 250px; overflow-y:auto;'>{highlighted_output}</div>", unsafe_allow_html=True)
            st.markdown(f"**Readability Score (Flesch):** `{readability_score:.1f}` (Higher is easier)")

        with tab4:
            st.markdown("<h5 style='margin-top: 1rem;'>Download Report</h5>", unsafe_allow_html=True)
            st.info("Download a plain text file containing the full analysis results.", icon="📄")
            analysis_data = {
                'user_input': st.session_state.user_input, 'label': labels[prediction], 'confidence': confidence,
                'sentiment': sentiment, 'sentiment_score': sentiment_score, 'readability_score': readability_score,
                'readability_grade': readability_grade, 'probabilities': probabilities, 'emotions': emotions
            }
            st.download_button(
                label="Download Full Report (.txt)",
                data=generate_report(analysis_data),
                file_name=f"analysis_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

    else:
        st.markdown("""
        <div class="placeholder-content">
            <div class="placeholder-icon">🔬</div>
            <h3>Awaiting Analysis</h3>
            <p style="color: #8b949e;">Enter text in the panel on the left and click 'Run Full Analysis' to see the results here.</p>
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("<footer>Made with ❤️ using Streamlit & Python by siddhant</footer>", unsafe_allow_html=True)
