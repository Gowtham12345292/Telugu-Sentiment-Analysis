import streamlit as st
from transformers import pipeline

# Page config
st.set_page_config(
    page_title="Telugu Sentiment Analysis",
    page_icon="📝",
    layout="centered"
)

# Load model (cached so it loads only once)
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="Gowthamvemula/Teugu_Sentimental_fine-tuning"
    )

# Label mapping
LABELS = {
    "LABEL_0": {"name": "Neutral", "emoji": "😐"},
    "LABEL_1": {"name": "Positive", "emoji": "😊"},
    "LABEL_2": {"name": "Negative", "emoji": "😞"}
}

# App title
st.title("📝 Telugu Sentiment Analysis")
st.markdown("Fine-tuned **mBERT** model for Telugu text sentiment classification")
st.markdown("---")

# Load model
pipe = load_model()

# Text input
text = st.text_area(
    "Enter Telugu text:",
    height=120,
    placeholder="ఈ సినిమా చాలా బాగుంది..."
)

# Predict button
if st.button("Analyze Sentiment", type="primary"):
    if text.strip():
        with st.spinner("Analyzing..."):
            result = pipe(text)
            label = result[0]["label"]
            score = result[0]["score"]

            sentiment = LABELS.get(label, {"name": "Unknown", "emoji": "❓"})

            # Display result
            st.markdown("---")
            col1, col2 = st.columns(2)

            with col1:
                st.metric("Sentiment", f"{sentiment['emoji']} {sentiment['name']}")
            with col2:
                st.metric("Confidence", f"{score:.2%}")

            # Color-coded result
            if sentiment["name"] == "Positive":
                st.success(f"The text is **Positive** with {score:.2%} confidence")
            elif sentiment["name"] == "Negative":
                st.error(f"The text is **Negative** with {score:.2%} confidence")
            else:
                st.info(f"The text is **Neutral** with {score:.2%} confidence")
    else:
        st.warning("Please enter some Telugu text to analyze")

# Example texts
st.markdown("---")
st.markdown("### Try these examples:")

examples = [
    ("ఈ సినిమా చాలా బాగుంది", "This movie is very good"),
    ("ఈ ఆహారం చాలా చెడుగా ఉంది", "This food is very bad"),
    ("నాకు ఈ రోజు చాలా సంతోషంగా ఉంది", "I am very happy today"),
    ("నేను ఈ వార్తలకు చాలా బాధపడ్డాను", "I felt very sad for this news"),
    ("ఈ వాతావరణం చాలా అద్భుతంగా ఉంది", "This weather is very wonderful"),
]

for telugu, english in examples:
    if st.button(f"{telugu} ({english})", key=telugu):
        with st.spinner("Analyzing..."):
            result = pipe(telugu)
            label = result[0]["label"]
            score = result[0]["score"]
            sentiment = LABELS.get(label, {"name": "Unknown", "emoji": "❓"})
            st.write(f"**Result:** {sentiment['emoji']} {sentiment['name']} ({score:.2%})")

# Footer
st.markdown("---")
st.markdown(
    "Built by **Vemula Gowtham** | "
    "[GitHub](https://github.com/Gowtham12345292) | "
    "[LinkedIn](https://linkedin.com/in/vemula-gowtham-624206286)"
)
