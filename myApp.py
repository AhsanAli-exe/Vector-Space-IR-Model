import nltk
import os

# Permanent NLTK data solution for Streamlit Cloud
nltk_data_path = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_path, exist_ok=True)
nltk.data.path.append(nltk_data_path)

# Download required NLTK data if missing
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', download_dir=nltk_data_path)
    nltk.download('punkt_tab', download_dir=nltk_data_path)  # Add this line
    nltk.download('wordnet', download_dir=nltk_data_path)   # For lemmatization
import streamlit as st
from computeSimilarity import rankDocs, loadTfidfData

st.set_page_config(page_title="VSM IR System", layout="centered",page_icon="🔍")

# Custom CSS for clean UI
st.markdown("""
<style>
    /* Remove all white space above */
    .stApp {
        padding-top: 1rem;
    }
    .search-box {
        border-radius: 8px;
        background: #f0f2f6;
        margin-bottom: 1rem;
    }
    .alpha-box{
            margin-bottom: 1rem;
    }
    .result-card {
        padding: 10px 15px;
        margin-bottom: 8px;
        border-radius: 6px;
        background: #ffffff;
        border-left: 4px solid #1e88e5;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .result-card h3 {
        color: #333333;
        margin: 0;
        font-size: 16px;
    }
    
    .score {
        color: #1e88e5;
        font-weight: bold;
        font-size: 14px;
    }
    
    /* Document content expander */
    .stExpander {
        margin-top: 5px;
    }
    
</style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def loadData():
    return loadTfidfData()

def docContent(doc_id):
    try:
        with open(f"Abstracts/{doc_id}.txt","r") as f:
            return f.read()
    except:
        return "Document not found in the Abstracts directory"


st.title("_Vector_ _Space_ :blue[Model] :sunglasses:")
st.header("Developed by: :blue[Ahsan Ali]")
with st.sidebar:
    st.title("[Ahsan Ali's LinkedIn](https://www.linkedin.com/in/ahsan--ali)")
    st.sidebar.title("ℹ️ How to Use")
    st.markdown("## Instructions")
    st.sidebar.markdown("""
    - **Phrase Queries:**
    1. Write any phrase.
    2. Only **alphabets** allowed.
    """)
    st.sidebar.write("💡 Try queries like:")
    st.sidebar.code("weak heuristic")
    st.sidebar.code("ensemble")

with st.form("searchForm"):
    with st.container():
        st.markdown('<div class="search-box">',unsafe_allow_html=True)
        query = st.text_input(
            "🔍 Search scientific abstracts",
            placeholder="Enter query...",
            label_visibility="collapsed"
        )
        st.markdown('</div>',unsafe_allow_html=True)

        with st.container():
            st.markdown('<div class="alpha-box">', unsafe_allow_html=True)
            alpha_value = st.number_input(
                "Similarity threshold (alpha)",
                min_value=0.0,
                max_value=1.0,
                value=0.001,
                step=0.001,
                format="%.3f",
                help="Set the minimum similarity score for documents to be included"
            )
            st.markdown('</div>', unsafe_allow_html=True)
        submitted = st.form_submit_button("Search")
        

if submitted:
    if not query or query.strip() == "":
        st.error("Please enter a search query")
    else:
        with st.spinner(f"Searching for '{query}'..."):
            term_list,idf,tfidf_vectors = loadData()
            rankedDocs = rankDocs(
                query,
                term_list,
                idf,
                tfidf_vectors,
                alpha=alpha_value,
            )
        st.subheader(f"Results for: '{query}'")
        if not rankedDocs:
            st.warning("No matching documents found")
        else:
            for rank,(doc_id,score) in enumerate(rankedDocs,1):
                with st.container():
                    st.markdown(f"""
                        <div class="result-card">
                            <h3>#{rank}: Document {doc_id} <span class="score">(Score: {score:.4f})</span></h3>
                        </div>
                    """,unsafe_allow_html=True)
                    content = docContent(doc_id)
                    if "not found" not in content.lower():
                        with st.expander("View Document Content",expanded=False):
                            st.text_area("",value=content,height=200, 
                                        key=f"content_{doc_id}",label_visibility="collapsed")
                    else:
                        st.warning(f"Document {doc_id} not found in the Abstracts directory.")
