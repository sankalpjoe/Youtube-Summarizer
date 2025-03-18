import streamlit as st
import os
import re
import time
import nltk
import pandas as pd
import plotly.express as px
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import TextFormatter
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
from transformers import pipeline
from collections import Counter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Set page configuration
st.set_page_config(
    page_title="YouTube Transcript Summarizer",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define the Agent class to manage the workflow
class YoutubeAgent:
    def __init__(self):
        self.video_id = None
        self.video_url = None
        self.transcript = None
        self.transcript_text = None
        self.summary = None
        self.topics = None
        self.sentiment = None
        self.word_count = 0
        self.sentence_count = 0
        self.status = {
            "input_processing": "pending",
            "content_extraction": "pending",
            "analysis": "pending",
            "summarization": "pending"
        }
        self.errors = []
        
    def validate_youtube_url(self, url):
        """Validate YouTube URL and extract video ID."""
        self.status["input_processing"] = "in_progress"
        
        # Common YouTube URL patterns
        patterns = [
            r'(?:https?:\/\/)?(?:www\.)?youtu\.?be(?:\.com)?\/(?:watch\?v=)?([^&\?\/\s]{11})',
            r'(?:https?:\/\/)?(?:www\.)?youtu\.?be(?:\.com)?\/(?:embed\/)?([^&\?\/\s]{11})',
            r'(?:https?:\/\/)?(?:www\.)?youtu\.?be\/([^&\?\/\s]{11})'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                self.video_id = match.group(1)
                self.video_url = url
                self.status["input_processing"] = "complete"
                return True
                
        self.errors.append("Invalid YouTube URL format")
        self.status["input_processing"] = "failed"
        return False
    
    def extract_transcript(self):
        """Extract transcript from YouTube video."""
        if not self.video_id:
            self.errors.append("No valid video ID")
            self.status["content_extraction"] = "failed"
            return False
            
        self.status["content_extraction"] = "in_progress"
        
        try:
            transcript = YouTubeTranscriptApi.get_transcript(self.video_id)
            formatter = TextFormatter()
            self.transcript = transcript
            self.transcript_text = formatter.format_transcript(transcript)
            self.word_count = len(word_tokenize(self.transcript_text))
            self.sentence_count = len(sent_tokenize(self.transcript_text))
            self.status["content_extraction"] = "complete"
            return True
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            self.errors.append(f"Transcript not available: {str(e)}")
            self.status["content_extraction"] = "failed"
            return False
        except Exception as e:
            self.errors.append(f"Error extracting transcript: {str(e)}")
            self.status["content_extraction"] = "failed"
            return False
    
    def analyze_text(self):
        """Analyze transcript text to extract insights."""
        if not self.transcript_text:
            self.errors.append("No transcript available to analyze")
            self.status["analysis"] = "failed"
            return False
            
        self.status["analysis"] = "in_progress"
        
        try:
            # Extract key topics
            stop_words = set(stopwords.words('english'))
            words = word_tokenize(self.transcript_text.lower())
            filtered_words = [word for word in words if word.isalnum() and word not in stop_words]
            
            # Get top 20 words
            word_freq = Counter(filtered_words)
            self.topics = word_freq.most_common(20)
            
            # Basic sentiment analysis
            try:
                sentiment_analyzer = pipeline("sentiment-analysis")
                
                # Split text into chunks for sentiment analysis (max 512 tokens)
                sentences = sent_tokenize(self.transcript_text)
                chunks = []
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) < 500:
                        current_chunk += sentence + " "
                    else:
                        chunks.append(current_chunk)
                        current_chunk = sentence + " "
                
                if current_chunk:
                    chunks.append(current_chunk)
                
                sentiments = []
                for chunk in chunks:
                    if chunk.strip():
                        result = sentiment_analyzer(chunk)
                        sentiments.append(result[0])
                
                # Calculate overall sentiment
                positive_count = sum(1 for item in sentiments if item['label'] == 'POSITIVE')
                negative_count = len(sentiments) - positive_count
                
                self.sentiment = {
                    'positive': positive_count / len(sentiments) if sentiments else 0,
                    'negative': negative_count / len(sentiments) if sentiments else 0,
                    'overall': 'Positive' if positive_count > negative_count else 'Negative'
                }
            except Exception as e:
                st.warning(f"Sentiment analysis could not be completed: {str(e)}")
                self.sentiment = {'overall': 'Unknown'}
            
            self.status["analysis"] = "complete"
            return True
        except Exception as e:
            self.errors.append(f"Error analyzing transcript: {str(e)}")
            self.status["analysis"] = "failed"
            return False
    
    def generate_summary(self, max_length=150, min_length=30):
        """Generate summary from transcript text."""
        if not self.transcript_text:
            self.errors.append("No transcript available to summarize")
            self.status["summarization"] = "failed"
            return False
            
        self.status["summarization"] = "in_progress"
        
        try:
            # Initialize the summarization pipeline
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
            
            # Split text into chunks if it's too long (BART has a limit of 1024 tokens)
            max_chunk_size = 1000  # Characters per chunk (approximate)
            chunks = [self.transcript_text[i:i + max_chunk_size] 
                      for i in range(0, len(self.transcript_text), max_chunk_size)]
            
            summaries = []
            for chunk in chunks:
                # Skip empty chunks
                if not chunk.strip():
                    continue
                    
                # Generate summary for the chunk
                result = summarizer(chunk, max_length=max_length, min_length=min_length, 
                                   do_sample=False)
                if result and len(result) > 0:
                    summaries.append(result[0]['summary_text'])
            
            # Combine summaries
            full_summary = " ".join(summaries)
            
            # If we have multiple chunks, perform a second summarization
            if len(summaries) > 1:
                full_summary = summarizer(full_summary, max_length=max_length, 
                                         min_length=min_length, do_sample=False)[0]['summary_text']
            
            self.summary = full_summary
            self.status["summarization"] = "complete"
            return True
        except Exception as e:
            self.errors.append(f"Error generating summary: {str(e)}")
            self.status["summarization"] = "failed"
            return False
    
    def run_workflow(self, url, max_length=150, min_length=30):
        """Execute the full agentic workflow."""
        # Reset the agent
        self.__init__()
        
        # Step 1: Validate URL
        if not self.validate_youtube_url(url):
            return False
        
        # Step 2: Extract transcript
        if not self.extract_transcript():
            return False
        
        # Step 3: Analyze transcript
        self.analyze_text()  # We continue even if analysis fails
        
        # Step 4: Generate summary
        if not self.generate_summary(max_length, min_length):
            return False
        
        return True

# UI Elements
def main():
    st.title("🎥 YouTube Transcript Summarizer")
    st.markdown("""
    This app uses AI to extract, analyze, and summarize YouTube video transcripts.
    """)
    
    # Initialize session state for the agent
    if 'agent' not in st.session_state:
        st.session_state.agent = YoutubeAgent()
    
    # Sidebar with options
    with st.sidebar:
        st.header("Options")
        
        max_length = st.slider(
            "Maximum summary length",
            min_value=50,
            max_value=500,
            value=150,
            step=10,
            help="Maximum length of the generated summary (in words)"
        )
        
        min_length = st.slider(
            "Minimum summary length",
            min_value=20,
            max_value=100,
            value=30,
            step=5,
            help="Minimum length of the generated summary (in words)"
        )
        
        st.markdown("---")
        st.write("### Additional Analysis")
        
        do_sentiment = st.checkbox("Sentiment Analysis", value=True)
        do_topics = st.checkbox("Topic Extraction", value=True)
        do_visualization = st.checkbox("Generate Visualizations", value=True)
    
    # Main input form
    with st.form("youtube_url_form"):
        st.write("Enter a YouTube video URL:")
        url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
        
        submit_button = st.form_submit_button("Process Video")
        
        if submit_button and url:
            with st.spinner("Processing..."):
                # Run the agent's workflow
                success = st.session_state.agent.run_workflow(url, max_length, min_length)
                
                if not success:
                    st.error("Failed to process the video")
                    for error in st.session_state.agent.errors:
                        st.error(error)
    
    # Display progress
    if 'agent' in st.session_state and any(status != "pending" for status in st.session_state.agent.status.values()):
        st.write("### Processing Status")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.session_state.agent.status["input_processing"] == "complete":
                st.success("URL Validated")
            elif st.session_state.agent.status["input_processing"] == "failed":
                st.error("URL Invalid")
            elif st.session_state.agent.status["input_processing"] == "in_progress":
                st.info("Validating URL...")
        
        with col2:
            if st.session_state.agent.status["content_extraction"] == "complete":
                st.success("Transcript Extracted")
            elif st.session_state.agent.status["content_extraction"] == "failed":
                st.error("Transcript Failed")
            elif st.session_state.agent.status["content_extraction"] == "in_progress":
                st.info("Extracting Transcript...")
        
        with col3:
            if st.session_state.agent.status["analysis"] == "complete":
                st.success("Analysis Complete")
            elif st.session_state.agent.status["analysis"] == "failed":
                st.error("Analysis Failed")
            elif st.session_state.agent.status["analysis"] == "in_progress":
                st.info("Analyzing Content...")
        
        with col4:
            if st.session_state.agent.status["summarization"] == "complete":
                st.success("Summary Generated")
            elif st.session_state.agent.status["summarization"] == "failed":
                st.error("Summary Failed")
            elif st.session_state.agent.status["summarization"] == "in_progress":
                st.info("Generating Summary...")
    
    # Display results
    if 'agent' in st.session_state and st.session_state.agent.summary:
        st.markdown("---")
        st.write("## Results")
        
        # Create tabs for different content
        tabs = st.tabs(["Summary", "Transcript", "Analysis", "Visualizations"])
        
        # Summary tab
        with tabs[0]:
            st.write("### Video Summary")
            st.write(st.session_state.agent.summary)
            
            st.download_button(
                label="Download Summary",
                data=st.session_state.agent.summary,
                file_name="youtube_summary.txt",
                mime="text/plain"
            )
        
        # Transcript tab
        with tabs[1]:
            st.write("### Full Transcript")
            st.text_area("Transcript Text", st.session_state.agent.transcript_text, height=300)
            
            st.write(f"Word count: {st.session_state.agent.word_count}")
            st.write(f"Sentence count: {st.session_state.agent.sentence_count}")
            
            st.download_button(
                label="Download Transcript",
                data=st.session_state.agent.transcript_text,
                file_name="youtube_transcript.txt",
                mime="text/plain"
            )
        
        # Analysis tab
        with tabs[2]:
            if do_sentiment and st.session_state.agent.sentiment:
                st.write("### Sentiment Analysis")
                st.write(f"Overall sentiment: **{st.session_state.agent.sentiment['overall']}**")
                
                if 'positive' in st.session_state.agent.sentiment:
                    # Create sentiment bar chart
                    sentiment_data = {
                        'Sentiment': ['Positive', 'Negative'],
                        'Percentage': [
                            st.session_state.agent.sentiment['positive'] * 100,
                            st.session_state.agent.sentiment['negative'] * 100
                        ]
                    }
                    df_sentiment = pd.DataFrame(sentiment_data)
                    
                    fig = px.bar(
                        df_sentiment, 
                        x='Sentiment', 
                        y='Percentage',
                        color='Sentiment',
                        color_discrete_map={'Positive': 'green', 'Negative': 'red'},
                        labels={'Percentage': 'Percentage (%)'}
                    )
                    st.plotly_chart(fig)
            
            if do_topics and st.session_state.agent.topics:
                st.write("### Key Topics")
                
                # Create a DataFrame for the topics
                topics_data = {
                    'Word': [word for word, count in st.session_state.agent.topics],
                    'Count': [count for word, count in st.session_state.agent.topics]
                }
                df_topics = pd.DataFrame(topics_data)
                
                # Show the topics in a table
                st.dataframe(df_topics)
        
        # Visualizations tab
        with tabs[3]:
            if do_visualization and st.session_state.agent.topics:
                st.write("### Keyword Frequency")
                
                # Create a DataFrame for the topics
                topics_data = {
                    'Word': [word for word, count in st.session_state.agent.topics[:10]],  # Top 10
                    'Count': [count for word, count in st.session_state.agent.topics[:10]]
                }
                df_topics = pd.DataFrame(topics_data)
                
                # Create bar chart
                fig = px.bar(
                    df_topics, 
                    x='Word', 
                    y='Count',
                    title='Top 10 Keywords',
                    color='Count',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig)
                
                # Create word cloud using plotly
                fig = px.scatter(
                    df_topics,
                    x=[i for i in range(len(df_topics))],
                    y=[i % 3 for i in range(len(df_topics))],
                    size='Count',
                    color='Count',
                    text='Word',
                    size_max=60,
                    color_continuous_scale='Viridis',
                    title='Keyword Bubble Chart'
                )
                fig.update_traces(textposition='top center')
                fig.update_layout(
                    xaxis={'showgrid': False, 'zeroline': False, 'visible': False},
                    yaxis={'showgrid': False, 'zeroline': False, 'visible': False}
                )
                st.plotly_chart(fig)

if __name__ == "__main__":
    main()
