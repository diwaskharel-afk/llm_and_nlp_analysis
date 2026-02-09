# Edgar Allan Poe Short Stories: NLP Analysis Pipeline

## 📚 Project Overview

This project presents a comprehensive Natural Language Processing (NLP) analysis of Edgar Allan Poe's complete corpus of short stories. The pipeline combines modern transformer-based models with traditional NLP techniques to perform semantic chunking, topic modeling, named entity recognition, keyphrase extraction, and gender bias analysis.

---

## 🎯 Research Objectives

1. **Semantic Structure Analysis**: Break down Poe's stories into semantically coherent chunks
2. **Thematic Exploration**: Identify dominant topics and narrative themes across the corpus
3. **Character & Concept Extraction**: Extract key entities and phrases from each story
4. **Gender Representation Analysis**: Investigate potential gender bias in character portrayal and trait attribution

---

## 📊 Dataset

### Source
**E.A. Poe's Corpus of Short Stories**

### Metadata Columns
- `title`: Story title
- `text`: Full story text
- `wikipedia_title`: Wikipedia reference
- `publication_date`: Original publication date
- `first_published_in`: Original publication venue
- `classification`: Genre/type classification
- `notes`: Additional annotations
- `normalized_date`: Standardized publication date

### Statistics
- **Total Stories**: 70
- **Text Length Range**: 1,000 – 20,000 words
- **Average Story Length**: 4,839.9 words
- **Total Corpus Size**: ~338,793 words

---

## 🔬 Methodology

### 1. Semantic Chunking

**Objective**: Split each story into semantically coherent segments while preserving narrative flow.

**Implementation**:
```python
Model: SentenceTransformer (all-MiniLM-L6-v2)
Similarity Threshold: Dynamic (configurable)
Chunk Size: MIN_TOKENS to MAX_TOKENS per chunk
Strategy: Cosine similarity-based segmentation
```

**Algorithm**:
1. Tokenize story into sentences using NLTK's `sent_tokenize`
2. Generate embeddings for each sentence using SentenceTransformer
3. Calculate cosine similarity between consecutive sentence embeddings
4. Split when:
   - Similarity drops below threshold (semantic shift detected)
   - Token count exceeds maximum limit
5. Merge final chunk if below minimum token threshold
6. Maintain incremental mean embedding for each chunk

**Output**: `samentic_chunk.csv` with columns:
- `title`: Story identifier
- `chunk_id`: Sequential chunk number
- `chunk_text`: Chunk content

---

### 2. Topic Modeling

**Objective**: Identify the dominant thematic topic for each complete story.

**Framework**: BERTopic

**Components**:
- **Embedding Model**: `SentenceTransformer (all-MiniLM-L6-v2)`
- **Dimensionality Reduction**: UMAP
  - `n_neighbors=5`
  - `n_components=5`
  - `metric='cosine'`
- **Clustering**: HDBSCAN
  - `min_cluster_size=2`
  - `min_samples=1`
  - `cluster_selection_method='eom'`
- **Vectorization**: CountVectorizer
  - `stop_words='english'`
  - `ngram_range=(1, 2)`

**Process**:
1. Process each story independently (per-story topic modeling)
2. Embed all chunks belonging to one story
3. Fit BERTopic model on story's chunks
4. Identify dominant topic using majority voting
5. Extract top 5 keywords for the dominant topic
6. Calculate topic confidence (% of chunks assigned to dominant topic)

**Output Columns** (added to `preprocessed_data.csv`):
- `topic_id`: Dominant topic identifier (-1 for noise/insufficient data)
- `topic_keywords`: Top 5 keywords representing the topic
- `num_chunks`: Number of chunks in the story
- `topic_confidence`: Confidence score (0-1)

**Edge Cases**:
- Stories with <3 chunks: Marked as "insufficient_data"
- No dominant topic found: Marked as "no_topic_found"
- Processing errors: Marked as "error"

---

### 3. Named Entity Recognition & Keyphrase Extraction

**Objective**: Extract character names and thematic concepts from each chunk.

**Tools**:
- **NER**: spaCy (`en_core_web_sm`)
- **Keyphrase Extraction**: KeyBERT with MMR (Maximal Marginal Relevance)

**Process**:

#### 3.1 Person Entity Extraction
```python
1. Process chunk_text with spaCy NLP pipeline
2. Filter entities where label == "PERSON"
3. Return list of person names
```

#### 3.2 Text Cleaning
```python
1. Remove identified person entities from text
2. Create cleaned_text for keyphrase extraction
   (prevents character names from dominating keyphrases)
```

#### 3.3 Keyphrase Extraction
```python
Parameters:
- keyphrase_ngram_range: (1, 3)
- stop_words: "english"
- use_mmr: True (for diversity)
- diversity: 0.5
- top_n: 10
```

**Output Columns** (added to chunk-level data):
- `person_entities`: List of character names
- `cleaned_text`: Text with person names removed
- `keyphrases`: Top 10 thematic keyphrases

---

### 4. Gender Bias Analysis

**Objective**: Detect and quantify gender representation and trait attribution bias in Poe's narratives.

**Research Questions**:
1. Are male/female characters represented equally?
2. Do male/female characters receive different trait attributions?
3. Are power/intelligence/bravery scored differently by gender?

**Methodology**: LLM-based Character Analysis

**Model**: `GPT-4.1-nano`
- Temperature: 0.1
- Max Tokens: 3,500

**Analysis Dimensions** (scored -2 to +2):
- **Intelligence**: Problem-solving, knowledge, wisdom, strategic thinking
- **Bravery**: Courage, facing danger, taking risks
- **Power**: Authority, control, influence over others/events
- **Agency**: Decision-making, plot driving, independent action
- **Emotionality**: Emotional expression and presence (neutral metric)

**Scoring System**:
- `+2`: Strongly demonstrated (multiple clear examples)
- `+1`: Demonstrated (one clear example)
- `0`: No evidence or neutral
- `-1`: Contradicted (shown to lack trait)
- `-2`: Strongly contradicted (multiple contradictions)

**Gender Identification Rules**:
- **ONLY** based on explicit evidence in text
- Pronouns: he/she/him/her
- Direct statements: "the woman", "the man"
- Ambiguous cases: Marked as "unknown"

**Process**:
1. For each chunk, extract ALL characters (not limited to preset count)
2. Identify gender using strict evidence-based rules
3. Score each character on 5 dimensions
4. Extract character traits mentioned in text
5. Flatten to character-level rows
6. Aggregate statistics by gender

**Output Files**:
- `raw_characters_{timestamp}.csv`: Character-level data with scores
- `bias_summary_{timestamp}.csv`: Aggregated gender statistics
- `processing_log_{timestamp}.txt`: Detailed execution log

**Statistical Analysis**:
- Character count and percentage by gender
- Mean, median, and standard deviation for each dimension
- Difference scores (male - female) for comparative analysis

---

## 📁 Project Structure

```
project/
│
├── samentic_chunk.csv              # Semantic chunks output
├── preprocessed_data.csv           # Original stories + topic data
│
├── semantic_chunking.ipynb         # Chunking implementation
├── topic_modeling_each_topic.ipynb # Topic modeling pipeline
├── ner_keyphrase_extraction.ipynb  # NER + KeyBERT processing
├── gender_bias_analysis.py         # LLM-based bias analysis
│
└── analysis_results/               # Output directory
    ├── raw_characters_*.csv
    ├── bias_summary_*.csv
    └── processing_log_*.txt
```

---

## 🛠️ Technologies Used

### Core Libraries
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **nltk**: Sentence tokenization

### NLP & ML
- **sentence-transformers**: Semantic embeddings
- **BERTopic**: Topic modeling framework
- **umap-learn**: Dimensionality reduction
- **hdbscan**: Density-based clustering
- **spaCy**: Named entity recognition
- **keybert**: Keyphrase extraction
- **scikit-learn**: Cosine similarity, CountVectorizer

### LLM Integration
- **openai**: GPT-4 API for gender bias analysis

---

## 🚀 Usage

### 1. Semantic Chunking
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
chunks = semantic_chunk_story(story_text, model)
```

### 2. Topic Modeling
```python
# Run on chunked data
python topic_modeling_each_topic.py

# Output: preprocessed_data.csv with topic columns
```

### 3. NER & Keyphrase Extraction
```python
# Process chunks
df["person_entities"], df["keyphrases"] = df.apply(process_chunk)
```

### 4. Gender Bias Analysis
```python
# Configure in script
NUM_STORIES = 10  # Adjust sample size

# Run analysis
python gender_bias_analysis.py

# Check results in analysis_results/
```

---

## 📈 Key Findings

### Semantic Chunking Results
- Successfully segmented all 70 stories into coherent semantic units
- Preserved narrative flow while maintaining manageable chunk sizes
- Enabled granular analysis of thematic shifts within stories

### Topic Modeling Insights
- Identified dominant themes across Poe's corpus
- Topics range from gothic horror to psychological introspection
- Confidence scores indicate thematic consistency within stories

### Character & Concept Extraction
- Extracted primary characters from each narrative segment
- Identified recurring motifs and thematic concepts
- Separated character analysis from thematic analysis for clarity

### Gender Bias Analysis
- Quantified gender representation in Poe's narratives
- Analyzed trait attribution patterns by gender
- Provided evidence-based scores for character dimensions

*(Note: Specific numerical findings depend on execution results)*

---

## ⚠️ Limitations & Considerations

1. **Historical Context**: Analysis reflects 19th-century literary conventions and social norms
2. **Sample Size**: Limited to Poe's short story corpus (70 stories)
3. **LLM Subjectivity**: Gender bias scores depend on GPT-4's interpretation
4. **Gender Binary**: Analysis focuses on binary gender identification due to historical text constraints
5. **Chunk Granularity**: Semantic chunking parameters may affect topic coherence

---

## 🔮 Future Work

- [ ] Expand to other 19th-century authors for comparative analysis
- [ ] Implement cross-corpus topic modeling
- [ ] Add sentiment analysis per chunk
- [ ] Develop visualization dashboard for findings
- [ ] Conduct temporal analysis of themes across Poe's career
- [ ] Implement more nuanced gender analysis frameworks
- [ ] Add character relationship network analysis

---




## 🙏 Acknowledgments

- **Edgar Allan Poe**: For the timeless literary corpus
- **HuggingFace**: For transformer models and embeddings
- **OpenAI**: For GPT-4 API access
- **Open-source NLP Community**: For the incredible tools and libraries

---



**Last Updated**: February 2025
