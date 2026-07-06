# Interview Analysis Pipeline

A comprehensive pipeline for transcribing and analyzing interview audio files with automatic language detection, sentiment analysis, and topic modeling.

## Features

- **Audio Transcription**: Whisper-based transcription with multiple model sizes
- **Speaker Diarization**: PyAnnote-based speaker identification
- **Language Support**: Automatic detection or forced language selection
- **Context-Aware**: Custom prompts for better transcription accuracy
- **Case-Insensitive**: Handles any VM directory or file extension case
- **Model Caching**: Download once, reuse forever
- **Multi-Language**: Support for 10+ languages

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# For GPU support (optional) — match your CUDA version
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cu130
```

## Usage

### Basic Commands

```bash
# Scan dataset structure
python -m src.pipeline.run --scan

# Perform speaker diarization (run before --transcribe)
python -m src.pipeline.run --diarize

# Transcribe audio files (English, small model - RECOMMENDED)
python -m src.pipeline.run --transcribe

# Sentiment analysis of the participant (run after --diarize and --transcribe)
python -m src.pipeline.run --sentiment

# Topic modeling of participants across all interviews (run after --transcribe)
python -m src.pipeline.run --topics

# Knowledge graph of the participant (Text2AMR2FRED, run after --transcribe)
python -m src.pipeline.run --kg

# Shared-opinion analysis across participants (run after --kg and --sentiment)
python -m src.pipeline.run --opinions

# Run every step in order: diarize -> transcribe -> sentiment -> topics -> kg -> opinions
python -m src.pipeline.run --all

# Human check of diarization + transcription (after the pipeline)
python scripts/human_check_transcript.py --per-file 3
```

### Directory Structure

```
interview_analysis/
├── data/
│   ├── raw/                # Place your audio files here
│   │   ├── VM01/           # Interview VM01
│   │   │   ├── file1.wav   # Audio file
│   │   │   └── file2.m4a   # Audio file (converted to WAV by scanner)
│   │   └── VM02/           # Interview VM02
│   │       └── ...
│   ├── processed/          # Processed outputs
│   │   ├── transcriptions/ # Transcription results
│   │   │   ├── VM01/       # VM01 transcriptions
│   │   │   │   ├── file1.wav.json
│   │   │   │   └── file2.m4a.json  # Transcription of converted WAV
│   │   │   └── VM02/       # VM02 transcriptions
│   │   │       └── ...
│   │   ├── diarization/    # Speaker diarization results
│   │   │   ├── VM01/       # VM01 diarization
│   │   │   │   ├── file1.wav.diarization.json
│   │   │   │   └── file2.m4a.diarization.json
│   │   │   └── VM02/       # VM02 diarization
│   │   │       └── ...
│   │   ├── sentiment/      # Participant sentiment results
│   │   │   └── VM01/       # <base>.<SPEAKER>.sentiment.json
│   │   ├── topics/         # Participant topic modeling results
│   │   │   ├── topics_model.json   # Global topic table (whole corpus)
│   │   │   └── VM01/       # <base>.<SPEAKER>.topics.json
│   │   ├── knowledge_graph/ # Participant RDF/OWL knowledge graphs
│   │   │   └── VM01/       # <base>.<SPEAKER>.ttl + .kg.json
│   │   └── opinions/       # Cross-participant shared-opinion analysis
│   │       └── opinions.json
│   └── results/            # Statistical analyses + figures
│       ├── sync_async_sentiment_scores.csv   # tidy per-interview scores
│       ├── sync_async_sentiment_tests.json   # paired tests + method agreement
│       ├── sync_async_paired.png             # paired-line plot per measure
│       ├── sync_async_method_correlation.png
│       ├── sentiment_histograms.png          # score distributions (normality check)
│       ├── representative_topics.csv         # ranked shared topics
│       ├── topic_participant_heatmap.png     # topic × participant sentiment
│       ├── top_topics_coverage.png           # coverage bar (consensus-coloured)
│       ├── emotion_profile.png               # corpus Hourglass emotions
│       ├── semantic_categories.csv           # act + theme categories
│       ├── semantic_act_heatmap.png          # act category × participant sentiment
│       ├── semantic_theme_heatmap.png        # theme category × participant sentiment
│       ├── emotions_felt_hourglass.png       # Hourglass within 'emotion' acts
│       ├── theme_act_crosstab.csv            # ACT × THEME cells (pooled)
│       ├── theme_act_sentiment.png           # ACT × THEME mean sentiment map
│       ├── theme_act_frequency.png           # ACT × THEME co-occurrence counts
│       ├── theme_act_crosstab_by_condition.csv
│       ├── theme_act_sentiment_by_condition.png  # Sync | Async maps
│       ├── theme_act_selected_bars.png       # Sync vs Async bars for key pairs
│       ├── cluster_features.csv              # per-interview signature + cluster
│       ├── cluster_validation.json           # k selection, stability, cluster×condition
│       ├── cluster_profiles.png              # cluster × feature profile (interpretable)
│       └── cluster_pca.png                   # interviews in signature space (PCA)
├── configs/                # Configuration files
│   └── config.json         # Main configuration
├── src/                    # Source code
│   ├── audio/             # Audio processing
│   │   ├── scanner.py     # Dataset scanner
│   │   ├── transcriber.py  # Transcription module
│   │   └── diarizer.py    # Speaker diarization module
│   ├── config.py          # Configuration management
│   ├── pipeline/          # Pipeline modules
│   │   └── run.py          # Main pipeline entry
│   └── text/              # Text processing (future)
├── README.md               # This file
├── requirements.txt        # Python dependencies
└── environment.yml         # Conda environment
```

## Configuration

Edit `configs/config.json` to customize:

```json
{
  "project": {
    "name": "Interview Analysis Pipeline",
    "version": "1.0.0"
  },
  "paths": {
    "raw_audio_dir": "data/raw",
    "processed_dir": "data/processed",
    "results_dir": "data/results"
  },
  "processing": {
    "device": "auto",
    "whisper_model": "small"
  },
  "huggingface": {
    "token": "YOUR_HUGGINGFACE_TOKEN_HERE",
    "offline": true
  },
  "logging": {
    "level": "INFO",
    "file": "logs/pipeline.log"
  }
}
```

**Key Features:**
- **Offline Mode**: Enabled by default (`HF_HUB_OFFLINE=1`) for privacy and performance
- **Hugging Face Token**: Required for PyAnnote speaker diarization
- **Flexible Paths**: Customize data directories
- **Device Selection**: Auto-detects GPU/CPU

**For Hugging Face token:**
1. Get your token from https://huggingface.co/settings/tokens
2. Add it to `configs/config.json` under `huggingface.token`
3. This enables PyAnnote speaker diarization with authenticated models
4. Token is used in offline mode - no network calls after initial model download

## Whisper Transcription Model Comparison

| Model  | Size   | Quality | Speed | Use Case |
|--------|--------|---------|-------|----------|
| tiny   | 72MB   | Low     | ⚡ Fast  | Quick tests, drafts |
| base   | 142MB  | Medium  | ⚡ Fast  | General use |
| small  | 466MB  | Good    | ⚡ Fast  | **Default** - best balance |
| medium | 1.5GB  | High    | 🐢 Slow  | High accuracy |
| large  | 2.88GB | Best    | 🐢 Slow  | Professional quality |

## Transcription Output

Transcription is **speaker-aware**: run `--diarize` first, then `--transcribe`.
Each Whisper segment is attributed to the diarization speaker it overlaps most,
and **one JSON file is written per speaker**:

```
data/processed/transcriptions/VM05/
├── interview.wav.SPEAKER_00.transcription.json
└── interview.wav.SPEAKER_01.transcription.json
```

Each per-speaker file contains only that speaker's segments:

```json
{
  "filename": "interview.wav",
  "speaker": "SPEAKER_00",
  "language": "en",
  "text": "Everything this speaker said, concatenated...",
  "segments": [
    {
      "start": 0.0,
      "end": 5.0,
      "previous_other": "The other speaker's preceding utterance",
      "text": "First segment text..."
    },
    ...
  ]
}
```

Each segment carries a `previous_other` field with the text of the most recent
preceding utterance from a *different* speaker, so short replies like
`"Okay, cool."` keep the context of what they responded to.

If no diarization result is found for a file, transcription falls back to a
single combined `interview.wav.transcription.json` and logs a warning.

Run options:

```bash
# Transcribe with auto language detection
python -m src.pipeline.run --transcribe --language auto

# Transcribe with custom context
python -m src.pipeline.run --transcribe --prompt "interview about experiment"

# Transcribe French interviews
python -m src.pipeline.run --transcribe --language fr --prompt "Entretien d'experience"

# Use different model sizes
python -m src.pipeline.run --transcribe --model tiny    # Fast, low quality (72MB)
python -m src.pipeline.run --transcribe --model small   # Default - good balance (466MB)
python -m src.pipeline.run --transcribe --model medium  # High quality (1.5GB)
python -m src.pipeline.run --transcribe --model large   # Best quality (2.88GB)

# Enable debug logging
python -m src.pipeline.run --transcribe --debug
```

## Speaker Diarization Output

Each diarization is saved as JSON with:

```json
{
  "filename": "interview.wav",
  "speakers": {
    "speaker_0": [
      {
        "start": 0.0,
        "end": 5.0,
        "duration": 5.0
      },
      {
        "start": 10.0,
        "end": 15.0,
        "duration": 5.0
      },
      {
        "total_speech_time": 10.0
      }
    ],
    "speaker_1": [
      {
        "start": 5.0,
        "end": 10.0,
        "duration": 5.0
      },
      {
        "total_speech_time": 5.0
      }
    ]
  },
  "speaker_count": 2,
  "total_duration": 120.5
}
```

## Sentiment Analysis Output

Sentiment is computed on the **participant only**. For each interview the
experimenter is assumed to be the first person to speak; the other speaker is the
participant. Each utterance gets a transformer 3-class label
(`cardiffnlp/twitter-roberta-base-sentiment-latest`: negative/neutral/positive)
plus a VADER compound score as a secondary signal.

The `Sync` / `Async` interview type is kept in the output so a participant's two
interviews can be compared later. Results are saved as
`data/processed/sentiment/VMxx/<base>.<SPEAKER>.sentiment.json`:

```json
{
  "vm": "VM05",
  "interview": "Async",
  "base_filename": "VM05-2-Async",
  "participant_speaker": "SPEAKER_01",
  "experimenter_speaker": "SPEAKER_00",
  "n_utterances": 71,
  "overall": {
    "distribution": {"negative": 0.04, "neutral": 0.61, "positive": 0.35},
    "counts": {"negative": 3, "neutral": 43, "positive": 25},
    "mean_signed_score": 0.29,
    "mean_vader_compound": 0.38,
    "label": "neutral"
  },
  "most_positive": { "text": "...", "signed_score": 0.97, "...": "..." },
  "most_negative": { "text": "...", "signed_score": -0.57, "...": "..." },
  "segments": [
    {
      "start": 26.8, "end": 30.6, "text": "...",
      "label": "positive", "score": 0.97,
      "scores": {"negative": 0.01, "neutral": 0.02, "positive": 0.97},
      "signed_score": 0.97, "vader_compound": 0.71
    }
  ]
}
```

> **Note:** the first transformer run downloads the model (~500 MB) once and caches it.

### Methods

Each participant utterance is scored with **two complementary approaches**, so a
contextual model and a transparent lexicon can be compared:

**1. Transformer 3-class (primary) — `label`, `score`, `scores`, `signed_score`**

A RoBERTa model fine-tuned for sentiment on tweets, returning calibrated
probabilities over `negative` / `neutral` / `positive`. We report the top label,
its probability, the full distribution, and a continuous
`signed_score = P(positive) − P(negative)` in `[-1, 1]`. This captures context and
negation far better than a lexicon, at the cost of a one-time model download and
GPU/CPU inference.

- Model: [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest)
  — a RoBERTa-base trained on ~124M tweets (2018–2021) and fine-tuned on the
  TweetEval sentiment benchmark.
- Library: [🤗 Transformers](https://github.com/huggingface/transformers) on [PyTorch](https://pytorch.org/).

**2. VADER (secondary) — `vader_compound`**

A rule-based, lexicon-driven analyzer that needs no training or download and is
fully offline. It returns a normalized `compound` score in `[-1, 1]`. Fast and
transparent, but weaker on nuanced/contextual phrasing; kept here as a lightweight
cross-check against the transformer.

- Package: [`vaderSentiment`](https://github.com/cjhutto/vaderSentiment).

### References

- **VADER** — Hutto, C.J. & Gilbert, E.E. (2014). *VADER: A Parsimonious Rule-based
  Model for Sentiment Analysis of Social Media Text.* Proceedings of the 8th
  International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI.
- **twitter-roberta ("latest") / TimeLMs** — Loureiro, D., Barbieri, F., Neves, L.,
  Espinosa-Anke, L., & Camacho-Collados, J. (2022). *TimeLMs: Diachronic Language
  Models from Twitter.* ACL 2022, System Demonstrations. arXiv:2202.03829.
- **TweetEval (sentiment fine-tuning benchmark)** — Barbieri, F., Camacho-Collados,
  J., Espinosa-Anke, L., & Neves, L. (2020). *TweetEval: Unified Benchmark and
  Comparative Evaluation for Tweet Classification.* Findings of EMNLP 2020.
  arXiv:2010.12421.
- **RoBERTa (base architecture)** — Liu, Y., et al. (2019). *RoBERTa: A Robustly
  Optimized BERT Pretraining Approach.* arXiv:1907.11692.
- **Transformers library** — Wolf, T., et al. (2020). *Transformers: State-of-the-Art
  Natural Language Processing.* EMNLP 2020, System Demonstrations. arXiv:1910.03771.

## Topic Modeling Output

Topics are modeled on the **participant only**, over **all interviews at once**
(topics are a corpus-level property), then reported per interview so `Sync`/`Async`
conditions can be compared. Each **document** is a participant utterance embedded
with its `previous_other` context.

Two files are produced:

- `data/processed/topics/topics_model.json` — the global topic table learned over
  the whole corpus (topic id, keywords, size, representative utterances). Topic
  `-1` is the outlier/no-topic bucket.
- `data/processed/topics/VMxx/<base>.<SPEAKER>.topics.json` — per-interview topic
  distribution and the topic assigned to each participant utterance.

```json
{
  "vm": "VM05",
  "interview": "Async",
  "base_filename": "VM05-2-Async",
  "participant_speaker": "SPEAKER_01",
  "topic_distribution": {"0": 42, "1": 17, "2": 11},
  "segments": [
    {
      "start": 73.5, "end": 75.98,
      "previous_other": "So you managed to relax more than others?",
      "text": "Yeah, exactly, exactly unlike last time, yes.",
      "topic_id": 0,
      "topic_keywords": ["body", "feel", "time", "phase", "moving"]
    }
  ]
}
```

> **Note:** topic modeling needs a corpus to form clusters. With only one or two
> interviews the result is thin (few topics + outliers); it becomes meaningful as
> more VMs are added. The embedding model (~90 MB) is downloaded once and cached.

### Method

- **BERTopic** — embedding-based topic modeling: sentence embeddings → UMAP
  dimensionality reduction → HDBSCAN clustering → class-based TF-IDF (c-TF-IDF)
  keyword extraction. English stopwords are removed from the keyword step only
  (embeddings use the full text, including `previous_other`).
- Embeddings: [`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
- Library: [BERTopic](https://github.com/MaartenGr/BERTopic).

### References

- **BERTopic** — Grootendorst, M. (2022). *BERTopic: Neural topic modeling with a
  class-based TF-IDF procedure.* arXiv:2203.05794.
- **Sentence-BERT (embeddings)** — Reimers, N. & Gurevych, I. (2019).
  *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks.* EMNLP 2019.
  arXiv:1908.10084.
- **UMAP (dimensionality reduction)** — McInnes, L., Healy, J., & Melville, J.
  (2018). *UMAP: Uniform Manifold Approximation and Projection for Dimension
  Reduction.* arXiv:1802.03426.
- **HDBSCAN (clustering)** — Campello, R.J.G.B., Moulavi, D., & Sander, J. (2013).
  *Density-Based Clustering Based on Hierarchical Density Estimates.* PAKDD 2013.

## Knowledge Graph Output

Knowledge graphs are built on the **participant only** via **Text2AMR2FRED**, fully
locally. Each participant utterance's own `text` (not `previous_other` — so topics
aren't contaminated by the experimenter's questions) is parsed to **Abstract Meaning
Representation** and converted to a **FRED RDF/OWL** graph; the per-utterance graphs
are merged into one graph per interview. (`previous_other` is kept as metadata on
each utterance.)

Two files per interview:

- `data/processed/knowledge_graph/VMxx/<base>.<SPEAKER>.ttl` — the merged RDF graph
  (Turtle), grounded in PropBank rolesets, VerbNet roles and the DUL ontology, with
  Boxer-style modality/negation (`boxing:hasTruthValue`).
- `data/processed/knowledge_graph/VMxx/<base>.<SPEAKER>.kg.json` — stats: triple
  count, distinct frames, and the most frequent frames/predicates.

**Provenance & merging:** FRED domain individuals are namespaced per utterance
(`fred:u{idx}_…`) so utterances never merge by accident, while shared vocabulary
(PropBank/DUL/VerbNet) stays common and queryable across the interview. Every
individual is linked to its source utterance via `ia:fromUtterance`, and each
utterance carries `ia:text`, `ia:previousOther`, `ia:startTime/endTime`, and the
`ia:interviewType` (Sync/Async). Example SPARQL:

```sparql
PREFIX ia: <http://interview-analysis.local/ontology#>
SELECT ?text (COUNT(?ind) AS ?frames) WHERE {
  ?u a ia:Utterance ; ia:text ?text .
  ?ind ia:fromUtterance ?u .
} GROUP BY ?u ?text ORDER BY DESC(?frames)
```

### One-time model setup

The AMR parse model is **not** installed by pip. Download it once (~500 MB) and
link it as amrlib's default `model_stog`:

```bash
DATA_DIR="venv/lib/python3.*/site-packages/amrlib/data"
mkdir -p "$DATA_DIR"
curl -L https://github.com/bjascob/amrlib-models/releases/download/parse_xfm_bart_base-v0_1_0/model_parse_xfm_bart_base-v0_1_0.tar.gz \
  | tar xz -C "$DATA_DIR"
ln -sfn model_parse_xfm_bart_base-v0_1_0 "$DATA_DIR/model_stog"
```

> `scripts/kg_smoke_test.py` proves the local text→AMR→RDF pipeline end-to-end.
> Wikidata post-processing is disabled by design (external 832 MB DB, meaningless
> for subjective speech) — the graphs are produced fully offline and deterministic.

### Method

- **AMR parsing** — a neural parser turns each sentence into an Abstract Meaning
  Representation graph (predicate–argument "who did what to whom").
  Model: `amrlib` `parse_xfm_bart_base` (fine-tuned BART, SPRING-style).
- **AMR → RDF/OWL** — [`py_amr2fred`](https://github.com/infovillasimius/amr2Fred)
  maps AMR to FRED's formal semantics + ontology design patterns.

### References

- **Text2AMR2FRED** — Gangemi, A., Graciotti, A., Meloni, A., et al. (2025).
  *Text2AMR2FRED, converting text into RDF/OWL knowledge graphs via abstract
  meaning representation.* Knowledge and Information Systems. doi:10.1007/s10115-025-02631-y.
- **FRED** — Gangemi, A., Presutti, V., Reforgiato Recupero, D., Nuzzolese, A.G.,
  Draicchio, F., & Mongiovì, M. (2017). *Semantic Web Machine Reading with FRED.*
  Semantic Web Journal 8(6).
- **AMR** — Banarescu, L., et al. (2013). *Abstract Meaning Representation for
  Sembanking.* Linguistic Annotation Workshop (LAW VII).
- **SPRING (AMR parsing)** — Bevilacqua, M., Blloshmi, R., & Navigli, R. (2021).
  *One SPRING to Rule Them Both: Symmetric AMR Semantic Parsing and Generation
  without a Complex Pipeline.* AAAI 2021.

## Opinion Analysis Output

A **frame-based, cross-participant** analysis (inspired by **Sentilo**) that asks:
*which opinions do participants share?* The opinion **holder** is always the
participant, so the stage focuses on **topics** and **sentiment**:

1. **Topics** — the noun concepts each utterance is *about* are read from the
   participant's FRED graph (`fred:` classes; PropBank verb frames are excluded as
   stance, not topic).
2. **Sentiment** — each `(participant, topic)` opinion is scored by the contextual
   utterance sentiment (transformer signed score) enriched with **concept-level
   affect** from the offline **SenticNet** KB (polarity + Hourglass emotion).
3. **Canonicalization** — topics are merged across participants (WordNet lemma +
   sentence-embedding similarity) so `rocks`≈`rock`≈`stone`.
4. **Shared-opinion metrics** — per canonical topic: coverage (how many
   participants), consensus score, spread, a `consensus_positive` /
   `consensus_negative` / `contested` label, and the **Sync vs Async** contrast.

Output: `data/processed/opinions/opinions.json`, with two sections:

- `profiles` — per participant: an **Individual Sentiment Score** plus their
  per-topic scores, mention counts and dominant emotions.
- `shared` — per canonical topic: coverage, consensus/spread, label, per-participant
  scores, Sync/Async means, and top emotions.

```json
{
  "shared": [
    {
      "topic": "sensation", "n_participants": 5, "consensus_score": 0.13,
      "spread": 0.08, "label": "consensus_positive",
      "sync_score": 0.20, "async_score": 0.07,
      "participant_scores": {"VM05/SPEAKER_01": 0.15, "...": "..."},
      "top_moods": [["joy", 4], ["calmness", 2]]
    }
  ]
}
```

> **Note:** topic detection is a heuristic (noun concepts from FRED). A few
> adverbial noun-homonyms (e.g. "still", "now") can slip through; consensus
> statistics get more reliable as more participants are added.

### Method & References

- **Approach** — frame-based opinion mining on FRED graphs, à la **Sentilo**:
  holder + topics/subtopics + sentiment propagation.
- **Concept affect** — offline **SenticNet** (`senticnet` package): concept
  polarity + Hourglass-model emotions, fully local (no API, no data leaves the host).

- **Sentilo** — Reforgiato Recupero, D., Presutti, V., Consoli, S., Gangemi, A., &
  Nuzzolese, A.G. (2015). *Sentilo: Frame-Based Sentiment Analysis.* Cognitive
  Computation 7(2). doi:10.1007/s12559-014-9302-z.
- **Frame-based opinion detection** — Gangemi, A., Presutti, V., & Reforgiato
  Recupero, D. (2014). *Frame-Based Detection of Opinion Holders and Topics: A Model
  and a Tool.* IEEE Computational Intelligence Magazine 9(1). doi:10.1109/MCI.2013.2291688.
- **SenticNet / sentic computing** — Cambria, E., et al. *SenticNet* — a
  commonsense knowledge base for concept-level sentiment analysis. https://sentic.net.

## Human Verification

### Transcrition and diarization (audio)

After running the pipeline, spot-check the two automated steps that everything else
depends on — **speaker diarization** and **transcription**:

```bash
python scripts/human_check_transcript.py --per-file 3      # 3 clips per interview
python scripts/human_check_transcript.py --vm VM05         # one interview set
python scripts/human_check_transcript.py --dry-run         # select+extract, no prompts
```

For each interview it samples short audio clips — spread across the timeline and
across both speakers (`--per-file`, 1–10) — plays each from the **raw** audio, and asks:

1. **who is speaking** — experimenter or participant (asked *before* revealing the
   pipeline's guess, to avoid confirmation bias);
2. whether the shown **transcription text** is correct.

Controls per clip: `r` replay, `s` skip, `q` quit (saves a partial report). At the end
it prints and saves (`data/results/human_check_<timestamp>.json`) a report of what
matched the pipeline and what did not — speaker mismatches and transcription errors,
each with its interview, timestamp and text. Requires an audio player (`ffplay`,
`paplay` or `aplay`).

### Emotion attribution (text only)

Each Hourglass emotion in the profile comes from a *concept* in a sentence (the FRED
topic concept → SenticNet moodtag). This script inverts that mapping so you can check
it:

```bash
python scripts/human_check_emotions.py                 # interactive verify
python scripts/human_check_emotions.py --list          # just print examples
python scripts/human_check_emotions.py --emotion joy --per-emotion 8
```

For each emotion it shows representative sentences (favouring diverse trigger concepts,
`--per-emotion` 1–20, and skipping sentences under `--min-words`), each with the
**trigger concept** and its interview. Interactively it asks whether the sentence
conveys the emotion (`y`/`n`/`s`/`q`) and reports per-emotion agreement +
misfits to `data/results/emotion_check_<timestamp>.json`.

> This is a genuine QA check: because emotions are attributed via a single topic
> concept's dictionary moodtag, some attributions are weak (e.g. `color → calmness`).
> The script makes that visible so the emotion profile is read with appropriate caution.

### Sentiment score distributions (text only)

```bash
python scripts/human_check_sentiments.py
```

Plots per-utterance histograms (with KDE + normal-reference curve) of the three
measures — `roberta`, `vader`, `opinion` (mean SenticNet concept polarity) — so you
can eyeball whether they look reasonable. Output: `data/results/sentiment_histograms.png`.
In practice RoBERTa is close to normal, VADER is zero-inflated (a spike at 0), and the
SenticNet-based opinion score is positively skewed — worth knowing before any test
that assumes normality.

## Statistical Analysis (data/results)

Analyses read the `data/processed/` outputs and write tables + figures to
`data/results/`. They are separate from the pipeline stages (run on demand).

### Aspect 1 — Sync vs Async sentiment

```bash
python -m src.analysis.sync_async_sentiment
```

Each VM is one participant measured in **both** conditions, so this is a
**within-subject (paired)** design. Three interview-level scores are compared —
`roberta` (transformer signed score), `vader` (lexicon compound), and `opinion`
(mean offline-SenticNet polarity over the FRED topics) — with, per measure:

- **paired t-test** + **Wilcoxon signed-rank** (two views, honest for small n),
- mean paired difference (95% CI) and **Cohen's dz**,
- **method agreement** (Pearson/Spearman among the three measures).

Outputs: `sync_async_sentiment_scores.csv`, `sync_async_sentiment_tests.json`,
`sync_async_paired.png` (paired-line plot), `sync_async_method_correlation.png`.

> With only a few participants this validates the **procedure**, not significance
> (Wilcoxon cannot reach p<0.05 below n=6).

### Aspect 2 — representative topics & opinions (all participants)

```bash
python -m src.analysis.representative_topics
```

Pools Sync + Async and reads `opinions.json` to represent, across the whole group:
the most representative topics (by participant coverage), each topic's consensus
sentiment (shared vs contested), and the corpus emotion profile.

Outputs: `representative_topics.csv`, `topic_participant_heatmap.png` (topic ×
participant sentiment; grey = not mentioned), `top_topics_coverage.png` (coverage
bar coloured by consensus), `emotion_profile.png` (SenticNet Hourglass emotions).

### Aspect 2b — semantic categories ("beyond words")

```bash
python -m src.analysis.semantic_categories
```

Lifts surface concepts to coarse **semantic categories** grounded in the FRED graph,
along two axes:

- **ACT** — verb frames → WordNet verb supersense: *perception* ("describing what
  they see"), *emotion* ("emotions felt"), *cognition* ("reflecting"), *motion*,
  *communication* …
- **THEME** — noun concepts → WordNet noun supersense: *body*, *time*, *objects*,
  *location/space*, *plant/flora*, *feeling* …

Each category is aggregated across participants (coverage, consensus sentiment,
Sync/Async, Hourglass emotions). Outputs: `semantic_categories.csv`,
`semantic_act_heatmap.png`, `semantic_theme_heatmap.png`, and
`emotions_felt_hourglass.png` (the emotion profile *within* emotion acts).

> Categories use WordNet supersenses over FRED's disambiguated frames/concepts.
> Noun sense selection is first-sense, so a few words land in a coarse class;
> the verb/act axis is cleaner (frames are disambiguated by the AMR parse).

### Aspect 2c — ACT × THEME cross-tab

```bash
python -m src.analysis.theme_act_crosstab
```

Crosses the two axes: for each utterance, every co-occurring `(act, theme)` inherits
the utterance sentiment; cells are averaged per participant then across participants.
Answers "how do they feel when **perceiving the body** vs **feeling about the
environment**?". Outputs: `theme_act_crosstab.csv`, `theme_act_sentiment.png`
(ACT × THEME mean-sentiment map), `theme_act_frequency.png` (co-occurrence counts).
Cells backed by fewer than 2 participants are greyed out.

It also compares the two axes **by condition**: `theme_act_sentiment_by_condition.png`
(Sync | Async maps, shared scale) and `theme_act_selected_bars.png` — grouped bars
(white = Sync, black = Async, ±SEM across participants) for a selection of ACT×THEME
pairs (a few named ones plus the largest Sync/Async gaps that have support in both
conditions). `theme_act_crosstab_by_condition.csv` has the per-condition cells.

### Aspect 3 — interpretable clustering of interviews

```bash
python -m src.analysis.participant_clustering
```

Groups interviews by a small, **named** "experience signature" (avoids clustering a
huge redundant matrix — the curse-of-dimensionality trap). Each interview →
7 features: 4 affective dimensions (mean SenticNet/Hourglass `sentics`),
`embodiment` (perceive/feel/body share), `reflection` (think/say share), and
`valence` (RoBERTa). Two-step validated pipeline: standardize → choose k by
**silhouette** → **k-means** (baseline) + **Gaussian Mixture** (soft) → **bootstrap
stability** (adjusted Rand) → **PCA** view.

Outputs: `cluster_profiles.png` (cluster × feature z-scores — each cluster is
describable in a sentence), `cluster_pca.png` (2-D, coloured by cluster, shape by
condition), `cluster_features.csv`, `cluster_validation.json` (k selection, stability,
cluster × Sync/Async).

> Meaningful at **n ≳ 30 participants** (expect 2–4 clusters)

## License

MIT License - See LICENSE file for details.

## Support

For issues or questions, please contact bruno.herbelin@epfl.ch.

---

*Last updated: 2026-07-05*
*Maintainer: Interview Analysis Team*