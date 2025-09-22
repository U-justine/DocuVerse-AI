# DocuVerse AI Guide

A modern PDF/text analysis and summarization app built with Streamlit.

## What’s inside
- Document upload and text input (5000-word limit for input text)
- Analysis: word/sentence counts, complexity index, reading-time estimate, and key phrases
- Summarization: styles (Executive, Academic, Bullet/Key Points, Narrative, Technical) and approaches (Extractive, Abstractive, Hybrid)
- Q&A: document-grounded answers only
- Download summaries as TXT or PDF-mime

## How summarization works (in code)
The summarization logic is in `app.py` under the class `QuantumSummarizer`.

- Entry point: `quantum_summarize(text, style, sentences, summary_type)`
  - Chooses a path based on `summary_type`: `extractive`, `abstractive`, or `hybrid`.

- Extractive (`_extractive_summary`):
  1. Split text into sentences and compute a score per sentence via `_quantum_score`.
  2. `_quantum_score` blends:
     - length_score: prefers informative lengths
     - pos_score: favors early sentences (lead-bias for overviews)
     - freq_score: term-frequency emphasis using `_quantum_frequency_analysis` over the whole doc
  3. Style-specific weighting via `_apply_quantum_weights` (e.g., executive favors early content, bullet favors shorter, punchy lines).
  4. Pick top-N scored sentences, restore original order, and join them.

- Abstractive (`_abstractive_summary`):
  1. Extract key concepts with `_extract_key_concepts` (term frequency without common stop words).
  2. Rank sentences by concept density with `_concept_score`.
  3. Create a compressed/abstracted version of the best sentences via `_abstract_sentence` by simplifying structure while preserving key terms.

- Hybrid (`_hybrid_summary`):
  1. Generate an extractive summary (signal preservation).
  2. Generate an abstractive summary (compression and smoothing).
  3. Merge and deduplicate with `_optimize_hybrid_summary` to remove redundancy and cap length.

All paths return a dictionary like:
```
{"summary": str, "confidence": float, "method": str, "type": str}
```

## How Q&A is retrieved
Q&A lives in `NeuroQA` (also in `app.py`). It is strictly document-grounded.

- Input: `question` + `document` (the extracted or pasted text)
- Context retrieval (`_discover_neural_contexts`):
  1. Split the document into sentences and build sliding windows (3-sentence chunks).
  2. Compute lexical overlap between question terms and each chunk.
  3. Keep the highest-overlap chunks sorted by score (top 3).
- Answer synthesis (`neural_answer`):
  1. Take the best chunk and split into sentences.
  2. Select the sentence with the largest word-overlap with the question as the answer (fallbacks to first sentence if needed).
  3. Report a confidence derived from the retrieval score.

This approach ensures answers are drawn only from the provided document text. If no overlap is found, the system responds accordingly.

## Running the app
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Usage flow
1. Document Upload or Text Input
2. Analysis → review stats and key phrases
3. Summary → choose Style + Approach, set Length → Generate Summary → Download
4. Q&A → ask a question → get a grounded answer

## Notes
- The “Engine” UI for external models is disabled by default to maximize portability. The built-in summarizer is fast and offline.
- PDF downloads are provided with text MIME for portability. Use a PDF generator (e.g., ReportLab) if you need fully formatted PDFs.

## File layout
```
DOCUVERSE AI/
├── app.py
├── requirements.txt
├── run.sh / run.bat
└── README.md (this guide)
```

## License
© 2025 Justine & Krishna. All Rights Reserved.
DocuVerse AI – Revolutionary PDF Intelligence Platform
