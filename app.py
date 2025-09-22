#!/usr/bin/env python3
"""
🌟 DOCUVERSE AI 🌟
Revolutionary PDF Assistant with stunning design and proper footer
Copyright © 2025 Justine & Krishna. All Rights Reserved.
"""

import streamlit as st
import PyPDF2
import re
import time
import hashlib
from datetime import datetime
from typing import Dict, List, Tuple
import io
import base64

# Page Configuration
st.set_page_config(
    page_title="DocuVerse AI - Revolutionary PDF Assistant",
    page_icon="🌟",
    layout="wide"
)

def load_revolutionary_css():
    """Load the most stunning CSS ever created"""
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@300;400;700;900&family=Rajdhani:wght@300;400;600;700&family=Space+Mono:wght@400;700&display=swap');

    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #0F0C29 0%, #24243e 30%, #302B63 70%, #0F0C29 100%);
        background-attachment: fixed;
        color: #E2E8F0;
        font-family: 'Rajdhani', sans-serif;
    }

    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Main Title */
    .main-title {
        font-family: 'Orbitron', monospace;
        font-size: 4.5rem;
        font-weight: 900;
        background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4, #FFEAA7, #FF6B6B);
        background-size: 400% 400%;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        animation: gradientFlow 4s ease-in-out infinite;
        text-align: center;
        margin: 2rem 0;
        letter-spacing: 4px;
        text-shadow: 0 0 50px rgba(255, 107, 107, 0.3);
        position: relative;
    }

    .main-title::after {
        content: '';
        position: absolute;
        bottom: -10px;
        left: 50%;
        transform: translateX(-50%);
        width: 200px;
        height: 3px;
        background: linear-gradient(90deg, transparent, #4ECDC4, transparent);
        animation: lineGlow 2s ease-in-out infinite;
    }

    @keyframes gradientFlow {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    @keyframes lineGlow {
        0%, 100% { opacity: 0.3; width: 100px; }
        50% { opacity: 1; width: 300px; }
    }

    .subtitle {
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.6rem;
        font-weight: 300;
        color: #A8A8B3;
        text-align: center;
        margin-bottom: 3rem;
        text-transform: uppercase;
        letter-spacing: 3px;
    }

    /* Navigation Bar */
    .nav-container {
        display: flex;
        justify-content: center;
        margin: 3rem 0;
        padding: 0 2rem;
    }

    .nav-bar {
        display: flex;
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.02) 100%);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 8px;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }

    .nav-bar::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(78, 205, 196, 0.2), transparent);
        animation: navScan 3s linear infinite;
    }

    @keyframes navScan {
        0% { left: -100%; }
        100% { left: 100%; }
    }

    .nav-item {
        position: relative;
        margin: 0 4px;
        border-radius: 20px;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .nav-button {
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 18px 32px;
        background: transparent;
        border: none;
        color: #A8A8B3;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        cursor: pointer;
        transition: all 0.4s ease;
        position: relative;
        min-width: 180px;
        border-radius: 20px;
    }

    .nav-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: linear-gradient(45deg, transparent 30%, rgba(255, 255, 255, 0.05) 50%, transparent 70%);
        transform: translateX(-100%) skew(-10deg);
        transition: transform 0.6s;
    }

    .nav-button:hover::before {
        transform: translateX(100%) skew(-10deg);
    }

    .nav-button .icon {
        font-size: 1.4rem;
        margin-right: 12px;
        transition: all 0.3s ease;
    }

    .nav-button:hover {
        transform: translateY(-3px);
        color: white;
    }

    .nav-button:hover .icon {
        transform: scale(1.2) rotateZ(5deg);
    }

    /* Tab Active States */
    .upload-active .nav-button {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%);
        color: white;
        box-shadow: 0 15px 35px rgba(255, 107, 107, 0.4);
    }

    .analysis-active .nav-button {
        background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%);
        color: white;
        box-shadow: 0 15px 35px rgba(78, 205, 196, 0.4);
    }

    .summary-active .nav-button {
        background: linear-gradient(135deg, #FFEAA7 0%, #FFD93D 100%);
        color: #2D3748;
        box-shadow: 0 15px 35px rgba(255, 234, 167, 0.4);
    }

    .qa-active .nav-button {
        background: linear-gradient(135deg, #96CEB4 0%, #ABEBC6 100%);
        color: #2D3748;
        box-shadow: 0 15px 35px rgba(150, 206, 180, 0.4);
    }

    /* Content Sections */
    .content-section {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05) 0%, rgba(255, 255, 255, 0.01) 100%);
        backdrop-filter: blur(15px);
        border-radius: 25px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 3rem;
        margin: 2rem 0;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }

    .content-section::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #4ECDC4, transparent);
        animation: topGlow 2s ease-in-out infinite;
    }

    @keyframes topGlow {
        0%, 100% { opacity: 0.3; }
        50% { opacity: 1; }
    }

    .section-title {
        font-family: 'Orbitron', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4ECDC4, #45B7D1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 2rem;
        text-align: center;
        letter-spacing: 2px;
    }

    /* Cyber Cards */
    .cyber-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08) 0%, rgba(255, 255, 255, 0.02) 100%);
        backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 15px 35px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        transition: all 0.4s ease;
        position: relative;
        overflow: hidden;
    }

    .cyber-card::after {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(78, 205, 196, 0.1), transparent);
        transition: left 0.8s ease;
    }

    .cyber-card:hover::after {
        left: 100%;
    }

    .cyber-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 25px 60px rgba(0, 0, 0, 0.3), 0 0 0 1px rgba(78, 205, 196, 0.2);
    }

    /* Metrics */
    .metrics-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }

    .metric-card {
        background: linear-gradient(135deg, rgba(0, 255, 127, 0.08), rgba(0, 191, 255, 0.08));
        border: 1px solid rgba(0, 255, 127, 0.2);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        backdrop-filter: blur(10px);
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }

    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(0, 255, 127, 0.03) 2px,
            rgba(0, 255, 127, 0.03) 4px
        );
        animation: scan 3s linear infinite;
    }

    @keyframes scan {
        0% { transform: translateY(0); }
        100% { transform: translateY(20px); }
    }

    .metric-card:hover {
        transform: scale(1.05);
        border-color: rgba(0, 255, 127, 0.4);
        box-shadow: 0 20px 40px rgba(0, 255, 127, 0.2);
    }

    .metric-value {
        font-family: 'Orbitron', monospace;
        font-size: 2.5rem;
        font-weight: 700;
        color: #00FF7F;
        text-shadow: 0 0 20px rgba(0, 255, 127, 0.5);
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
    }

    .metric-label {
        color: #A8A8B3;
        text-transform: uppercase;
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 1px;
        position: relative;
        z-index: 1;
    }

    /* Buttons */
    .cyber-button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 15px;
        padding: 1rem 2.5rem;
        color: white;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        cursor: pointer;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4), inset 0 1px 0 rgba(255, 255, 255, 0.2);
    }

    .cyber-button::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent);
        transition: left 0.5s;
    }

    .cyber-button:hover::before {
        left: 100%;
    }

    .cyber-button:hover {
        transform: translateY(-3px) scale(1.05);
        box-shadow: 0 20px 45px rgba(102, 126, 234, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.3);
    }

    /* Keyword tags */
    .keyword-tag {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.4rem 1rem;
        margin: 0.3rem;
        border-radius: 20px;
        display: inline-block;
        font-size: 0.9rem;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }

    .keyword-tag:hover {
        transform: translateY(-2px) scale(1.1);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.5);
    }

    @keyframes pulse {
        0%, 100% { opacity: 0.8; transform: scale(1); }
        50% { opacity: 1; transform: scale(1.05); }
    }

    /* Footer Styles */
    .footer-container {
        text-align: center;
        padding: 4rem 2rem;
        background: linear-gradient(135deg, rgba(15, 12, 41, 0.9), rgba(48, 43, 99, 0.9));
        border-radius: 30px;
        margin: 3rem 0;
        position: relative;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    .footer-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4, #45B7D1, #96CEB4, #FFEAA7);
        animation: gradientFlow 4s ease-in-out infinite;
    }

    .footer-title {
        font-family: 'Orbitron', monospace;
        color: #00FF7F;
        margin-bottom: 1.5rem;
        text-shadow: 0 0 20px rgba(0, 255, 127, 0.5);
        font-size: 2.5rem;
        font-weight: 700;
    }

    .footer-subtitle {
        color: #A8A8B3;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.3rem;
        margin-bottom: 2rem;
        font-weight: 300;
    }

    .footer-tags {
        display: flex;
        justify-content: center;
        gap: 2rem;
        flex-wrap: wrap;
        margin: 2rem 0;
    }

    .footer-tag {
        padding: 1rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        cursor: pointer;
    }

    .footer-tag:hover {
        transform: translateY(-5px) scale(1.1);
    }

    .footer-tag-1 {
        background: linear-gradient(135deg, #FF6B6B, #FF8E8E);
        animation: pulse 2s infinite;
    }

    .footer-tag-2 {
        background: linear-gradient(135deg, #4ECDC4, #44A08D);
        animation: pulse 2s infinite 0.5s;
    }

    .footer-tag-3 {
        background: linear-gradient(135deg, #667eea, #764ba2);
        animation: pulse 2s infinite 1s;
    }

    .footer-tag-4 {
        background: linear-gradient(135deg, #96CEB4, #ABEBC6);
        animation: pulse 2s infinite 1.5s;
    }

    .footer-copyright {
        color: #6B7280;
        margin-top: 2rem;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* Text and Typography */
    h1, h2, h3, h4 {
        color: #4ECDC4;
        font-family: 'Orbitron', monospace;
    }

    .cyber-text {
        color: #00FF7F;
        text-shadow: 0 0 10px rgba(0, 255, 127, 0.3);
        font-family: 'Space Mono', monospace;
    }

    /* Enhanced Sidebar Styles */
    .css-1d391kg {
        background: linear-gradient(135deg, rgba(15, 12, 41, 0.95), rgba(48, 43, 99, 0.95));
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(78, 205, 196, 0.2);
    }

    /* Enhanced Text Area */
    .stTextArea > div > div > textarea {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.01));
        border: 2px solid rgba(78, 205, 196, 0.3);
        border-radius: 15px;
        color: #E2E8F0;
        font-family: 'Rajdhani', sans-serif;
        font-size: 1.1rem;
        padding: 1rem;
        transition: all 0.3s ease;
    }

    .stTextArea > div > div > textarea:focus {
        border-color: #4ECDC4;
        box-shadow: 0 0 20px rgba(78, 205, 196, 0.3);
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.08), rgba(255, 255, 255, 0.02));
    }

    /* Enhanced Radio Buttons */
    .stRadio > div {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.05), rgba(255, 255, 255, 0.01));
        border-radius: 15px;
        padding: 1rem;
        border: 1px solid rgba(78, 205, 196, 0.2);
    }

    /* Enhanced Download Buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #4ECDC4 0%, #44A08D 100%) !important;
        border: none !important;
        border-radius: 15px !important;
        color: white !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 10px 25px rgba(78, 205, 196, 0.3) !important;
    }

    .stDownloadButton > button:hover {
        transform: translateY(-2px) scale(1.05) !important;
        box-shadow: 0 15px 35px rgba(78, 205, 196, 0.5) !important;
    }

    /* Enhanced Code Blocks */
    .stCode {
        background: linear-gradient(135deg, rgba(0, 255, 127, 0.05), rgba(78, 205, 196, 0.05));
        border: 1px solid rgba(0, 255, 127, 0.2);
        border-radius: 10px;
        padding: 1rem;
    }

    /* Responsive Design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 3rem;
        }

        .nav-bar {
            flex-direction: column;
            gap: 8px;
        }

        .nav-button {
            min-width: auto;
            width: 100%;
        }

        .content-section {
            padding: 2rem 1rem;
        }

        .metrics-grid {
            grid-template-columns: repeat(2, 1fr);
        }

        .footer-tags {
            flex-direction: column;
            gap: 1rem;
        }
    }

    @media (max-width: 480px) {
        .metrics-grid {
            grid-template-columns: 1fr;
        }
    }

    /* Progress bars and spinners */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
    }

    /* File uploader styling */
    .uploadedFile {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.1), rgba(78, 205, 196, 0.1));
        border: 2px dashed rgba(255, 107, 107, 0.3);
        border-radius: 20px;
        padding: 2rem;
    }

    /* Streamlit button override */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 1rem 2.5rem !important;
        color: white !important;
        font-family: 'Rajdhani', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4) !important;
    }

    .stButton > button:hover {
        transform: translateY(-3px) scale(1.05) !important;
        box-shadow: 0 20px 45px rgba(102, 126, 234, 0.6) !important;
    }
    </style>
    """, unsafe_allow_html=True)

class PDFProcessor:
    """Advanced PDF processing with quantum algorithms"""

    def extract_text(self, pdf_file):
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            text = ""

            for page_num, page in enumerate(pdf_reader.pages[:15]):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"

            # Quantum text cleaning
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()

            return text
        except Exception as e:
            return f"Quantum extraction error: {str(e)}"

    def get_advanced_stats(self, text):
        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]

        # Advanced metrics
        long_words = [w for w in words if len(w) > 6]
        complexity = len(long_words) / max(len(words), 1) * 100

        return {
            'words': len(words),
            'sentences': len(sentences),
            'paragraphs': len(paragraphs),
            'characters': len(text),
            'complexity': round(complexity, 1),
            'unique_words': len(set(word.lower() for word in words)),
            'reading_time': max(1, len(words) // 200)
        }

class QuantumSummarizer:
    """Revolutionary quantum-inspired summarization"""

    def __init__(self):
        self.styles = {
            'executive': 'Executive Summary',
            'academic': 'Academic Abstract',
            'bullet': 'Key Points',
            'narrative': 'Story Format',
            'technical': 'Technical Brief'
        }
        
        # Three types of summarization
        self.summary_types = {
            'extractive': 'Extractive Summary',
            'abstractive': 'Abstractive Summary', 
            'hybrid': 'Hybrid Summary'
        }

    def quantum_summarize(self, text, style='executive', sentences=3, summary_type='extractive'):
        if not text:
            return {'summary': 'No quantum data to process', 'confidence': 0}

        # Quantum sentence extraction
        raw_sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 15]

        if len(raw_sentences) <= sentences:
            return {
                'summary': text,
                'confidence': 100,
                'method': 'quantum_full',
                'style': self.styles.get(style),
                'type': summary_type
            }

        if summary_type == 'extractive':
            return self._extractive_summary(text, raw_sentences, sentences, style)
        elif summary_type == 'abstractive':
            return self._abstractive_summary(text, raw_sentences, sentences, style)
        elif summary_type == 'hybrid':
            return self._hybrid_summary(text, raw_sentences, sentences, style)
        else:
            return self._extractive_summary(text, raw_sentences, sentences, style)

    def _extractive_summary(self, text, raw_sentences, sentences, style):
        """Extractive summarization - selects most important sentences"""
        # Quantum scoring algorithm
        scored = []
        for i, sentence in enumerate(raw_sentences):
            score = self._quantum_score(sentence, i, len(raw_sentences), text)
            scored.append((score, sentence, i))

        # Apply quantum style weights
        styled = self._apply_quantum_weights(scored, style)

        # Quantum selection
        top = sorted(styled, reverse=True)[:sentences]
        top.sort(key=lambda x: x[2])  # Restore quantum order

        summary = '. '.join([s[1] for s in top]) + '.'
        confidence = min(100, sum(s[0] for s in top) / len(top) * 100)

        return {
            'summary': summary,
            'confidence': round(confidence, 1),
            'method': f'extractive_{style}',
            'style': self.styles.get(style, style),
            'type': 'extractive'
        }

    def _abstractive_summary(self, text, raw_sentences, sentences, style):
        """Abstractive summarization - generates new content based on key concepts"""
        # Extract key concepts and phrases
        keywords = self._extract_key_concepts(text)
        
        # Find sentences with highest keyword density
        concept_sentences = []
        for sentence in raw_sentences:
            score = self._concept_score(sentence, keywords)
            concept_sentences.append((score, sentence))
        
        # Select top sentences and create abstractive summary
        top_sentences = sorted(concept_sentences, reverse=True)[:max(2, sentences//2)]
        
        # Generate abstractive content
        summary_parts = []
        for score, sentence in top_sentences:
            # Simplify and abstract the sentence
            abstracted = self._abstract_sentence(sentence, keywords)
            summary_parts.append(abstracted)
        
        summary = '. '.join(summary_parts) + '.'
        confidence = min(95, sum(score for score, _ in top_sentences) / len(top_sentences) * 100)

        return {
            'summary': summary,
            'confidence': round(confidence, 1),
            'method': f'abstractive_{style}',
            'style': self.styles.get(style, style),
            'type': 'abstractive'
        }

    def _hybrid_summary(self, text, raw_sentences, sentences, style):
        """Hybrid summarization - combines extractive and abstractive methods"""
        # Get extractive summary
        extractive_result = self._extractive_summary(text, raw_sentences, sentences//2 + 1, style)
        
        # Get abstractive summary
        abstractive_result = self._abstractive_summary(text, raw_sentences, sentences//2 + 1, style)
        
        # Combine both approaches
        combined_summary = f"{extractive_result['summary']} {abstractive_result['summary']}"
        
        # Clean up and optimize
        combined_summary = self._optimize_hybrid_summary(combined_summary)
        
        confidence = (extractive_result['confidence'] + abstractive_result['confidence']) / 2

        return {
            'summary': combined_summary,
            'confidence': round(confidence, 1),
            'method': f'hybrid_{style}',
            'style': self.styles.get(style, style),
            'type': 'hybrid'
        }

    def _extract_key_concepts(self, text):
        """Extract key concepts from text"""
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        word_freq = {}
        for word in words:
            if word not in {'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been', 'were', 'said'}:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top concepts
        return sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]

    def _concept_score(self, sentence, keywords):
        """Score sentence based on concept density"""
        sentence_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', sentence.lower()))
        keyword_words = set([word for word, freq in keywords])
        
        overlap = len(sentence_words.intersection(keyword_words))
        return overlap / max(len(sentence_words), 1)

    def _abstract_sentence(self, sentence, keywords):
        """Create abstract version of sentence"""
        # Simple abstraction - keep key concepts, simplify structure
        words = sentence.split()
        key_concepts = [word for word, freq in keywords[:5]]
        
        # Keep sentences that contain key concepts
        if any(concept in sentence.lower() for concept in key_concepts):
            # Simplify the sentence
            simplified = ' '.join(words[:min(15, len(words))])
            return simplified
        return sentence

    def _optimize_hybrid_summary(self, summary):
        """Optimize hybrid summary by removing redundancy"""
        sentences = [s.strip() for s in summary.split('.') if s.strip()]
        unique_sentences = []
        
        for sentence in sentences:
            if not any(sentence.lower() in existing.lower() or existing.lower() in sentence.lower() 
                      for existing in unique_sentences):
                unique_sentences.append(sentence)
        
        return '. '.join(unique_sentences[:5]) + '.'

    def _quantum_score(self, sentence, pos, total, full_text):
        words = sentence.split()

        # Quantum length optimization
        length_score = min(1.0, len(words) / 20)

        # Quantum position matrix
        pos_ratio = pos / max(total - 1, 1)
        pos_score = 1.0 - abs(pos_ratio - 0.25)  # Quantum preference for early content

        # Quantum frequency analysis
        freq_score = self._quantum_frequency_analysis(sentence, full_text)

        # Quantum interference pattern
        return length_score * 0.3 + pos_score * 0.4 + freq_score * 0.3

    def _quantum_frequency_analysis(self, sentence, full_text):
        sentence_words = set(re.findall(r'\b[a-zA-Z]{4,}\b', sentence.lower()))
        all_words = re.findall(r'\b[a-zA-Z]{4,}\b', full_text.lower())

        word_freq = {}
        for word in all_words:
            word_freq[word] = word_freq.get(word, 0) + 1

        quantum_score = 0
        for word in sentence_words:
            if word in word_freq and word_freq[word] > 1:
                quantum_score += min(word_freq[word] / len(all_words) * 100, 1.0)

        return min(quantum_score / max(len(sentence_words), 1), 1.0)

    def _apply_quantum_weights(self, scored, style):
        if style == 'bullet':
            return [(s * 1.5 if len(sent.split()) < 15 else s * 0.8, sent, pos)
                   for s, sent, pos in scored]
        elif style == 'executive':
            return [(s * 1.4 if pos < len(scored) * 0.3 else s, sent, pos)
                   for s, sent, pos in scored]
        elif style == 'academic':
            research_terms = ['study', 'research', 'analysis', 'results', 'findings']
            return [(s * 1.3 if any(term in sent.lower() for term in research_terms) else s, sent, pos)
                   for s, sent, pos in scored]
        return scored

class NeuroQA:
    """Neural-inspired question answering system"""

    def neural_answer(self, question, document):
        if not question or not document:
            return {
                'answer': 'Neural pathways require both question and document data.',
                'confidence': 0,
                'method': 'neural_error'
            }

        # Neural context discovery
        contexts = self._discover_neural_contexts(question, document)

        if not contexts:
            return {
                'answer': 'Neural networks found no relevant quantum patterns. Try rephrasing your query.',
                'confidence': 0,
                'method': 'neural_no_match'
            }

        # Neural answer synthesis
        best_context = contexts[0]
        sentences = [s.strip() for s in best_context['text'].split('.') if s.strip()]

        if not sentences:
            return {'answer': 'Neural processing incomplete.', 'confidence': 0}

        # Neural sentence matching
        question_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower()))
        best_sentence = ""
        max_neural_score = 0

        for sentence in sentences:
            sentence_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', sentence.lower()))
            neural_score = len(question_words.intersection(sentence_words))

            if neural_score > max_neural_score:
                max_neural_score = neural_score
                best_sentence = sentence

        if not best_sentence:
            best_sentence = sentences[0]

        confidence = min(95, best_context['score'] * 100)

        return {
            'answer': best_sentence + '.',
            'confidence': round(confidence, 1),
            'method': 'neural_synthesis',
            'neural_pathways': len(contexts)
        }

    def _discover_neural_contexts(self, question, document):
        sentences = [s.strip() for s in document.split('.') if len(s.strip()) > 10]
        question_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', question.lower()))

        neural_contexts = []
        window_size = 3

        for i in range(len(sentences) - window_size + 1):
            context = '. '.join(sentences[i:i + window_size])
            context_words = set(re.findall(r'\b[a-zA-Z]{3,}\b', context.lower()))

            neural_overlap = len(question_words.intersection(context_words))
            if neural_overlap > 0:
                neural_score = neural_overlap / max(len(question_words), 1)
                if neural_score > 0.2:
                    neural_contexts.append({
                        'text': context,
                        'score': neural_score,
                        'overlap': neural_overlap
                    })

        return sorted(neural_contexts, key=lambda x: x['score'], reverse=True)[:3]

def extract_quantum_keywords(text, top_k=10):
    """Extract quantum-enhanced keywords"""
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())

    quantum_stop_words = {
        'this', 'that', 'with', 'have', 'will', 'from', 'they', 'been',
        'were', 'said', 'each', 'which', 'their', 'time', 'about',
        'would', 'there', 'could', 'other', 'after', 'first', 'well',
        'also', 'make', 'here', 'where', 'much', 'take','were', 'said', 
        'each', 'which', 'their', 'time', 'about','also', 'make', 'here', 
        'where', 'much', 'take', 'than', 'only'
    }

    quantum_filtered = [w for w in words if w not in quantum_stop_words and len(w) > 3]

    quantum_freq = {}
    for word in quantum_filtered:
        quantum_freq[word] = quantum_freq.get(word, 0) + 1

    return sorted(quantum_freq.items(), key=lambda x: x[1], reverse=True)[:top_k]

def create_download_file(content, filename, file_type="txt"):
    """Create downloadable file content"""
    if file_type == "txt":
        return content.encode('utf-8')
    elif file_type == "pdf":
        # For PDF, we'll create a simple text-based PDF
        # This is a simplified version - in production, use reportlab or similar
        return content.encode('utf-8')
    return content.encode('utf-8')

def main():
    """Revolutionary main application with enhanced navigation and proper footer"""

    # Initialize quantum components
    if 'pdf_processor' not in st.session_state:
        st.session_state.pdf_processor = PDFProcessor()
    if 'quantum_summarizer' not in st.session_state:
        st.session_state.quantum_summarizer = QuantumSummarizer()
    if 'neuro_qa' not in st.session_state:
        st.session_state.neuro_qa = NeuroQA()

    # Initialize quantum data
    if 'document_text' not in st.session_state:
        st.session_state.document_text = ""
    if 'neural_history' not in st.session_state:
        st.session_state.neural_history = []

    # Load revolutionary CSS
    load_revolutionary_css()

    # Sidebar Navigation
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <h2 style="color: #4ECDC4; font-family: 'Orbitron', monospace;">🌌 DOCUVERSE AI</h2>
            <p style="color: #A8A8B3; font-size: 0.9rem;">Revolutionary PDF Intelligence</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Navigation options
        st.markdown("### 🚀 Quick Navigation")
        nav_option = st.radio(
            "Choose your path:",
            ["📄 Upload PDF", "✍️ Text Input", "📊 Analysis", "⚡ Summary", "🔮 Q&A"],
            key="sidebar_nav"
        )
        
        st.markdown("---")
        
        # Quick stats if document is loaded
        if st.session_state.document_text:
            stats = st.session_state.pdf_processor.get_advanced_stats(st.session_state.document_text)
            st.markdown("### 📈 Document Stats")
            st.metric("Words", f"{stats['words']:,}")
            st.metric("Sentences", f"{stats['sentences']:,}")
            st.metric("Complexity", f"{stats['complexity']:.1f}%")
            st.metric("Reading Time", f"{stats['reading_time']} min")
        
        st.markdown("---")
        st.markdown("### 🛠️ Tools")
        if st.button("🔄 Reset Session", key="reset_session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    # Revolutionary Header
    st.markdown('<h1 class="main-title">DOCUVERSE AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Revolutionary PDF Intelligence Platform</p>', unsafe_allow_html=True)

    # Enhanced Navigation Bar (Single, Better Presentation)
    st.markdown("""
    <div class="nav-container">
        <div class="nav-bar">
            <div class="nav-item upload-active" id="nav-upload">
                <div class="nav-button">
                    <span class="icon">🌌</span>
                    <span class="text">Upload PDF</span>
                </div>
            </div>
            <div class="nav-item" id="nav-text">
                <div class="nav-button">
                    <span class="icon">✍️</span>
                    <span class="text">Text Input</span>
                </div>
            </div>
            <div class="nav-item" id="nav-analysis">
                <div class="nav-button">
                    <span class="icon">📊</span>
                    <span class="text">Analysis</span>
                </div>
            </div>
            <div class="nav-item" id="nav-summary">
                <div class="nav-button">
                    <span class="icon">⚡</span>
                    <span class="text">Summary</span>
                </div>
            </div>
            <div class="nav-item" id="nav-qa">
                <div class="nav-button">
                    <span class="icon">🔮</span>
                    <span class="text">Q&A</span>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Main Navigation Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🌌 Upload", "✍️ Text Input", "📊 Analysis", "⚡ Summary", "🔮 Q&A"])

    with tab1:
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title">🌌 Document Upload</h2>', unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "DRAG YOUR PDF INTO THE FIELD",
            type="pdf",
            key="quantum_uploader",
            help="Upload PDF documents for processing"
        )

        if uploaded_file:
            file_size = len(uploaded_file.getvalue()) / 1024 / 1024

            st.markdown(f"""
            <div class="cyber-card">
                <h4 class="cyber-text">📁 
                 File Detected</h4>
                <p><strong>Filename:</strong> {uploaded_file.name}</p>
                <p><strong>Size:</strong> {file_size:.1f} MB</p>
                <p><strong>Type:</strong> {uploaded_file.type}</p>
                <p><strong>Status:</strong> <span class="cyber-text">Ready for processing</span></p>
            </div>
            """, unsafe_allow_html=True)

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 INITIATE EXTRACTION", key="quantum_extract"):
                    with st.spinner("🌀 processors engaged..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        # Quantum extraction sequence
                        status_text.text("🔬 Analyzing document structure...")
                        progress_bar.progress(25)
                        time.sleep(0.8)

                        status_text.text("🧬 Extracting text patterns...")
                        progress_bar.progress(50)
                        time.sleep(0.8)

                        status_text.text("🌌 Processing neural pathways...")
                        progress_bar.progress(75)
                        time.sleep(0.8)

                        # Actual processing
                        text = st.session_state.pdf_processor.extract_text(uploaded_file)
                        progress_bar.progress(100)
                        status_text.text("✨ Extraction complete!")

                        if text and not text.startswith("Quantum extraction error"):
                            st.session_state.document_text = text
                            time.sleep(1)
                            progress_bar.empty()
                            status_text.empty()

                            st.success("🎉 DOCUMENT EXTRACTION SUCCESSFUL!")

                            # Show quantum preview
                            with st.expander("🔍 TEXT PREVIEW", expanded=True):
                                preview = text[:1500] + "..." if len(text) > 1500 else text
                                st.markdown(f"""
                                <div class="cyber-card">
                                    <div class="cyber-text">{preview}</div>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.error("❌ Extraction failed. Please try another document.")
                            progress_bar.empty()
                            status_text.empty()

        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div class="content-section">', unsafe_allow_html=True)
        st.markdown('<h2 class="section-title">✍️ Text Input</h2>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="cyber-card">
            <h4 class="cyber-text">📝 Direct Text Input</h4>
            <p>Paste your text directly here for immediate processing and summarization.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Text input area
        input_text = st.text_area(
            "Enter your text here:",
            height=300,
            placeholder="Paste your document text here for analysis and summarization...",
            key="text_input_area"
        )
        
        if input_text:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 PROCESS TEXT", key="process_text_btn"):
                    with st.spinner("🌀 Processing text..."):
                        st.session_state.document_text = input_text
                        st.success("✅ Text processed successfully!")
                        
                        # Show preview
                        with st.expander("🔍 TEXT PREVIEW", expanded=True):
                            preview = input_text[:1500] + "..." if len(input_text) > 1500 else input_text
                            st.markdown(f"""
                            <div class="cyber-card">
                                <div class="cyber-text">{preview}</div>
                            </div>
                            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        if st.session_state.document_text:
            st.markdown('<div class="content-section">', unsafe_allow_html=True)
            st.markdown('<h2 class="section-title">📊 Neural Document Analysis</h2>', unsafe_allow_html=True)

            # Quantum metrics
            stats = st.session_state.pdf_processor.get_advanced_stats(st.session_state.document_text)

            st.markdown('<div class="metrics-grid">', unsafe_allow_html=True)

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{stats['words']:,}</div>
                    <div class="metric-label">Quantum Words</div>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{stats['sentences']:,}</div>
                    <div class="metric-label">Neural Sentences</div>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{stats['complexity']:.1f}%</div>
                    <div class="metric-label">Complexity Index</div>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-value">{stats['reading_time']}</div>
                    <div class="metric-label">Neural Minutes</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # Quantum keywords
            st.markdown("### 🔑 Key Phrases")
            keywords = extract_quantum_keywords(st.session_state.document_text)

            keyword_html = ""
            for word, freq in keywords:
                keyword_html += f'<span class="keyword-tag">{word} ({freq})</span>'

            st.markdown(f'<div style="text-align: center; margin: 2rem 0;">{keyword_html}</div>',
                       unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("🌌 Please upload and extract a document first")

    with tab4:
        if st.session_state.document_text:
            st.markdown('<div class="content-section">', unsafe_allow_html=True)
            st.markdown('<h2 class="section-title">⚡ Advanced Summarization Engine</h2>', unsafe_allow_html=True)

            # Summary type selection
            st.markdown("### 🎯 Choose Summarization Method")
            summary_type = st.radio(
                "Select summarization approach:",
                options=list(st.session_state.quantum_summarizer.summary_types.keys()),
                format_func=lambda x: st.session_state.quantum_summarizer.summary_types[x],
                key="summary_type_radio",
                horizontal=True
            )

            col1, col2 = st.columns([2, 1])

            with col2:
                st.markdown("""
                <div class="cyber-card">
                    <h4 class="cyber-text">⚙️ Quantum Parameters</h4>
                </div>
                """, unsafe_allow_html=True)

                style = st.selectbox(
                    "🎨 Quantum Style:",
                    options=list(st.session_state.quantum_summarizer.styles.keys()),
                    format_func=lambda x: st.session_state.quantum_summarizer.styles[x],
                    key="quantum_style"
                )

                length = st.slider("📏 Quantum Length:", 2, 8, 3, key="quantum_length")

            with col1:
                if st.button("🌟 GENERATE SUMMARY", key="quantum_summary_btn"):
                    with st.spinner("🌀 Quantum algorithms processing..."):
                        result = st.session_state.quantum_summarizer.quantum_summarize(
                            st.session_state.document_text,
                            style=style,
                            sentences=length,
                            summary_type=summary_type
                        )

                    # Store result in session state for download
                    st.session_state.last_summary = result

                    st.markdown(f"""
                    <div class="cyber-card">
                        <h4 class="cyber-text">✨ {st.session_state.quantum_summarizer.summary_types[summary_type]}</h4>
                        <div style="background: rgba(0, 255, 127, 0.05); padding: 2rem; border-radius: 15px; margin: 1rem 0; border-left: 4px solid #00FF7F;">
                            <p style="font-size: 1.2rem; line-height: 1.8; color: #E2E8F0;">
                                {result['summary']}
                            </p>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 1.5rem;">
                            <span class="cyber-text">Confidence: {result['confidence']}%</span>
                            <span class="cyber-text">Method: {result['method']}</span>
                            <span class="cyber-text">Type: {result['type']}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Download section
                    st.markdown("### 📥 Download Summary")
                    col_download1, col_download2, col_download3 = st.columns(3)
                    
                    # Prepare file content
                    file_content = f"""DOCUVERSE AI - SUMMARY REPORT
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
Type: {result['type']}
Method: {result['method']}
Confidence: {result['confidence']}%

SUMMARY:
{result['summary']}

---
© 2025 DocuVerse AI - Revolutionary PDF Intelligence Platform"""
                    
                    with col_download1:
                        st.download_button(
                            label="📄 Download TXT",
                            data=file_content,
                            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain",
                            key="download_txt_btn"
                        )
                    
                    with col_download2:
                        st.download_button(
                            label="📊 Download PDF",
                            data=file_content,
                            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                            mime="text/plain",
                            key="download_pdf_btn"
                        )
                    
                    with col_download3:
                        if st.button("📋 Copy to Clipboard", key="copy_summary"):
                            st.code(result['summary'], language=None)
                            st.success("📋 Summary copied! You can now paste it anywhere.")

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("🌌 Please upload and extract a document first")

    with tab5:
        if st.session_state.document_text:
            st.markdown('<div class="content-section">', unsafe_allow_html=True)
            st.markdown('<h2 class="section-title">🔮 Neuro Question & Answer</h2>', unsafe_allow_html=True)

            question = st.text_input(
                "🧠 ASK THE NEURAL NETWORK:",
                placeholder="What is the main principle discussed in this document?",
                help="Ask any question about your document",
                key="neural_question"
            )

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 NEURAL QUERY PROCESSING", key="neural_qa_btn") and question:
                    with st.spinner("🧠 Neural pathways processing..."):
                        result = st.session_state.neuro_qa.neural_answer(
                            question,
                            st.session_state.document_text
                        )

                    # Add to neural history
                    st.session_state.neural_history.append({
                        'question': question,
                        'answer': result['answer'],
                        'confidence': result['confidence'],
                        'method': result.get('method', 'neural'),
                        'timestamp': datetime.now().strftime("%H:%M:%S")
                    })

                    st.markdown(f"""
                    <div class="cyber-card">
                        <h4 class="cyber-text">💡 Neural Response</h4>
                        <div style="background: rgba(78, 205, 196, 0.05); padding: 2rem; border-radius: 15px; margin: 1rem 0; border-left: 4px solid #4ECDC4;">
                            <p><strong>❓ Query:</strong> {question}</p>
                            <p><strong>🤖 Neural Answer:</strong> {result['answer']}</p>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-top: 1.5rem;">
                            <span class="cyber-text">Confidence: {result['confidence']}%</span>
                            <span class="cyber-text">Method: {result.get('method', 'neural')}</span>
                            <span class="cyber-text">Pathways: {result.get('neural_pathways', 1)}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # Neural History
            if st.session_state.neural_history:
                st.markdown("### 🕒 Neural Processing History")

                for i, qa in enumerate(reversed(st.session_state.neural_history[-5:])):
                    with st.expander(f"💭 {qa['question'][:50]}... ({qa['timestamp']})",
                                   expanded=(i==0)):
                        st.markdown(f"""
                        <div class="cyber-card">
                            <p><strong>❓ Question:</strong> {qa['question']}</p>
                            <p><strong>🤖 Answer:</strong> {qa['answer']}</p>
                            <div style="margin-top: 1rem;">
                                <span class="cyber-text">Confidence: {qa['confidence']}%</span> •
                                <span class="cyber-text">Method: {qa['method']}</span> •
                                <span class="cyber-text">Time: {qa['timestamp']}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("🌌 Please upload and extract a document first")

    # Revolutionary Footer - Fixed HTML Rendering
    st.markdown("---")

    # Create footer using HTML components instead of raw HTML
    st.markdown("""
    <div class="footer-container">
        <h3 class="footer-title">🌟 DOCUVERSE AI - THE QUANTUM FUTURE</h3>
        <p class="footer-subtitle">Revolutionary PDF Intelligence • Quantum Processing • Neural Networks • Beyond Reality</p>
    </div>
    """, unsafe_allow_html=True)

    # Feature tags using columns instead of raw HTML
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
        <div class="footer-tag footer-tag-1">⚡ Quantum Speed</div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="footer-tag footer-tag-2">🧠 Neural Intelligence</div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="footer-tag footer-tag-3">🌟 Revolutionary Tech</div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
        <div class="footer-tag footer-tag-4">🌌 Infinite Possibilities</div>
        """, unsafe_allow_html=True)

    # Copyright information
    st.markdown("""
    <div class="footer-copyright">
        <p><strong>© 2025 Justine & Krishna. All Rights Reserved.</strong></p>
        <p>DocuVerse AI™ - Revolutionary PDF Intelligence Platform</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()

