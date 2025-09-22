# DocuVerse AI - Revolutionary PDF Assistant

A streamlined, modern PDF analysis and summarization tool built with Streamlit and advanced AI technologies.

## Features

- **Document Upload & Processing**: Upload PDF files and extract text content
- **Smart Analysis**: Get detailed statistics including word count, sentences, complexity score, and estimated reading time
- **Intelligent Summarization**: Generate summaries with multiple styles (Executive, Technical, Key Points, Narrative)
- **AI Question & Answer**: Ask questions about your document and get intelligent responses
- **Modern UI**: Clean, icon-based interface with Font Awesome icons instead of emojis

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation & Launch

1. **Clone or download the project**
   ```bash
   cd TEXT-SUMMARIZER-CHATBOT-main
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Launch the application**

   **On Linux/Mac:**
   ```bash
   ./run.sh
   ```

   **On Windows:**
   ```batch
   run.bat
   ```

   **Or manually:**
   ```bash
   streamlit run app.py
   ```

4. **Open in browser**
   - The application will automatically open at `http://localhost:8501`

## Usage

### 1. Upload Document
- Click on the "Upload" tab
- Select a PDF file using the file uploader
- Click "Extract Text" to process the document

### 2. Analyze Document
- Navigate to the "Analysis" tab
- View document statistics (words, sentences, complexity, reading time)
- See extracted keywords and key terms

### 3. Generate Summary
- Go to the "Summary" tab
- Choose a summary style (Executive, Technical, Key Points, or Narrative)
- Adjust the number of sentences (2-8)
- Click "Generate Summary"

### 4. Ask Questions
- Visit the "Q&A" tab
- Type your question about the document
- Click "Get Answer" to receive an AI-generated response
- View chat history of previous questions and answers

## Technical Features

- **Optimized Performance**: Uses caching to improve speed and reduce processing time
- **Advanced Text Processing**: Intelligent sentence scoring and keyword extraction
- **Responsive Design**: Modern CSS with gradients and smooth animations
- **Font Awesome Icons**: Professional icon set instead of emojis
- **Error Handling**: Robust error handling for file processing and text extraction

## Dependencies

- `streamlit` - Web application framework
- `PyPDF2` - PDF text extraction
- `transformers` - Advanced AI models
- `torch` - Machine learning backend
- `numpy` - Numerical computing
- `pandas` - Data manipulation
- `requests` - HTTP requests

## File Structure

```
TEXT-SUMMARIZER-CHATBOT-main/
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── run.sh             # Unix/Linux/Mac launcher
├── run.bat            # Windows launcher
├── README.md          # This file
├── static/            # Static assets
└── templates/         # HTML templates
```

## System Requirements

- **Operating System**: Windows, macOS, or Linux
- **Python**: 3.8 or higher
- **Memory**: 4GB RAM recommended
- **Storage**: 2GB free space for dependencies
- **Internet**: Required for initial setup and AI model downloads

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install --upgrade -r requirements.txt
   ```

2. **Port Already in Use**
   ```bash
   streamlit run app.py --server.port 8502
   ```

3. **PDF Processing Errors**
   - Ensure PDF is not password-protected
   - Try with a different PDF file
   - Check file size (large files may take longer)

### Performance Tips

- For better performance, ensure you have at least 4GB RAM
- Close other resource-intensive applications
- Use smaller PDF files for faster processing

## Contributing

This is a streamlined version focused on core functionality. The application has been optimized for:
- Clean, maintainable code
- Professional icon-based UI
- Fast performance
- Essential features only

## License

© 2025 Justine & Krishna. All Rights Reserved.

DocuVerse AI™ - Revolutionary PDF Intelligence Platform

---

**Built with ❤️ and AI Technology**