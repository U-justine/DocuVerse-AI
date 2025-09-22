// DocuSum - Interactive JavaScript for Document Summarizer ChatBot
// Enhanced JavaScript for DocuSum Application

// Global variables
let uploadedFile = null;
let currentFileId = null;
let isProcessing = false;
let chatHistory = [];
let documentText = "";
let currentInputMode = "upload"; // 'upload' or 'text'
let textContent = "";

// DOM elements
const uploadArea = document.getElementById("uploadArea");
const fileInput = document.getElementById("fileInput");
const uploadProgress = document.getElementById("uploadProgress");
const documentInfo = document.getElementById("documentInfo");
const chatMessages = document.getElementById("chatMessages");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");
const typingIndicator = document.getElementById("typingIndicator");
const statusIndicator = document.getElementById("statusIndicator");
const statusText = document.getElementById("statusText");
const textInput = document.getElementById("textInput");
const wordCount = document.getElementById("wordCount");
const charCount = document.getElementById("charCount");
const readTime = document.getElementById("readTime");
const analyzeBtn = document.getElementById("analyzeBtn");
const textInputPanel = document.getElementById("textInputPanel");
const uploadPanel = document.getElementById("uploadPanel");

// Initialize the application
document.addEventListener("DOMContentLoaded", function () {
  initializeApp();
  setupEventListeners();
  updateNavOnScroll();
  initializeAnimations();
  initializeSmoothScrolling();
});

function initializeApp() {
  // Set initial status
  updateStatus("ready", "Ready to help");

  // Initialize text input if available
  if (textInput) {
    updateTextStats();
  }

  // Add welcome message animation
  setTimeout(() => {
    const welcomeMessage = document.querySelector(".assistant-message");
    if (welcomeMessage) {
      welcomeMessage.style.animation = "fadeInUp 0.6s ease-out";
    }
  }, 500);

  // Initialize notification styles
  initializeNotificationStyles();
}

function setupEventListeners() {
  // File upload events
  if (uploadArea) {
    uploadArea.addEventListener("click", () => fileInput && fileInput.click());
    uploadArea.addEventListener("dragover", handleDragOver);
    uploadArea.addEventListener("dragleave", handleDragLeave);
    uploadArea.addEventListener("drop", handleDrop);
  }

  if (fileInput) {
    fileInput.addEventListener("change", handleFileSelect);
  }

  // Chat events
  if (chatInput) {
    chatInput.addEventListener("keypress", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
  }

  // Text input events
  if (textInput) {
    textInput.addEventListener("input", updateTextStats);
    textInput.addEventListener("paste", () => {
      setTimeout(updateTextStats, 100);
    });
  }

  // Mobile menu toggle
  const navToggle = document.querySelector(".nav-toggle");
  const navMenu = document.querySelector(".nav-menu");

  if (navToggle && navMenu) {
    navToggle.addEventListener("click", () => {
      navMenu.classList.toggle("active");
    });
  }
}

// File upload handlers
function handleDragOver(e) {
  e.preventDefault();
  if (uploadArea) uploadArea.classList.add("dragover");
}

function handleDragLeave(e) {
  e.preventDefault();
  if (uploadArea) uploadArea.classList.remove("dragover");
}

function handleDrop(e) {
  e.preventDefault();
  if (uploadArea) uploadArea.classList.remove("dragover");

  const files = e.dataTransfer.files;
  if (files.length > 0) {
    handleFile(files[0]);
  }
}

function handleFileSelect(e) {
  const file = e.target.files[0];
  if (file) {
    handleFile(file);
  }
}

function handleFile(file) {
  if (file.type !== "application/pdf") {
    showNotification("Please select a PDF file.", "error");
    return;
  }

  if (file.size > 10 * 1024 * 1024) {
    // 10MB limit
    showNotification("File size must be less than 10MB.", "error");
    return;
  }

  uploadedFile = file;
  showUploadProgress();
  uploadFileToServer(file);
}

// Text input functionality
function updateTextStats() {
  if (!textInput) return;

  const text = textInput.value;
  const words = text.trim() ? text.trim().split(/\s+/).length : 0;
  const chars = text.length;
  const readingTime = Math.max(1, Math.ceil(words / 200));

  if (wordCount) wordCount.textContent = words.toLocaleString();
  if (charCount) charCount.textContent = chars.toLocaleString();
  if (readTime) readTime.textContent = `${readingTime} min`;

  // Enable/disable analyze button
  if (analyzeBtn) {
    analyzeBtn.disabled = words < 10;
    analyzeBtn.classList.toggle("disabled", words < 10);
  }

  // Update chat input status
  textContent = text;
  updateChatInputStatus();
}

function switchTab(tabName) {
  currentInputMode = tabName;

  // Update tab buttons
  document.querySelectorAll(".selector-tab").forEach((tab) => {
    tab.classList.remove("active");
  });

  const activeTab = document.querySelector(`[data-tab="${tabName}"]`);
  if (activeTab) {
    activeTab.classList.add("active");
  }

  // Show/hide panels
  if (textInputPanel && uploadPanel) {
    if (tabName === "text") {
      textInputPanel.style.display = "block";
      uploadPanel.style.display = "none";
      textInputPanel.classList.add("fade-in-up");
    } else {
      textInputPanel.style.display = "none";
      uploadPanel.style.display = "block";
      uploadPanel.classList.add("fade-in-up");
    }
  }

  updateChatInputStatus();
}

function updateChatInputStatus() {
  const hasContent =
    (currentInputMode === "text" && textContent.trim()) ||
    (currentInputMode === "upload" && uploadedFile);

  if (chatInput && sendBtn) {
    chatInput.disabled = !hasContent;
    sendBtn.disabled = !hasContent;

    chatInput.placeholder = hasContent
      ? "Ask a question about your content..."
      : "Upload a document or enter text first...";
  }
}

async function analyzeText() {
  if (!textContent.trim()) {
    showNotification("Please enter some text to analyze.", "error");
    return;
  }

  if (isProcessing) return;

  isProcessing = true;
  updateStatus("processing", "Analyzing text...");

  if (analyzeBtn) {
    analyzeBtn.disabled = true;
    analyzeBtn.innerHTML =
      '<i class="fas fa-spinner fa-spin"></i> Analyzing...';
  }

  try {
    // Call Flask API for text analysis
    const response = await fetch("/api/analyze-text", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        text: textContent,
        summary_length:
          document.getElementById("summaryLength")?.value || "medium",
        summary_type:
          document.getElementById("summaryType")?.value || "abstractive",
      }),
    });

    const result = await response.json();

    if (result.success) {
      // Display results
      displayTextAnalysisResults(result);

      // Enable chat functionality
      updateChatInputStatus();

      showNotification("Text analysis completed successfully!", "success");
      updateStatus("ready", "Analysis complete - Ready for questions");
    } else {
      throw new Error(result.error || "Analysis failed");
    }
  } catch (error) {
    console.error("Analysis error:", error);
    showNotification("Analysis failed. Please try again.", "error");
    updateStatus("error", "Analysis failed");
  } finally {
    isProcessing = false;
    if (analyzeBtn) {
      analyzeBtn.disabled = false;
      analyzeBtn.innerHTML = '<i class="fas fa-brain"></i> Analyze & Summarize';
    }
  }
}

function generateMockSummary(text) {
  const words = text.trim().split(/\s+/);
  const sentences = text.split(/[.!?]+/).filter((s) => s.trim());

  // Extract first few sentences as summary
  const summaryLength = Math.min(3, Math.ceil(sentences.length * 0.3));
  const summary = sentences.slice(0, summaryLength).join(". ").trim() + ".";

  return {
    summary: summary,
    stats: {
      word_count: words.length,
      sentence_count: sentences.length,
      reading_time: Math.ceil(words.length / 200),
    },
    keywords: extractKeywords(text),
  };
}

function extractKeywords(text) {
  const words = text.toLowerCase().match(/\b\w{4,}\b/g) || [];
  const frequency = {};

  words.forEach((word) => {
    frequency[word] = (frequency[word] || 0) + 1;
  });

  return Object.entries(frequency)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 10)
    .map(([word, freq]) => ({ word, frequency: freq }));
}

function displayTextAnalysisResults(results) {
  const stats = results.stats || {};
  const keywords = results.keywords || [];

  // Create results container
  const resultsHtml = `
        <div class="summary-result">
            <h4><i class="fas fa-file-text"></i> Summary</h4>
            <p>${results.summary}</p>
        </div>

        <div class="metrics-row">
            <div class="metric-item">
                <span class="value">${stats.word_count || 0}</span>
                <span class="label">Words</span>
            </div>
            <div class="metric-item">
                <span class="value">${stats.sentence_count || 0}</span>
                <span class="label">Sentences</span>
            </div>
            <div class="metric-item">
                <span class="value">${stats.reading_time || 0}</span>
                <span class="label">Min Read</span>
            </div>
        </div>

        <div class="keywords-display">
            <strong>Key Terms:</strong>
            ${keywords.map((k) => `<span class="keyword-badge">${k.word} (${k.frequency})</span>`).join("")}
        </div>
    `;

  // Find or create results container
  let resultsContainer = document.querySelector(".analysis-results");
  if (!resultsContainer) {
    resultsContainer = document.createElement("div");
    resultsContainer.className = "analysis-results";
    if (textInputPanel) {
      textInputPanel.appendChild(resultsContainer);
    }
  }

  resultsContainer.innerHTML = resultsHtml;

  // Add to chat as assistant message
  const stats = results.stats || {};
  const keywords = results.keywords || [];
  addMessage(
    "assistant",
    `I've analyzed your text! Here's what I found:\n\n**Summary:** ${results.summary}\n\n**Stats:** ${stats.word_count || 0} words, ${stats.sentence_count || 0} sentences, ~${stats.reading_time || 0} minute read.\n\n**Top Keywords:** ${keywords
      .slice(0, 5)
      .map((k) => k.word)
      .join(", ")}`,
  );
}

function clearText() {
  if (textInput) {
    textInput.value = "";
    updateTextStats();
  }

  // Remove results
  const resultsContainer = document.querySelector(".analysis-results");
  if (resultsContainer) {
    resultsContainer.remove();
  }

  // Reset chat status
  updateChatInputStatus();
  updateStatus("ready", "Ready to help");
}

function pasteExample() {
  const exampleText = `Artificial Intelligence (AI) has revolutionized numerous industries and aspects of our daily lives. From healthcare and finance to transportation and entertainment, AI technologies are being implemented to improve efficiency, accuracy, and decision-making processes.

Machine learning, a subset of AI, enables computers to learn and improve from experience without being explicitly programmed. This technology has led to breakthroughs in natural language processing, computer vision, and predictive analytics.

The future of AI holds immense potential, with applications ranging from autonomous vehicles and smart cities to personalized medicine and climate change solutions. However, it also raises important questions about ethics, privacy, and the impact on employment that society must address as we continue to advance these technologies.`;

  if (textInput) {
    textInput.value = exampleText;
    updateTextStats();
    textInput.focus();
  }
}

function scrollToTextInput() {
  switchTab("text");
  setTimeout(() => {
    const textSection = document.getElementById("app-interface");
    if (textSection) {
      textSection.scrollIntoView({ behavior: "smooth" });
    }
  }, 100);
}

// Enhanced message handling for both text and document content
async function sendMessage() {
  const message = chatInput.value.trim();
  if (!message || isProcessing) return;

  // Check if we have content to work with
  const hasContent =
    (currentInputMode === "text" && textContent.trim()) ||
    (currentInputMode === "upload" && uploadedFile);

  if (!hasContent) {
    showNotification(
      "Please upload a document or enter text first.",
      "warning",
    );
    return;
  }

  // Add user message
  addMessage("user", message);
  chatInput.value = "";

  // Show typing indicator
  showTypingIndicator();
  isProcessing = true;

  try {
    // Call Flask API for chat
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: message,
        file_id: currentFileId,
        mode: currentInputMode,
      }),
    });

    const result = await response.json();

    if (result.success) {
      hideTypingIndicator();
      addMessage("assistant", result.response);

      // Add to chat history
      chatHistory.push({
        user: message,
        assistant: result.response,
        timestamp: new Date().toISOString(),
        mode: currentInputMode,
        message_id: result.message_id,
      });
    } else {
      throw new Error(result.error || "Chat request failed");
    }
  } catch (error) {
    console.error("Chat error:", error);
    hideTypingIndicator();
    addMessage(
      "assistant",
      "I apologize, but I encountered an error. Please try asking your question again.",
    );
  } finally {
    isProcessing = false;
  }
}

function generateContextualResponse(question, mode) {
  const lowerQuestion = question.toLowerCase();

  // Context-aware responses
  if (
    lowerQuestion.includes("summary") ||
    lowerQuestion.includes("summarize")
  ) {
    return mode === "text"
      ? "Based on the text you provided, I've already generated a summary above. Would you like me to focus on any specific aspect or create a different type of summary?"
      : "I can help you summarize your document. Please make sure you've uploaded a PDF file and processed it first.";
  }

  if (
    lowerQuestion.includes("main topic") ||
    lowerQuestion.includes("main point")
  ) {
    return mode === "text"
      ? "From your text, the main topic appears to focus on the key themes and concepts you've shared. Let me analyze the content more specifically..."
      : "I can identify the main topics in your document once it's been processed. Make sure to upload and analyze your PDF first.";
  }

  if (
    lowerQuestion.includes("key point") ||
    lowerQuestion.includes("important")
  ) {
    return "Here are the key points I've identified: 1) The primary concepts discussed, 2) Supporting details and evidence, 3) Main conclusions or implications. Would you like me to elaborate on any of these?";
  }

  if (lowerQuestion.includes("explain") || lowerQuestion.includes("clarify")) {
    return "I'd be happy to explain that in simpler terms. Could you specify which part you'd like me to clarify? I can break down complex concepts into more digestible explanations.";
  }

  // Default contextual response
  return `That's an interesting question about your ${mode === "text" ? "text" : "document"}. Based on the content you've provided, I can help you understand the key concepts and provide detailed insights. What specific aspect would you like me to focus on?`;
}

function addMessage(sender, message) {
  const messageDiv = document.createElement("div");
  messageDiv.className = `message ${sender}-message`;

  const timestamp = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });

  messageDiv.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-${sender === "user" ? "user" : "robot"}"></i>
        </div>
        <div class="message-content">
            <p>${message.replace(/\n/g, "<br>")}</p>
        </div>
        <div class="message-time">${timestamp}</div>
    `;

  if (chatMessages) {
    chatMessages.appendChild(messageDiv);
    scrollChatToBottom();
  }
}

function showTypingIndicator() {
  if (typingIndicator) {
    typingIndicator.style.display = "flex";
    scrollChatToBottom();
  }
}

function hideTypingIndicator() {
  if (typingIndicator) {
    typingIndicator.style.display = "none";
  }
}

function scrollChatToBottom() {
  if (chatMessages) {
    chatMessages.scrollTop = chatMessages.scrollHeight;
  }
}

async function askQuickQuestion(question) {
  if (chatInput) {
    chatInput.value = question;
    await sendMessage();
  }
}

function clearChat() {
  if (chatMessages) {
    // Keep only the initial assistant message
    const initialMessage = chatMessages.querySelector(
      ".message.assistant-message",
    );
    chatMessages.innerHTML = "";
    if (initialMessage) {
      chatMessages.appendChild(initialMessage);
    }
  }
  chatHistory = [];
}

function showUploadProgress() {
  if (uploadArea) uploadArea.style.display = "none";
  if (uploadProgress) uploadProgress.style.display = "block";
  if (documentInfo) documentInfo.style.display = "none";
}

async function uploadFileToServer(file) {
  const progressFill = document.querySelector(".progress-fill");
  const progressText = document.querySelector(".progress-text");

  updateStatus("processing", "Uploading and processing document...");

  try {
    // Upload file to Flask API
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/api/upload", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    if (result.success) {
      // Simulate progress for visual feedback
      let progress = 0;
      const interval = setInterval(() => {
        progress += 20;
        if (progress >= 100) {
          progress = 100;
          clearInterval(interval);
          setTimeout(() => {
            documentText = result.preview;
            currentFileId = result.file_id;
            showDocumentInfo(file);
            updateStatus("ready", "Document ready for analysis");
            showNotification("Document uploaded successfully!", "success");
          }, 300);
        }
        if (progressFill) progressFill.style.width = progress + "%";
        if (progressText)
          progressText.textContent = `Processing... ${Math.round(progress)}%`;
      }, 200);
    } else {
      throw new Error(result.error || "Upload failed");
    }
  } catch (error) {
    console.error("Upload error:", error);
    showNotification("Upload failed: " + error.message, "error");
    resetUploadArea();
    updateStatus("ready", "Ready to help");
  }
}

function showDocumentInfo(file) {
  if (uploadProgress) uploadProgress.style.display = "none";
  if (documentInfo) documentInfo.style.display = "block";

  // Update document details
  const docName = document.getElementById("docName");
  const docSize = document.getElementById("docSize");

  if (docName) docName.textContent = file.name;
  if (docSize) docSize.textContent = `Size: ${formatFileSize(file.size)}`;

  // Enable chat input
  updateChatInputStatus();

  // Add document uploaded message to chat
  addMessage(
    "assistant",
    `Great! I've processed "${file.name}". You can now ask me questions about the document or request a summary.`,
  );
}

function removeDocument() {
  uploadedFile = null;
  currentFileId = null;
  documentText = "";
  resetUploadArea();
  updateChatInputStatus();
  updateStatus("ready", "Ready to help");
}

function resetUploadArea() {
  if (uploadArea) uploadArea.style.display = "block";
  if (uploadProgress) uploadProgress.style.display = "none";
  if (documentInfo) documentInfo.style.display = "none";

  if (fileInput) fileInput.value = "";
}

async function summarizeDocument() {
  if (!uploadedFile || isProcessing) return;

  isProcessing = true;
  updateStatus("processing", "Generating summary...");

  const summarizeBtn = document.getElementById("summarizeBtn");
  if (summarizeBtn) {
    summarizeBtn.disabled = true;
    summarizeBtn.innerHTML =
      '<i class="fas fa-spinner fa-spin"></i> Summarizing...';
  }

  try {
    // Call Flask API for summarization
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: "Please provide a summary of this document",
        file_id: currentFileId,
        mode: "upload",
      }),
    });

    const result = await response.json();

    if (result.success) {
      addMessage(
        "assistant",
        `📄 **Document Summary**\n\n${result.response}\n\nYou can now ask me questions about your document!`,
      );

      showNotification("Document summarized successfully!", "success");
      updateStatus("ready", "Ready for questions");
    } else {
      throw new Error(result.error || "Summarization failed");
    }
  } catch (error) {
    console.error("Summarization error:", error);
    showNotification("Summarization failed. Please try again.", "error");
    updateStatus("error", "Summarization failed");
  } finally {
    isProcessing = false;
    if (summarizeBtn) {
      summarizeBtn.disabled = false;
      summarizeBtn.innerHTML = '<i class="fas fa-compress-alt"></i> Summarize';
    }
  }
}

function updateStatus(status, message) {
  if (statusIndicator) {
    statusIndicator.className = `status-indicator ${status}`;
  }
  if (statusText) {
    statusText.textContent = message;
  }
}

function formatFileSize(bytes) {
  if (bytes === 0) return "0 Bytes";
  const k = 1024;
  const sizes = ["Bytes", "KB", "MB", "GB"];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
}

function showNotification(message, type = "info") {
  // Create notification element
  const notification = document.createElement("div");
  notification.className = `notification notification-${type}`;
  notification.innerHTML = `
    <div class="notification-content">
      <i class="fas fa-${getNotificationIcon(type)}"></i>
      <span>${message}</span>
      <button class="notification-close" onclick="this.parentElement.parentElement.remove()">
        <i class="fas fa-times"></i>
      </button>
    </div>
  `;

  // Add to page
  document.body.appendChild(notification);

  // Auto remove after 5 seconds
  setTimeout(() => {
    if (notification.parentElement) {
      notification.remove();
    }
  }, 5000);
}

function getNotificationIcon(type) {
  switch (type) {
    case "success":
      return "check-circle";
    case "error":
      return "exclamation-circle";
    case "warning":
      return "exclamation-triangle";
    default:
      return "info-circle";
  }
}

function scrollToUpload() {
  const appSection = document.getElementById("app-interface");
  if (appSection) {
    appSection.scrollIntoView({ behavior: "smooth" });
  }
}

function updateNavOnScroll() {
  window.addEventListener("scroll", () => {
    const navbar = document.querySelector(".navbar");
    if (navbar) {
      if (window.scrollY > 100) {
        navbar.style.background = "rgba(255, 255, 255, 0.95)";
        navbar.style.backdropFilter = "blur(10px)";
      } else {
        navbar.style.background = "rgba(255, 255, 255, 0.1)";
        navbar.style.backdropFilter = "blur(5px)";
      }
    }
  });
}

function initializeAnimations() {
  // Add intersection observer for fade-in animations
  const observer = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
      if (entry.isIntersecting) {
        entry.target.style.opacity = "1";
        entry.target.style.transform = "translateY(0)";
      }
    });
  });

  // Observe elements for animation
  document
    .querySelectorAll(".feature-card, .step, .highlight-card")
    .forEach((el) => {
      el.style.opacity = "0";
      el.style.transform = "translateY(20px)";
      el.style.transition = "opacity 0.6s ease, transform 0.6s ease";
      observer.observe(el);
    });
}

// Enhanced smooth scrolling
function initializeSmoothScrolling() {
  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", function (e) {
      e.preventDefault();
      const target = document.querySelector(this.getAttribute("href"));
      if (target) {
        target.scrollIntoView({
          behavior: "smooth",
          block: "start",
        });
      }
    });
  });
}

// Initialize notification styles
function initializeNotificationStyles() {
  const notificationStyles = `
    .notification {
      position: fixed;
      top: 20px;
      right: 20px;
      background: white;
      border-radius: 12px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.2);
      z-index: 1000;
      animation: slideIn 0.3s ease;
      max-width: 400px;
    }

    .notification-content {
      display: flex;
      align-items: center;
      gap: 1rem;
      padding: 1rem 1.5rem;
    }

    .notification-success { border-left: 4px solid #10b981; }
    .notification-error { border-left: 4px solid #ef4444; }
    .notification-warning { border-left: 4px solid #f59e0b; }
    .notification-info { border-left: 4px solid #3b82f6; }

    .notification-close {
      background: none;
      border: none;
      cursor: pointer;
      color: #64748b;
      margin-left: auto;
    }

    @keyframes slideIn {
      from { transform: translateX(100%); opacity: 0; }
      to { transform: translateX(0); opacity: 1; }
    }
  `;

  // Add styles to head if not already present
  if (!document.querySelector("#notification-styles")) {
    const styleSheet = document.createElement("style");
    styleSheet.id = "notification-styles";
    styleSheet.textContent = notificationStyles;
    document.head.appendChild(styleSheet);
  }
}
