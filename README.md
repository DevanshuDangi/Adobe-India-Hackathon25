# Challenge 1b: Multi-Collection PDF Analysis

## Overview
Advanced PDF analysis solution that processes multiple document collections and extracts relevant content based on specific personas and use cases using machine learning techniques for intelligent document understanding.

## Features
- **Persona-driven Analysis**: Extracts content relevant to specific user roles and tasks
- **Multi-document Processing**: Handles multiple PDF collections simultaneously
- **Intelligent Section Ranking**: Uses TF-IDF and cosine similarity for relevance scoring
- **Automated Text Extraction**: Advanced PDF parsing with PyMuPDF
- **Structured Output**: JSON format with metadata and ranked results
- **Docker Support**: Containerized deployment for consistent environments
- **Cross-platform Compatibility**: Works on Windows, macOS, and Linux

## Project Structure
```
Challenge_1b/
├── Dockerfile                       # Docker container configuration
├── docker-compose.yml              # Docker Compose setup
├── .dockerignore                   # Docker ignore file
├── requirements.txt                # Python dependencies
├── main.py                         # Main application script
├── README.md                       # This file
├── Collection1/                    # Travel Planning
│   ├── PDFs/                       # South of France guides
│   ├── challenge1b_input.json      # Input configuration
│   └── challenge1b_output.json     # Analysis results
├── Collection2/                    # Adobe Acrobat Learning
│   ├── PDFs/                       # Acrobat tutorials
│   ├── challenge1b_input.json      # Input configuration
│   └── challenge1b_output.json     # Analysis results
├── Collection3/                    # Recipe Collection
│   ├── PDFs/                       # Cooking guides
│   ├── challenge1b_input.json      # Input configuration
│   └── challenge1b_output.json     # Analysis results
└── vnev/                           # Virtual environment (local)
```

## Collections

### Collection 1: Travel Planning
- **Challenge ID**: round_1b_002
- **Persona**: Travel Planner
- **Task**: Plan a 4-day trip for 10 college friends to South of France
- **Documents**: 7 travel guides

### Collection 2: Adobe Acrobat Learning
- **Challenge ID**: round_1b_003
- **Persona**: HR Professional
- **Task**: Create and manage fillable forms for onboarding and compliance
- **Documents**: 15 Acrobat guides

### Collection 3: Recipe Collection
- **Challenge ID**: round_1b_001
- **Persona**: Food Contractor
- **Task**: Prepare vegetarian buffet-style dinner menu for corporate gathering
- **Documents**: 9 cooking guides

## Installation & Setup

### Prerequisites
- Python 3.11 or higher
- Docker (for containerized deployment)
- Git

### Method 1: Local Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd Challenge_1b
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv vnev
   source vnev/bin/activate  # On Windows: vnev\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the application**:
   ```bash
   python main.py
   ```

### Method 2: Docker Deployment (Recommended)

1. **Using Docker Compose (Easy)**:
   ```bash
   # Build and run all services
   docker-compose up --build
   
   # Run in detached mode
   docker-compose up -d --build
   
   # View logs
   docker-compose logs -f
   
   # Stop containers
   docker-compose down
   ```

2. **Using Docker directly**:
   ```bash
   # Build the image
   docker build -t document-intelligence .
   
   # Run with volume mounts
   docker run -v $(pwd)/Collection1:/app/Collection1 \
              -v $(pwd)/Collection2:/app/Collection2 \
              -v $(pwd)/Collection3:/app/Collection3 \
              document-intelligence
   ```

3. **For development/debugging**:
   ```bash
   # Interactive mode
   docker run -it -v $(pwd):/app document-intelligence bash
   
   # Or with docker-compose
   docker-compose exec document-intelligence bash
   ```

## Usage

### Input Configuration
Each collection requires a `challenge1b_input.json` file:

```json
{
  "challenge_info": {
    "challenge_id": "round_1b_XXX",
    "test_case_name": "specific_test_case"
  },
  "documents": [
    {"filename": "document1.pdf", "title": "Document Title"},
    {"filename": "document2.pdf", "title": "Another Document"}
  ],
  "persona": {
    "role": "User Persona (e.g., Travel Planner, HR Professional)"
  },
  "job_to_be_done": {
    "task": "Specific task description and requirements"
  }
}
```

### Running Analysis
The application automatically processes all collections in the directory:

```bash
# Local execution
python main.py

# Docker execution
docker-compose up --build
```

### Output Structure
Results are saved as `challenge1b_output.json` in each collection directory:

```json
{
  "metadata": {
    "input_documents": ["list of processed files"],
    "persona": "User Persona",
    "job_to_be_done": "Task description",
    "processing_timestamp": "2025-01-XX:XX:XX"
  },
  "extracted_sections": [
    {
      "document": "source.pdf",
      "section_title": "Extracted Section Title",
      "importance_rank": 1,
      "page_number": 1
    }
  ],
  "subsection_analysis": [
    {
      "document": "source.pdf",
      "refined_text": "Most relevant content extracted",
      "page_number": 1
    }
  ]
}
```

## Algorithm Details

### Document Processing Pipeline
1. **PDF Text Extraction**: Uses PyMuPDF for robust text extraction with position information
2. **Section Detection**: Identifies document sections using formatting heuristics (headers, capitalization)
3. **Content Cleaning**: Removes noise, normalizes text, and handles special characters
4. **Relevance Scoring**: Uses TF-IDF vectorization and cosine similarity against persona+task queries
5. **Ranking & Selection**: Sorts sections by relevance score and selects top candidates
6. **Subsection Refinement**: Extracts most relevant sentences from top-ranked sections

### Key Parameters
- `max_sections`: Maximum sections to extract (default: 10)
- `max_subs`: Maximum subsections for detailed analysis (default: 5)
- `TF-IDF features`: 500 features with 1-2 gram analysis
- `min_content_length`: 20 characters minimum for section inclusion

## Dependencies

### Core Libraries
- **PyMuPDF (fitz)**: PDF text extraction and manipulation
- **scikit-learn**: Machine learning algorithms for text analysis
- **numpy**: Numerical computing support

### System Requirements
- **Memory**: Minimum 2GB RAM (4GB recommended)
- **Storage**: 1GB free space for processing
- **CPU**: Multi-core processor recommended for large collections

## Docker Configuration

### Container Features
- **Base Image**: Python 3.11-slim for optimal size and performance
- **Security**: Runs as non-root user
- **Optimization**: Multi-stage build with dependency caching
- **Volume Support**: Seamless data mounting for collections
- **Environment**: Isolated and reproducible execution environment

### Volume Mapping
```yaml
volumes:
  - ./Collection1:/app/Collection1
  - ./Collection2:/app/Collection2  
  - ./Collection3:/app/Collection3
  # Add more collections as needed
```

## Troubleshooting

### Common Issues

1. **PyMuPDF Installation Error**:
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install python3-dev
   
   # On macOS
   brew install python3
   
   # Using Docker (recommended)
   docker-compose up --build
   ```

2. **Memory Issues with Large PDFs**:
   - Increase Docker memory limit
   - Process collections separately
   - Use smaller `max_sections` parameter

3. **Permission Errors**:
   ```bash
   # Fix file permissions
   chmod -R 755 Collection*/
   
   # Or use Docker which handles permissions automatically
   docker-compose up
   ```

4. **Empty Output**:
   - Verify PDF files are readable
   - Check `challenge1b_input.json` format
   - Ensure PDF contains extractable text (not scanned images)

### Debug Mode
Enable detailed logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Performance Optimization

### For Large Collections
- Use Docker with adequate memory allocation
- Process collections in parallel (modify main.py)
- Implement batch processing for memory efficiency
- Consider text caching for repeated analysis

### Speed Improvements
- Use SSD storage for better I/O performance
- Increase Docker memory limits
- Pre-process PDFs to extract text only once

## Contributing

### Development Setup
1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test with Docker
4. Submit pull request with detailed description

### Code Style
- Follow PEP 8 guidelines
- Add type hints for new functions
- Include docstrings for complex methods
- Test with multiple PDF formats

## License
This project is part of the Adobe India Hackathon 2025 Challenge 1b.

## Support
For issues and questions:
1. Check the troubleshooting section
2. Review Docker logs: `docker-compose logs`
3. Test with sample collections provided
4. Verify input JSON format compliance

---

**Last Updated**: January 2025  
**Version**: 1.0.0  
**Compatibility**: Python 3.11+, Docker 20.0+