# Fact-Check Connectors for TruthLens Phase 3

This directory contains the fact-checking source connectors for the TruthLens Evidence Retrieval System. These connectors implement RSS/HTML scraping for major fact-checking organizations and provide normalized evidence data with fuzzy matching capabilities.

## 🎯 **Overview**

The fact-check connectors automatically retrieve and process fact-checking articles from multiple sources, normalize them to the TruthLens evidence schema, and provide fuzzy matching capabilities to find relevant fact-checks for specific claims.

## 🔌 **Available Connectors**

### 1. **PIB Fact Check** (`PIBFactCheckConnector`)
- **Source**: Government of India Press Information Bureau
- **URL**: https://pib.gov.in
- **RSS**: https://pib.gov.in/rss/fact-check.xml
- **Rate Limit**: 5 requests/minute (conservative for government site)
- **Language**: Primarily Hindi and English
- **Focus**: Government fact-checks, official statements

### 2. **PolitiFact** (`PolitiFactConnector`)
- **Source**: Pulitzer Prize-winning fact-checking website
- **URL**: https://www.politifact.com
- **RSS**: https://www.politifact.com/feeds/fact-check.xml
- **Rate Limit**: 15 requests/minute
- **Language**: English
- **Focus**: Political claims, public statements

### 3. **BOOM Live** (`BOOMLiveConnector`)
- **Source**: Indian fact-checking and news verification platform
- **URL**: https://www.boomlive.in
- **RSS**: https://www.boomlive.in/feed
- **Rate Limit**: 10 requests/minute
- **Language**: English, Hindi, regional languages
- **Focus**: Social media misinformation, viral claims

### 4. **AltNews** (`AltNewsConnector`)
- **Source**: Indian fact-checking website focusing on misinformation
- **URL**: https://www.altnews.in
- **RSS**: https://www.altnews.in/feed
- **Rate Limit**: 8 requests/minute
- **Language**: English, Hindi
- **Focus**: Misinformation, fake news, social media claims

## 🚀 **Quick Start**

### Installation

```bash
# Install required dependencies
pip install -r connectors/requirements_factcheck.txt

# Or install individually
pip install trafilatura beautifulsoup4 feedparser dateparser tldextract fuzzywuzzy
```

### Basic Usage

```python
import asyncio
from connectors.factcheck import FactCheckAggregator

async def main():
    # Create aggregator
    aggregator = FactCheckAggregator()
    
    # Search for claims across all sources
    claims = ["Climate change is caused by human activities", "Vaccines cause autism"]
    results = await aggregator.search_all_sources(claims, max_results=5)
    
    # Process results
    for source, source_results in results.items():
        print(f"\n{source.upper()} Results:")
        for claim, evidence, score in source_results:
            print(f"  Claim: {claim[:50]}...")
            print(f"  Evidence: {evidence.title}")
            print(f"  Score: {score}")
            print(f"  URL: {evidence.url}")

# Run
asyncio.run(main())
```

### Individual Connector Usage

```python
from connectors.factcheck import PIBFactCheckConnector

# Create PIB connector
pib = PIBFactCheckConnector()

# Fetch fact-checks
async def fetch_pib_facts():
    fact_checks = await pib.fetch_fact_checks()
    for evidence in fact_checks:
        print(f"Title: {evidence.title}")
        print(f"URL: {evidence.url}")
        print(f"Language: {evidence.language}")
        print(f"Published: {evidence.published_at}")
        print("-" * 50)

# Run
asyncio.run(fetch_pib_facts())
```

## 🔧 **Features**

### **RSS Feed Processing**
- Automatic RSS feed parsing and item extraction
- Support for various RSS formats and encodings
- Error handling and fallback mechanisms

### **HTML Article Scraping**
- **Primary**: `trafilatura` for clean text extraction
- **Fallback**: BeautifulSoup for basic HTML parsing
- Automatic metadata extraction (title, date, author)
- Content cleaning and normalization

### **Language Detection**
- Automatic language detection for multilingual content
- Support for English, Hindi, Tamil, Marathi, and more
- Unicode character set analysis

### **Evidence Normalization**
- Converts scraped data to TruthLens `RawEvidence` format
- Automatic ID generation and metadata enrichment
- Source-specific configuration and tracking

### **Fuzzy Matching**
- Uses `fuzzywuzzy` for claim-evidence similarity scoring
- Multiple similarity algorithms (ratio, partial_ratio, token_sort_ratio)
- Configurable relevance thresholds

### **Rate Limiting**
- Robots.txt compliant rate limiting
- Per-connector rate limit configuration
- Automatic backoff and retry mechanisms

## 📊 **Data Flow**

```
RSS Feed → Article URLs → HTML Scraping → Text Extraction → 
Language Detection → Evidence Normalization → Fuzzy Matching → Results
```

## 🧪 **Testing**

### Run Tests

```bash
# Run all fact-check tests
python -m pytest tests/test_factcheck.py -v

# Run specific test class
python -m pytest tests/test_factcheck.py::TestFactCheckConnectors -v

# Run with coverage
python -m pytest tests/test_factcheck.py --cov=connectors --cov-report=html
```

### Test Data

The test suite includes:
- **10 sample claims** in multiple languages (English, Hindi, Marathi, Tamil)
- **Multiple categories**: Environment, Health, Science
- **Expected source mappings** for each claim
- **Keyword sets** for relevance testing

### Demo Script

```bash
# Run demonstration script
python connectors/demo_factcheck.py
```

## ⚙️ **Configuration**

### Rate Limits
Each connector has configurable rate limits to respect source policies:

```python
# Conservative rate limiting for government sites
pib = PIBFactCheckConnector()  # 5 requests/minute

# Standard rate limiting for news sites
politifact = PolitiFactConnector()  # 15 requests/minute

# Moderate rate limiting for fact-check sites
boom = BOOMLiveConnector()  # 10 requests/minute
```

### User Agents
Connectors use respectful user agents:

```python
'User-Agent': 'TruthLens/1.0 (Fact-Checking Bot)'
```

## 🔍 **Adding New Sources**

To add a new fact-check source:

1. **Create Connector Class**
```python
class NewSourceConnector(FactCheckConnector):
    def __init__(self):
        super().__init__(
            base_url="https://newsource.com",
            name="New Source",
            rate_limit=10
        )
        self.rss_url = "https://newsource.com/feed"
    
    async def fetch_fact_checks(self) -> List[RawEvidence]:
        # Implement source-specific logic
        pass
    
    async def search_claims(self, claims: List[str], max_results: int = 10):
        # Implement claim search logic
        pass
```

2. **Add to Aggregator**
```python
class FactCheckAggregator:
    def __init__(self):
        self.connectors = {
            # ... existing connectors ...
            'newsource': NewSourceConnector()
        }
```

3. **Update Tests**
- Add test cases for the new connector
- Update test data with expected sources
- Verify evidence normalization

## 📈 **Performance Considerations**

### **Caching**
- RSS feeds are fetched fresh each time (consider adding caching)
- Article content is scraped on-demand
- Evidence objects are created for each article

### **Scalability**
- Connectors can be run in parallel using asyncio
- Rate limiting prevents overwhelming source servers
- Batch processing supported for multiple claims

### **Error Handling**
- Network timeouts and connection errors
- RSS parsing failures
- HTML scraping failures
- Rate limit violations

## 🚨 **Limitations and Considerations**

### **Rate Limiting**
- Respect source rate limits to avoid being blocked
- Implement exponential backoff for failed requests
- Monitor and adjust rate limits based on source response

### **Content Changes**
- RSS feeds and HTML structures may change
- Regular testing and monitoring required
- Fallback mechanisms for parsing failures

### **Legal and Ethical**
- Respect robots.txt and site terms of service
- Use data only for fact-checking purposes
- Attribute sources appropriately

## 🔮 **Future Enhancements**

### **Planned Features**
- [ ] Caching layer for RSS feeds and articles
- [ ] More sophisticated language detection
- [ ] Content deduplication across sources
- [ ] Machine learning-based relevance scoring
- [ ] Support for more fact-check sources

### **Integration Points**
- [ ] Vector database for semantic search
- [ ] Evidence processing pipeline
- [ ] API endpoints for claim verification
- [ ] Real-time fact-check monitoring

## 📚 **Dependencies**

### **Core Dependencies**
- `trafilatura>=7.0.0` - Advanced text extraction
- `beautifulsoup4>=4.12.0` - HTML parsing
- `feedparser>=6.0.0` - RSS feed processing
- `dateparser>=1.2.0` - Date parsing
- `tldextract>=5.1.0` - Domain extraction
- `fuzzywuzzy>=0.18.0` - Fuzzy string matching

### **Testing Dependencies**
- `pytest>=7.4.0` - Testing framework
- `pytest-asyncio>=0.23.0` - Async testing support

## 🤝 **Contributing**

1. **Fork the repository**
2. **Create a feature branch**
3. **Add tests for new functionality**
4. **Ensure all tests pass**
5. **Submit a pull request**

## 📄 **License**

This project is part of TruthLens and follows the same licensing terms.

## 🆘 **Support**

For issues and questions:
1. Check the test suite for usage examples
2. Review the demo script for implementation patterns
3. Open an issue with detailed error information
4. Ensure all dependencies are properly installed

---

**Note**: These connectors are designed for educational and research purposes. Always respect the terms of service and robots.txt files of the sources you're connecting to.
