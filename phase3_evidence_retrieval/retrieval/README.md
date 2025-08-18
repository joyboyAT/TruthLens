# Grounded Search for TruthLens Phase 3

This directory contains the grounded search functionality for the TruthLens Evidence Retrieval System. It provides pluggable web search capabilities with site filtering and recency-based search for news, government, and health sources.

## 🎯 **Overview**

The grounded search module integrates with multiple search APIs (SerpAPI, Bing Search API) and provides:
- **Pluggable search clients** for different search providers
- **Trusted domain filtering** for reliable sources
- **Time-based filtering** (day, week, month, year)
- **Content extraction** with advanced text processing
- **Published date parsing** from multiple metadata sources

## 🔌 **Available Search Clients**

### 1. **SerpAPI Client** (`SerpAPIClient`)
- **Provider**: SerpAPI (Google search results)
- **Features**: High-quality results, extensive metadata
- **Rate Limit**: Configurable (default: 60 requests/minute)
- **Max Results**: Up to 100 per query
- **Requirements**: SerpAPI API key

### 2. **Bing Search API Client** (`BingSearchClient`)
- **Provider**: Microsoft Bing Search API
- **Features**: Microsoft's search index, structured data
- **Rate Limit**: Configurable (default: 60 requests/minute)
- **Max Results**: Up to 50 per query
- **Requirements**: Bing Search API key

### 3. **Mock Search Client** (`MockSearchClient`)
- **Provider**: Mock data for testing
- **Features**: No API key required, realistic test data
- **Rate Limit**: Configurable (default: 60 requests/minute)
- **Max Results**: Up to 5 per query
- **Use Case**: Development, testing, demonstrations

## 🚀 **Quick Start**

### Installation

```bash
# Install required dependencies
pip install -r retrieval/requirements_grounded_search.txt

# Or install individually
pip install trafilatura beautifulsoup4 dateparser requests
```

### Basic Usage

```python
import asyncio
from retrieval.grounded_search import create_search_client, GroundedSearchEngine

async def main():
    # Create a mock search client (no API key required)
    search_client = create_search_client("mock")
    
    # Create search engine with trusted domains
    engine = GroundedSearchEngine(search_client)
    
    # Search for claims
    claims = ["Climate change is caused by human activities", "Vaccines are safe"]
    results = await engine.search_claims(
        claims=claims,
        max_results=5,
        time_filter="m",  # Past month
        extract_content=True
    )
    
    # Process results
    for claim, claim_results in results.items():
        print(f"\nClaim: {claim}")
        for result in claim_results:
            print(f"  Title: {result.title}")
            print(f"  URL: {result.url}")
            print(f"  Published: {result.published_at}")
            print(f"  Snippet: {result.snippet[:100]}...")

# Run
asyncio.run(main())
```

### Using Real Search APIs

```python
from retrieval.grounded_search import create_search_client

# SerpAPI (requires API key)
serp_client = create_search_client("serpapi", "your_serpapi_key")

# Bing Search API (requires API key)
bing_client = create_search_client("bing", "your_bing_api_key")

# Create search engine
engine = GroundedSearchEngine(serp_client)
```

## 🔧 **Features**

### **Pluggable Search Architecture**
- **Base Class**: `WebSearchClient` with common functionality
- **Factory Function**: `create_search_client()` for easy client creation
- **Unified Interface**: All clients implement the same search method
- **Easy Extension**: Add new search providers by extending `WebSearchClient`

### **Trusted Domain Filtering**
- **Pre-configured Domains**: Government, health, news, academic sources
- **Customizable**: Add/remove trusted domains at runtime
- **Category-based**: Group domains by type (government, health, news, academic)
- **Automatic Filtering**: Results automatically filtered to trusted sources

### **Time-based Filtering**
- **Day**: Past 24 hours
- **Week**: Past 7 days
- **Month**: Past 30 days
- **Year**: Past 365 days
- **API-specific**: Mapped to each search provider's format

### **Content Extraction**
- **Primary**: `trafilatura` for clean text extraction
- **Fallback**: BeautifulSoup for basic HTML parsing
- **Metadata Extraction**: Title, author, description, language
- **Text Cleaning**: Normalization and whitespace handling

### **Published Date Parsing**
- **Meta Tags**: `article:published_time`, `publish_date`, etc.
- **JSON-LD**: Structured data extraction
- **OpenGraph**: Social media metadata
- **Time Tags**: HTML5 `<time>` elements
- **Fallback**: Search result dates when available

### **Rate Limiting**
- **Configurable Limits**: Per-client rate limiting
- **Robots.txt Compliant**: Respectful API usage
- **Automatic Backoff**: Prevents overwhelming search providers

## 📊 **Data Flow**

```
Claim Input → Search Query → API Search → Result Filtering → 
Content Extraction → Metadata Parsing → Published Date → Final Results
```

## 🧪 **Testing**

### Run Tests

```bash
# Run all grounded search tests
python -m pytest tests/test_grounded_search.py -v

# Run specific test class
python -m pytest tests/test_grounded_search.py::TestSearchResult -v

# Run with coverage
python -m pytest tests/test_grounded_search.py --cov=retrieval --cov-report=html
```

### Test Requirements Met

The test suite verifies:
- **5 test claims** processed successfully
- **Top-5 URLs** with publish dates
- **256-char snippets** (when possible)
- **≥80% published_at parsing** success rate

### Demo Script

```bash
# Run demonstration script
python retrieval/grounded_search.py
```

## ⚙️ **Configuration**

### Trusted Domains

Default trusted domains include:

```python
# Government sources
"pib.gov.in", "gov.in", "mea.gov.in", "mohfw.gov.in"

# International organizations
"who.int", "un.org", "worldbank.org", "imf.org"

# Reputable news sources
"timesofindia.indiatimes.com", "thehindu.com", "reuters.com"

# Health and science
"mayoclinic.org", "webmd.com", "nih.gov", "cdc.gov"

# Academic
"wikipedia.org", "scholar.google.com", "researchgate.net"
```

### Custom Domain Management

```python
engine = GroundedSearchEngine(search_client)

# Add new trusted domain
engine.add_trusted_domain("newsource.com")

# Remove domain
engine.remove_trusted_domain("untrusted.com")

# Get domain categories
categories = engine.get_domain_categories()
```

### Time Filter Options

```python
# Search with different time filters
results = await engine.search_claims(
    claims=claims,
    time_filter="d",  # Past day
    max_results=10
)

# Available time filters:
# "d" - Past day
# "w" - Past week  
# "m" - Past month (default)
# "y" - Past year
```

## 🔍 **Adding New Search Providers**

To add a new search provider:

1. **Create Client Class**
```python
class NewSearchClient(WebSearchClient):
    def __init__(self, api_key: str, rate_limit: int = 60):
        super().__init__(api_key, rate_limit)
        self.base_url = "https://api.newsource.com/search"
    
    async def search(self, query: str, max_results: int = 10, 
                    time_filter: str = "m", site_filters: List[str] = None):
        # Implement search logic
        pass
```

2. **Add to Factory Function**
```python
def create_search_client(client_type: str, api_key: str = "") -> WebSearchClient:
    if client_type.lower() == "newsource":
        if not api_key:
            raise ValueError("NewSearch requires an API key")
        return NewSearchClient(api_key)
    # ... existing clients ...
```

3. **Update Tests**
- Add test cases for the new client
- Verify search functionality
- Test rate limiting and error handling

## 📈 **Performance Considerations**

### **Rate Limiting**
- **Per-client limits**: Prevent API quota exhaustion
- **Configurable**: Adjust based on provider limits
- **Automatic backoff**: Respect provider rate limits

### **Content Extraction**
- **On-demand**: Extract content only when requested
- **Parallel processing**: Multiple URLs processed concurrently
- **Caching**: Consider adding content caching for repeated URLs

### **Search Optimization**
- **Site filtering**: Reduce irrelevant results
- **Time filtering**: Focus on recent content
- **Result limiting**: Control API usage and processing time

## 🚨 **Limitations and Considerations**

### **API Dependencies**
- **API Keys**: SerpAPI and Bing require valid API keys
- **Rate Limits**: Respect provider rate limits
- **Quota Management**: Monitor API usage and costs

### **Content Extraction**
- **Network delays**: Content extraction adds latency
- **Parsing failures**: Some sites may not parse correctly
- **Rate limiting**: Respect site-specific rate limits

### **Domain Filtering**
- **Trust assumptions**: Pre-configured domains may change
- **Source verification**: Validate domain trustworthiness
- **Dynamic updates**: Consider automated domain validation

## 🔮 **Future Enhancements**

### **Planned Features**
- [ ] Content caching layer
- [ ] More search providers (Google Custom Search, DuckDuckGo)
- [ ] Advanced domain validation
- [ ] Machine learning-based source credibility scoring
- [ ] Automated domain whitelist updates

### **Integration Points**
- [ ] Vector database for semantic search
- [ ] Evidence processing pipeline
- [ ] Real-time search monitoring
- [ ] Search result ranking improvements

## 📚 **Dependencies**

### **Core Dependencies**
- `trafilatura>=7.0.0` - Advanced text extraction
- `beautifulsoup4>=4.12.0` - HTML parsing
- `dateparser>=1.2.0` - Date parsing
- `requests>=2.32.0` - HTTP requests

### **Optional Dependencies**
- `feedparser>=6.0.0` - RSS feed processing

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

**Note**: When using real search APIs, ensure you have valid API keys and respect the terms of service and rate limits of the providers.
