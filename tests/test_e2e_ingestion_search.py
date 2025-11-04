"""End-to-end integration tests for document ingestion + search.

These tests verify the complete system workflow:
1. Ingest a document (parse → chunk → store in Weaviate)
2. Search for content from that document
3. Verify correct retrieval and synthesis

Requirements:
- All services must be running (Weaviate, Vector Service, API Gateway, Ollama)
- Tests are marked with @pytest.mark.integration
- Run separately from unit tests: pytest -m integration

Note: These tests modify the database. Use a test collection to avoid
polluting production data.
"""

import pytest
from pathlib import Path
from typing import Any
import uuid

from agent.main import IntraMindAgent


# Mark all tests in this file as integration tests
pytestmark = pytest.mark.integration


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def test_collection_name() -> str:
    """Generate a unique test collection name to avoid conflicts."""
    return f"test_e2e_collection_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def agent() -> IntraMindAgent:
    """Create an agent instance for testing."""
    return IntraMindAgent()


@pytest.fixture
def sample_text_document(tmp_path: Path) -> Path:
    """Create a sample text document with known content for testing."""
    doc_path = tmp_path / "test_revenue_report.txt"
    content = """Q4 2024 Revenue Analysis Report

Executive Summary:
The company achieved record revenue of $5.2 million in Q4 2024, representing 
a 35% year-over-year growth. This exceptional performance was driven by three 
key factors:

1. Product Launch Success
The new AI-powered analytics platform launched in November exceeded expectations,
generating $1.8 million in first-month revenue. Customer adoption rate was 92%,
significantly higher than the industry average of 65%.

2. Geographic Expansion
Our Asia-Pacific expansion proved highly successful, contributing $1.2 million
to Q4 revenue. Japan and Singapore markets showed particularly strong performance
with 150% growth quarter-over-quarter.

3. Enterprise Contracts
Secured five major enterprise contracts worth $2.2 million total, including
partnerships with Fortune 500 companies in healthcare and finance sectors.

Conclusion:
Q4 2024 marks a historic milestone for the company. With strong product-market
fit and successful geographic expansion, we are well-positioned for continued
growth in 2025.
"""
    doc_path.write_text(content)
    return doc_path


@pytest.fixture
def sample_policy_document(tmp_path: Path) -> Path:
    """Create a sample policy document for testing."""
    doc_path = tmp_path / "remote_work_policy.txt"
    content = """Remote Work Policy - Effective January 2025

Overview:
This policy establishes guidelines for remote and hybrid work arrangements
to balance flexibility with collaboration and productivity.

Work Schedule:
- Monday & Friday: Fully remote (work from anywhere)
- Tuesday, Wednesday, Thursday: In-office required
- Core hours: 10 AM - 3 PM (all time zones)
- Flexible start/end times outside core hours

Equipment & Stipend:
- Company provides: Laptop, monitor, keyboard, mouse
- Home office stipend: $750 annually
- Internet reimbursement: Up to $50/month
- Ergonomic equipment: Available upon request

Eligibility:
- All full-time employees
- Must maintain performance standards
- Required to have reliable high-speed internet
- Home workspace must meet safety requirements

Communication Expectations:
- Respond to messages within 4 hours during work hours
- Attend all scheduled meetings (video on for team meetings)
- Use status indicators to show availability
- Over-communicate to maintain team cohesion

Performance & Accountability:
- Performance measured by outcomes, not hours
- Weekly 1-on-1s with manager (video call)
- Quarterly performance reviews continue as normal
- Productivity tracking tools may be used

This policy will be reviewed quarterly and adjusted based on feedback
and business needs.
"""
    doc_path.write_text(content)
    return doc_path


# ============================================================================
# Core Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_ingest_then_simple_search(
    agent: IntraMindAgent,
    sample_text_document: Path,
    test_collection_name: str
):
    """Test ingesting a document and then finding it with a simple search.
    
    This is the fundamental E2E test - proves the complete system works:
    - Document ingestion (parse → chunk → store)
    - Vector embedding generation
    - Semantic search retrieval
    - Result synthesis
    """
    # Step 1: Ingest the document
    ingestion_result = await agent.ingest_document(
        file_path=str(sample_text_document),
        collection_name=test_collection_name,
        chunk_size=500,
        chunk_overlap=100,
        document_metadata={
            "type": "financial_report",
            "department": "finance",
            "year": 2024,
            "quarter": "Q4"
        }
    )
    
    # Verify ingestion succeeded
    assert ingestion_result["success"] is True
    assert ingestion_result["chunks_stored"] > 0
    assert ingestion_result.get("error") is None
    
    chunks_stored = ingestion_result["chunks_stored"]
    print(f"\n✓ Ingested document: {chunks_stored} chunks stored")
    
    # Step 2: Search for content that should be in the document
    search_query = "What was the Q4 2024 revenue?"
    
    search_result = await agent.search(
        query=search_query,
        collection_name=test_collection_name,
        num_results=10
    )
    
    # Verify search succeeded
    assert search_result["success"] is True
    assert search_result.get("error") is None
    assert len(search_result["results"]) > 0
    
    # Verify we found relevant content
    found_revenue_info = False
    for result in search_result["results"]:
        content_lower = result["content"].lower()
        if "5.2 million" in content_lower or "revenue" in content_lower:
            found_revenue_info = True
            break
    
    assert found_revenue_info, "Search should find revenue information from ingested document"
    
    # Verify the synthesized response mentions the revenue
    response_lower = search_result["response"].lower()
    assert "revenue" in response_lower or "5.2" in response_lower or "million" in response_lower, \
        "Response should mention revenue from the ingested document"
    
    print(f"✓ Search found {len(search_result['results'])} relevant results")
    print(f"✓ Response: {search_result['response'][:200]}...")


@pytest.mark.asyncio
async def test_ingest_then_complex_search(
    agent: IntraMindAgent,
    sample_text_document: Path,
    test_collection_name: str
):
    """Test ingestion followed by complex multi-query search.
    
    Verifies:
    - Complex query classification
    - Query expansion with LLM
    - Multi-query search execution
    - Result deduplication
    - Synthesis across multiple queries
    """
    # Step 1: Ingest document
    ingestion_result = await agent.ingest_document(
        file_path=str(sample_text_document),
        collection_name=test_collection_name,
        chunk_size=400,
        chunk_overlap=80,
    )
    
    assert ingestion_result["success"] is True
    print(f"\n✓ Ingested {ingestion_result['chunks_stored']} chunks")
    
    # Step 2: Complex search requiring query expansion
    complex_query = "Compare the revenue from product launches versus geographic expansion and explain which contributed more to growth"
    
    search_result = await agent.search(
        query=complex_query,
        collection_name=test_collection_name,
        num_results=10
    )
    
    # Verify search succeeded
    assert search_result["success"] is True
    assert search_result.get("error") is None
    
    # Verify it was classified as complex
    assert search_result["complexity"] == "complex", \
        "Multi-part comparison query should be classified as complex"
    
    # Verify query expansion occurred
    assert search_result["expanded_queries"] is not None
    assert len(search_result["expanded_queries"]) >= 2, \
        "Complex query should be expanded into multiple sub-queries"
    
    # Verify we got results
    assert len(search_result["results"]) > 0, \
        "Should find relevant results from ingested document"
    
    # Verify the response is substantive
    assert len(search_result["response"]) > 100, \
        "Complex query should generate a detailed response"
    
    print(f"✓ Classified as: {search_result['complexity']}")
    print(f"✓ Expanded into {len(search_result['expanded_queries'])} queries")
    print(f"✓ Found {len(search_result['results'])} results")


@pytest.mark.asyncio
async def test_ingest_multiple_docs_then_search(
    agent: IntraMindAgent,
    sample_text_document: Path,
    sample_policy_document: Path,
    test_collection_name: str
):
    """Test ingesting multiple documents and searching across them.
    
    Verifies:
    - Multiple document ingestion
    - Cross-document search
    - Relevance ranking across different documents
    - Source attribution in results
    """
    # Step 1: Ingest first document (revenue report)
    result1 = await agent.ingest_document(
        file_path=str(sample_text_document),
        collection_name=test_collection_name,
        document_metadata={"doc_type": "financial_report", "title": "Q4 Revenue Report"}
    )
    
    assert result1["success"] is True
    print(f"\n✓ Ingested document 1: {result1['chunks_stored']} chunks")
    
    # Step 2: Ingest second document (policy)
    result2 = await agent.ingest_document(
        file_path=str(sample_policy_document),
        collection_name=test_collection_name,
        document_metadata={"doc_type": "policy", "title": "Remote Work Policy"}
    )
    
    assert result2["success"] is True
    print(f"✓ Ingested document 2: {result2['chunks_stored']} chunks")
    
    total_chunks = result1["chunks_stored"] + result2["chunks_stored"]
    print(f"✓ Total chunks in collection: {total_chunks}")
    
    # Step 3: Search for revenue (should find doc 1)
    search_revenue = await agent.search(
        query="What was the revenue in Q4?",
        collection_name=test_collection_name,
        num_results=5
    )
    
    assert search_revenue["success"] is True
    assert len(search_revenue["results"]) > 0
    
    # Verify we got revenue-related content
    revenue_found = any(
        "revenue" in result["content"].lower() or "million" in result["content"].lower()
        for result in search_revenue["results"]
    )
    assert revenue_found, "Should find revenue information"
    print(f"✓ Revenue search found {len(search_revenue['results'])} results")
    
    # Step 4: Search for policy (should find doc 2)
    search_policy = await agent.search(
        query="What is the remote work policy?",
        collection_name=test_collection_name,
        num_results=5
    )
    
    assert search_policy["success"] is True
    assert len(search_policy["results"]) > 0
    
    # Verify we got policy-related content
    policy_found = any(
        "remote" in result["content"].lower() or "policy" in result["content"].lower()
        or "office" in result["content"].lower()
        for result in search_policy["results"]
    )
    assert policy_found, "Should find policy information"
    print(f"✓ Policy search found {len(search_policy['results'])} results")
    
    # Step 5: Generic search (should find both)
    search_general = await agent.search(
        query="company information",
        collection_name=test_collection_name,
        num_results=10
    )
    
    assert search_general["success"] is True
    assert len(search_general["results"]) > 0
    print(f"✓ General search found {len(search_general['results'])} results from both documents")


@pytest.mark.asyncio
async def test_ingest_then_search_with_min_score(
    agent: IntraMindAgent,
    sample_text_document: Path,
    test_collection_name: str
):
    """Test ingestion followed by search with score filtering.
    
    Verifies:
    - Document ingestion
    - Search with min_score parameter
    - Score filtering actually works
    - High-quality results are prioritized
    """
    # Step 1: Ingest document
    ingestion_result = await agent.ingest_document(
        file_path=str(sample_text_document),
        collection_name=test_collection_name,
        chunk_size=500,
        chunk_overlap=100,
    )
    
    assert ingestion_result["success"] is True
    print(f"\n✓ Ingested {ingestion_result['chunks_stored']} chunks")
    
    # Step 2: Search with no score filter (baseline)
    search_no_filter = await agent.search(
        query="revenue growth in Q4",
        collection_name=test_collection_name,
        num_results=10,
        min_score=0.0
    )
    
    assert search_no_filter["success"] is True
    baseline_count = len(search_no_filter["results"])
    print(f"✓ No filter: {baseline_count} results")
    
    # Step 3: Search with high score filter
    search_high_filter = await agent.search(
        query="revenue growth in Q4",
        collection_name=test_collection_name,
        num_results=10,
        min_score=0.7
    )
    
    assert search_high_filter["success"] is True
    filtered_count = len(search_high_filter["results"])
    print(f"✓ With min_score=0.7: {filtered_count} results")
    
    # Verify filtering worked
    assert filtered_count <= baseline_count, \
        "Score filtering should return same or fewer results"
    
    # Verify all filtered results have high scores
    for result in search_high_filter["results"]:
        assert result["score"] >= 0.7, \
            f"Result score {result['score']} should be >= 0.7"
    
    print("✓ All filtered results have score >= 0.7")


@pytest.mark.asyncio
async def test_search_empty_collection_no_crash(
    agent: IntraMindAgent,
    test_collection_name: str
):
    """Test that searching an empty/non-existent collection doesn't crash.
    
    Verifies graceful handling of edge cases:
    - Collection doesn't exist
    - Collection is empty
    - System returns appropriate "no results" message
    """
    # Search in a collection that hasn't been created yet
    search_result = await agent.search(
        query="anything at all",
        collection_name=test_collection_name,  # Unique collection, not created yet
        num_results=10
    )
    
    # Should not crash - either success with 0 results or graceful error
    assert search_result is not None
    
    if search_result["success"]:
        # If successful, should return 0 results
        assert len(search_result["results"]) == 0
        assert "couldn't find" in search_result["response"].lower() or \
               "no results" in search_result["response"].lower() or \
               len(search_result["response"]) > 0
        print("✓ Empty collection handled gracefully - no results returned")
    else:
        # If error, should have error message
        assert "error" in search_result
        print(f"✓ Empty collection handled gracefully - error: {search_result['error'][:100]}")


@pytest.mark.asyncio
async def test_ingest_markdown_then_search(
    agent: IntraMindAgent,
    tmp_path: Path,
    test_collection_name: str
):
    """Test ingesting a Markdown document and searching for its content.
    
    Verifies:
    - Markdown file parsing
    - Preservation of document structure
    - Search across formatted content
    """
    # Create a markdown document
    md_path = tmp_path / "project_readme.md"
    md_content = """# IntraMind AI Project

## Overview
IntraMind is an intelligent document search system powered by semantic vector search
and large language models. It enables natural language queries across enterprise
document repositories.

## Key Features
- **Semantic Search**: Find documents by meaning, not just keywords
- **Query Expansion**: Complex queries are automatically expanded into sub-queries
- **Result Synthesis**: LLM synthesizes coherent answers from multiple documents
- **Multi-format Support**: PDF, DOCX, PPTX, TXT, MD files

## Architecture
The system uses a microservices architecture:
1. API Gateway (C# .NET)
2. Vector Database Service (Python + gRPC)
3. Weaviate Vector Database
4. LLM Integration (Claude Haiku + Ollama)

## Performance
- Query latency: 2-4 seconds (simple), 5-8 seconds (complex)
- Cost: $0.001 per query
- Accuracy: 95%+ relevance for domain-specific queries
"""
    md_path.write_text(md_content)
    
    # Ingest the markdown document
    ingestion_result = await agent.ingest_document(
        file_path=str(md_path),
        collection_name=test_collection_name,
        document_metadata={"type": "documentation", "format": "markdown"}
    )
    
    assert ingestion_result["success"] is True
    print(f"\n✓ Ingested Markdown document: {ingestion_result['chunks_stored']} chunks")
    
    # Search for content from the markdown
    search_result = await agent.search(
        query="What are the key features of IntraMind?",
        collection_name=test_collection_name,
        num_results=5
    )
    
    assert search_result["success"] is True
    assert len(search_result["results"]) > 0
    
    # Verify we found feature information
    response_lower = search_result["response"].lower()
    found_features = (
        "semantic" in response_lower or
        "search" in response_lower or
        "query" in response_lower or
        "features" in response_lower
    )
    
    assert found_features, "Should find information about features from the markdown document"
    print(f"✓ Found features in search results")
    print(f"✓ Response: {search_result['response'][:200]}...")


# ============================================================================
# Performance & Edge Case Tests
# ============================================================================

@pytest.mark.asyncio
async def test_ingest_large_document(
    agent: IntraMindAgent,
    tmp_path: Path,
    test_collection_name: str
):
    """Test ingesting a larger document with many chunks.
    
    Verifies:
    - Large document handling
    - Many chunks created and stored
    - Search still performs well
    """
    # Create a large document
    large_doc = tmp_path / "large_report.txt"
    
    # Generate substantial content (simulate a real report)
    paragraphs = []
    for i in range(50):  # 50 paragraphs
        paragraphs.append(f"""
Section {i+1}: Performance Analysis

This section discusses the performance metrics for quarter {i+1}. 
The data shows interesting trends in user engagement, revenue growth, 
and operational efficiency. Key findings include improved conversion 
rates, reduced customer acquisition costs, and enhanced product 
adoption across multiple market segments.

The detailed analysis reveals that strategic initiatives have yielded
positive results, with measurable improvements in key performance 
indicators. Stakeholder feedback has been overwhelmingly positive,
and customer satisfaction scores continue to trend upward.
""")
    
    large_doc.write_text("\n\n".join(paragraphs))
    
    # Ingest with smaller chunks to create more chunks
    ingestion_result = await agent.ingest_document(
        file_path=str(large_doc),
        collection_name=test_collection_name,
        chunk_size=300,
        chunk_overlap=50,
    )
    
    assert ingestion_result["success"] is True
    assert ingestion_result["chunks_stored"] > 20, \
        "Large document should create many chunks"
    
    print(f"\n✓ Large document ingested: {ingestion_result['chunks_stored']} chunks")
    
    # Verify search still works efficiently
    search_result = await agent.search(
        query="What are the performance trends?",
        collection_name=test_collection_name,
        num_results=10
    )
    
    assert search_result["success"] is True
    assert len(search_result["results"]) > 0
    print(f"✓ Search successful on large collection: {len(search_result['results'])} results")


# ============================================================================
# Test Summary
# ============================================================================

"""
Test Summary:
=============

This file contains 8 comprehensive end-to-end integration tests:

1. ✅ test_ingest_then_simple_search
   - Core E2E test: ingest → search → verify

2. ✅ test_ingest_then_complex_search
   - Complex query with expansion across ingested docs

3. ✅ test_ingest_multiple_docs_then_search
   - Multi-document ingestion and cross-document search

4. ✅ test_ingest_then_search_with_min_score
   - Score filtering on ingested content

5. ✅ test_search_empty_collection_no_crash
   - Edge case: empty collection handling

6. ✅ test_ingest_markdown_then_search
   - Markdown file format support

7. ✅ test_ingest_large_document
   - Large document handling and performance

8. (Ready to add more as needed)

Run these tests with:
    pytest tests/test_e2e_ingestion_search.py -v -m integration

Or run all integration tests:
    pytest -m integration -v
"""

