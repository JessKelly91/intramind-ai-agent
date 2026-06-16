"""Comprehensive tests for document ingestion workflow."""

import importlib
import pytest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Any

from agent.main import IntraMindAgent
from workflows.ingestion_workflow import (
    validate_document,
    extract_content,
    redact_pii,
    chunk_content,
    store_chunks,
    handle_error,
    ingestion_workflow,
    MAX_FILE_SIZE_BYTES,
)
from models.state import IngestionWorkflowState

ingestion_workflow_module = importlib.import_module("workflows.ingestion_workflow")


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_text_file(tmp_path: Path) -> Path:
    """Create a sample text file for testing."""
    file_path = tmp_path / "test_document.txt"
    file_path.write_text("This is a test document.\n\nIt has multiple paragraphs.\n\nAnd some content to chunk.")
    return file_path


@pytest.fixture
def sample_large_text_file(tmp_path: Path) -> Path:
    """Create a large text file for testing."""
    file_path = tmp_path / "large_document.txt"
    # Create content larger than MAX_FILE_SIZE_BYTES
    large_content = "x" * (MAX_FILE_SIZE_BYTES + 1000)
    file_path.write_text(large_content)
    return file_path


@pytest.fixture
def sample_empty_file(tmp_path: Path) -> Path:
    """Create an empty file for testing."""
    file_path = tmp_path / "empty.txt"
    file_path.write_text("")
    return file_path


@pytest.fixture
def sample_markdown_file(tmp_path: Path) -> Path:
    """Create a markdown file for testing."""
    file_path = tmp_path / "readme.md"
    content = """# Test Document

This is a markdown file with:
- Lists
- Headers
- Multiple paragraphs

## Section 2

More content here for testing chunking behavior.
"""
    file_path.write_text(content)
    return file_path


@pytest.fixture
def base_state() -> IngestionWorkflowState:
    """Create a base state for testing."""
    return {
        "messages": [],
        "user_query": "",
        "current_step": "start",
        "next_step": None,
        "workflow_complete": False,
        "search_strategy": None,
        "search_query": None,
        "search_results": None,
        "num_results": 0,
        "document_path": None,
        "document_type": None,
        "extracted_content": None,
        "document_metadata": {},
        "response": None,
        "citations": None,
        "error": None,
        "retry_count": 0,
        "file_path": "",
        "collection_name": "test_collection",
        "chunk_size": 100,
        "chunk_overlap": 20,
        "chunks": None,
        "inserted_ids": None,
    }


@pytest.fixture
def mock_api_client():
    """Create a mock API client."""
    client = MagicMock()
    client.insert_documents_batch = AsyncMock(
        return_value={"ids": ["id1", "id2", "id3"]}
    )
    return client


# ============================================================================
# 1. Validation Node Tests
# ============================================================================

@pytest.mark.asyncio
async def test_validate_document_success(base_state: IngestionWorkflowState, sample_text_file: Path):
    """Test successful document validation."""
    state = {**base_state, "file_path": str(sample_text_file)}
    
    result = await validate_document(state)
    
    assert result["error"] is None
    assert result["next_step"] == "extract_content"
    assert result["current_step"] == "validate_document"
    assert result["document_type"] == "text"
    assert result["document_metadata"]["filename"] == "test_document.txt"
    assert result["document_metadata"]["file_type"] == "text"
    assert result["document_metadata"]["file_size_bytes"] > 0


@pytest.mark.asyncio
async def test_validate_document_missing_file(base_state: IngestionWorkflowState):
    """Test validation fails when file doesn't exist."""
    state = {**base_state, "file_path": "/nonexistent/file.txt"}
    
    result = await validate_document(state)
    
    assert result["error"] is not None
    assert "does not exist" in result["error"]
    assert result["next_step"] == "handle_error"


@pytest.mark.asyncio
async def test_validate_document_no_file_path(base_state: IngestionWorkflowState):
    """Test validation fails when no file path provided."""
    state = {**base_state, "file_path": ""}
    
    result = await validate_document(state)
    
    assert result["error"] == "No file path provided"
    assert result["next_step"] == "handle_error"


@pytest.mark.asyncio
async def test_validate_document_unsupported_type(base_state: IngestionWorkflowState, tmp_path: Path):
    """Test validation fails for unsupported file types."""
    unsupported_file = tmp_path / "test.xyz"
    unsupported_file.write_text("content")
    
    state = {**base_state, "file_path": str(unsupported_file)}
    
    result = await validate_document(state)
    
    assert result["error"] is not None
    assert "Unsupported file type" in result["error"]
    assert result["next_step"] == "handle_error"


@pytest.mark.asyncio
async def test_validate_document_too_large(base_state: IngestionWorkflowState, sample_large_text_file: Path):
    """Test validation fails when file is too large."""
    state = {**base_state, "file_path": str(sample_large_text_file)}
    
    result = await validate_document(state)
    
    assert result["error"] is not None
    assert "too large" in result["error"].lower()
    assert result["next_step"] == "handle_error"


@pytest.mark.asyncio
async def test_validate_document_empty_file(base_state: IngestionWorkflowState, sample_empty_file: Path):
    """Test validation fails for empty files."""
    state = {**base_state, "file_path": str(sample_empty_file)}
    
    result = await validate_document(state)
    
    assert result["error"] == "File is empty"
    assert result["next_step"] == "handle_error"


@pytest.mark.asyncio
async def test_validate_document_no_collection_name(base_state: IngestionWorkflowState, sample_text_file: Path):
    """Test validation fails when no collection name provided."""
    state = {**base_state, "file_path": str(sample_text_file), "collection_name": ""}
    
    result = await validate_document(state)
    
    assert result["error"] == "No collection name provided"
    assert result["next_step"] == "handle_error"


@pytest.mark.asyncio
async def test_validate_document_supported_formats(base_state: IngestionWorkflowState, tmp_path: Path):
    """Test validation succeeds for all supported formats."""
    supported_extensions = [".txt", ".md", ".pdf", ".docx", ".pptx", ".png", ".jpg"]
    
    for ext in supported_extensions:
        test_file = tmp_path / f"test{ext}"
        test_file.write_bytes(b"test content")
        
        state = {**base_state, "file_path": str(test_file)}
        result = await validate_document(state)
        
        assert result["error"] is None, f"Failed for {ext}"
        assert result["next_step"] == "extract_content"


# ============================================================================
# 2. Content Extraction Tests
# ============================================================================

@pytest.mark.asyncio
async def test_extract_text_file(base_state: IngestionWorkflowState, sample_text_file: Path):
    """Test extraction from text file."""
    state = {
        **base_state,
        "file_path": str(sample_text_file),
        "document_type": "text",
        "current_step": "validate_document",
    }
    
    result = await extract_content(state)
    
    assert result["error"] is None
    assert result["next_step"] == "redact_pii"
    assert result["extracted_content"] is not None
    assert "test document" in result["extracted_content"]
    assert result["document_metadata"]["content_length"] > 0
    assert "extraction_timestamp" in result["document_metadata"]


@pytest.mark.asyncio
async def test_redact_pii_permissive_passes_when_redactor_unavailable(
    base_state: IngestionWorkflowState, monkeypatch
):
    """Permissive mode should preserve local/dev ingestion when Presidio is unavailable."""
    monkeypatch.setattr(ingestion_workflow_module.settings, "enable_pii_redaction", True)
    monkeypatch.setattr(
        ingestion_workflow_module.settings, "pii_redaction_required", False
    )
    mock_redactor = MagicMock()
    mock_redactor.available = False
    state = {
        **base_state,
        "extracted_content": "Contact Jane at jane@example.com",
        "document_metadata": {},
    }

    with patch(
        "workflows.ingestion_workflow.get_default_redactor",
        return_value=mock_redactor,
    ):
        result = await redact_pii(state)

    assert result["next_step"] == "chunk_content"
    assert result["error"] is None
    assert result["extracted_content"] == "Contact Jane at jane@example.com"
    assert result["document_metadata"]["pii_redaction_applied"] is False
    assert result["document_metadata"]["pii_redaction_required"] is False
    assert (
        result["document_metadata"]["pii_redaction_skipped_reason"]
        == "redactor_unavailable"
    )


@pytest.mark.asyncio
async def test_redact_pii_required_blocks_when_redactor_unavailable(
    base_state: IngestionWorkflowState, monkeypatch
):
    """Required mode should fail closed instead of storing unredacted content."""
    monkeypatch.setattr(ingestion_workflow_module.settings, "enable_pii_redaction", True)
    monkeypatch.setattr(
        ingestion_workflow_module.settings, "pii_redaction_required", True
    )
    mock_redactor = MagicMock()
    mock_redactor.available = False
    state = {
        **base_state,
        "extracted_content": "Contact Jane at jane@example.com",
        "document_metadata": {},
    }

    with patch(
        "workflows.ingestion_workflow.get_default_redactor",
        return_value=mock_redactor,
    ):
        result = await redact_pii(state)

    assert result["next_step"] == "handle_error"
    assert "PII redactor unavailable" in result["error"]
    assert result["document_metadata"]["pii_redaction_applied"] is False
    assert result["document_metadata"]["pii_redaction_required"] is True
    assert (
        result["document_metadata"]["pii_redaction_skipped_reason"]
        == "redactor_unavailable"
    )


@pytest.mark.asyncio
async def test_redact_pii_required_blocks_when_redaction_disabled(
    base_state: IngestionWorkflowState, monkeypatch
):
    """Contradictory production config should fail before chunk storage."""
    monkeypatch.setattr(ingestion_workflow_module.settings, "enable_pii_redaction", False)
    monkeypatch.setattr(
        ingestion_workflow_module.settings, "pii_redaction_required", True
    )
    state = {
        **base_state,
        "extracted_content": "Contact Jane at jane@example.com",
        "document_metadata": {},
    }

    result = await redact_pii(state)

    assert result["next_step"] == "handle_error"
    assert result["error"] == "PII redaction is required but disabled"
    assert result["document_metadata"]["pii_redaction_required"] is True
    assert (
        result["document_metadata"]["pii_redaction_skipped_reason"]
        == "redaction_disabled"
    )


@pytest.mark.asyncio
async def test_extract_markdown_file(base_state: IngestionWorkflowState, sample_markdown_file: Path):
    """Test extraction from markdown file."""
    state = {
        **base_state,
        "file_path": str(sample_markdown_file),
        "document_type": "text",
    }
    
    result = await extract_content(state)
    
    assert result["error"] is None
    assert "Test Document" in result["extracted_content"]
    assert "Section 2" in result["extracted_content"]


@pytest.mark.asyncio
async def test_extract_text_encoding_detection(base_state: IngestionWorkflowState, tmp_path: Path):
    """Test text extraction with encoding detection."""
    # Create a UTF-8 file
    test_file = tmp_path / "utf8.txt"
    test_file.write_text("Testing UTF-8: héllo wörld 🌍", encoding="utf-8")
    
    state = {
        **base_state,
        "file_path": str(test_file),
        "document_type": "text",
    }
    
    result = await extract_content(state)
    
    assert result["error"] is None
    assert "héllo wörld" in result["extracted_content"]
    assert result["document_metadata"]["encoding"] is not None


@pytest.mark.asyncio
async def test_extract_content_empty_file(base_state: IngestionWorkflowState, tmp_path: Path):
    """Test extraction fails for file with no content."""
    empty_file = tmp_path / "empty.txt"
    empty_file.write_text("   \n\n   ")  # Only whitespace
    
    state = {
        **base_state,
        "file_path": str(empty_file),
        "document_type": "text",
    }
    
    result = await extract_content(state)
    
    assert result["error"] == "No content extracted from file"
    assert result["next_step"] == "handle_error"


@pytest.mark.asyncio
async def test_extract_pdf_mock(base_state: IngestionWorkflowState, tmp_path: Path):
    """Test PDF extraction with mocked pypdf."""
    pdf_file = tmp_path / "test.pdf"
    pdf_file.write_bytes(b"fake pdf content")
    
    # Mock the PdfReader
    with patch("pypdf.PdfReader") as mock_reader:
        mock_page = MagicMock()
        mock_page.extract_text.return_value = "This is page 1 content"
        
        mock_pdf = MagicMock()
        mock_pdf.pages = [mock_page]
        mock_pdf.pdf_header = "1.7"
        mock_pdf.metadata = {
            "/Author": "Test Author",
            "/Title": "Test Title",
        }
        
        mock_reader.return_value = mock_pdf
        
        state = {
            **base_state,
            "file_path": str(pdf_file),
            "document_type": "pdf",
        }
        
        result = await extract_content(state)
        
        assert result["error"] is None
        assert "page 1 content" in result["extracted_content"].lower()
        assert result["document_metadata"]["page_count"] == 1
        assert result["document_metadata"]["author"] == "Test Author"


@pytest.mark.asyncio
async def test_extract_docx_mock(base_state: IngestionWorkflowState, tmp_path: Path):
    """Test DOCX extraction with mocked python-docx."""
    docx_file = tmp_path / "test.docx"
    docx_file.write_bytes(b"fake docx content")
    
    with patch("docx.Document") as mock_doc_class:
        mock_para1 = MagicMock()
        mock_para1.text = "First paragraph"
        mock_para2 = MagicMock()
        mock_para2.text = "Second paragraph"
        
        mock_doc = MagicMock()
        mock_doc.paragraphs = [mock_para1, mock_para2]
        mock_doc.tables = []
        mock_doc.core_properties.author = "Doc Author"
        mock_doc.core_properties.title = "Doc Title"
        mock_doc.core_properties.created = None
        mock_doc.core_properties.modified = None
        
        mock_doc_class.return_value = mock_doc
        
        state = {
            **base_state,
            "file_path": str(docx_file),
            "document_type": "docx",
        }
        
        result = await extract_content(state)
        
        assert result["error"] is None
        assert "First paragraph" in result["extracted_content"]
        assert "Second paragraph" in result["extracted_content"]
        assert result["document_metadata"]["author"] == "Doc Author"


@pytest.mark.asyncio
async def test_extract_pptx_mock(base_state: IngestionWorkflowState, tmp_path: Path):
    """Test PPTX extraction with mocked python-pptx."""
    pptx_file = tmp_path / "test.pptx"
    pptx_file.write_bytes(b"fake pptx content")
    
    with patch("pptx.Presentation") as mock_prs_class:
        mock_shape1 = MagicMock()
        mock_shape1.text = "Slide 1 Title"
        mock_shape2 = MagicMock()
        mock_shape2.text = "Slide 1 Content"
        
        mock_slide = MagicMock()
        mock_slide.shapes = [mock_shape1, mock_shape2]
        
        mock_prs = MagicMock()
        mock_prs.slides = [mock_slide]
        mock_prs.core_properties.author = "Presentation Author"
        mock_prs.core_properties.title = "Presentation Title"
        mock_prs.core_properties.created = None
        mock_prs.core_properties.modified = None
        
        mock_prs_class.return_value = mock_prs
        
        state = {
            **base_state,
            "file_path": str(pptx_file),
            "document_type": "pptx",
        }
        
        result = await extract_content(state)
        
        assert result["error"] is None
        assert "Slide 1" in result["extracted_content"]
        assert "Slide 1 Title" in result["extracted_content"]
        assert result["document_metadata"]["slide_count"] == 1


# ============================================================================
# 3. Chunking Tests
# ============================================================================

@pytest.mark.asyncio
async def test_chunk_content_simple(base_state: IngestionWorkflowState):
    """Test basic content chunking."""
    long_content = "This is a test. " * 100  # Create long content
    
    state = {
        **base_state,
        "extracted_content": long_content,
        "chunk_size": 100,
        "chunk_overlap": 20,
    }
    
    result = await chunk_content(state)
    
    assert result["error"] is None
    assert result["next_step"] == "store_chunks"
    assert result["chunks"] is not None
    assert len(result["chunks"]) > 1
    assert result["document_metadata"]["total_chunks"] == len(result["chunks"])
    
    # Verify chunk structure
    for i, chunk in enumerate(result["chunks"]):
        assert "content" in chunk
        assert "index" in chunk
        assert chunk["index"] == i
        assert "char_count" in chunk
        assert len(chunk["content"]) > 0


@pytest.mark.asyncio
async def test_chunk_content_with_overlap(base_state: IngestionWorkflowState):
    """Test that chunk overlap is applied correctly."""
    content = "A" * 150 + "B" * 150  # 300 characters
    
    state = {
        **base_state,
        "extracted_content": content,
        "chunk_size": 100,
        "chunk_overlap": 20,
    }
    
    result = await chunk_content(state)
    
    assert result["error"] is None
    chunks = result["chunks"]
    assert len(chunks) >= 2
    
    # Check that overlap exists (rough check)
    # With 300 chars, 100 chunk size, 20 overlap, we expect multiple chunks
    assert len(chunks) >= 2


@pytest.mark.asyncio
async def test_chunk_content_small_text(base_state: IngestionWorkflowState):
    """Test chunking with text smaller than chunk size."""
    small_content = "This is a small document."
    
    state = {
        **base_state,
        "extracted_content": small_content,
        "chunk_size": 1000,
        "chunk_overlap": 200,
    }
    
    result = await chunk_content(state)
    
    assert result["error"] is None
    assert len(result["chunks"]) == 1
    assert result["chunks"][0]["content"] == small_content


@pytest.mark.asyncio
async def test_chunk_content_preserves_structure(base_state: IngestionWorkflowState):
    """Test that chunking preserves paragraph structure."""
    content = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
    
    state = {
        **base_state,
        "extracted_content": content,
        "chunk_size": 50,
        "chunk_overlap": 10,
    }
    
    result = await chunk_content(state)
    
    assert result["error"] is None
    # RecursiveCharacterTextSplitter should respect \n\n boundaries
    assert result["chunks"] is not None


@pytest.mark.asyncio
async def test_chunk_content_metadata(base_state: IngestionWorkflowState):
    """Test that chunk metadata is properly set."""
    state = {
        **base_state,
        "extracted_content": "Test content " * 50,
        "chunk_size": 100,
        "chunk_overlap": 20,
        "document_metadata": {"existing": "metadata"},
    }
    
    result = await chunk_content(state)
    
    assert result["error"] is None
    metadata = result["document_metadata"]
    assert metadata["existing"] == "metadata"
    assert metadata["total_chunks"] == len(result["chunks"])
    assert metadata["chunk_size"] == 100
    assert metadata["chunk_overlap"] == 20


# ============================================================================
# 4. Storage Tests
# ============================================================================

@pytest.mark.asyncio
async def test_store_chunks_success(base_state: IngestionWorkflowState, mock_api_client):
    """Test successful chunk storage."""
    chunks = [
        {"content": "Chunk 1", "index": 0, "char_count": 7},
        {"content": "Chunk 2", "index": 1, "char_count": 7},
        {"content": "Chunk 3", "index": 2, "char_count": 7},
    ]
    
    state = {
        **base_state,
        "chunks": chunks,
        "collection_name": "test_collection",
        "document_metadata": {"filename": "test.txt"},
    }
    
    with patch("workflows.ingestion_workflow.get_api_client", return_value=mock_api_client):
        result = await store_chunks(state)
    
    assert result["error"] is None
    assert result["workflow_complete"] is True
    assert result["inserted_ids"] == ["id1", "id2", "id3"]
    assert result["document_metadata"]["inserted_count"] == 3
    assert result["document_metadata"]["failed_count"] == 0
    
    # Verify API client was called correctly
    mock_api_client.insert_documents_batch.assert_called_once()
    call_args = mock_api_client.insert_documents_batch.call_args
    assert call_args.kwargs["collection_name"] == "test_collection"
    assert len(call_args.kwargs["documents"]) == 3


@pytest.mark.asyncio
async def test_store_chunks_with_metadata(base_state: IngestionWorkflowState, mock_api_client):
    """Test that metadata is properly attached to stored chunks."""
    chunks = [
        {"content": "Test chunk", "index": 0, "char_count": 10},
    ]
    
    state = {
        **base_state,
        "chunks": chunks,
        "collection_name": "test_collection",
        "document_metadata": {
            "filename": "test.txt",
            "file_type": "text",
            "author": "Test Author",
        },
    }
    
    with patch("workflows.ingestion_workflow.get_api_client", return_value=mock_api_client):
        result = await store_chunks(state)
    
    assert result["error"] is None
    
    # Check that metadata was passed correctly
    call_args = mock_api_client.insert_documents_batch.call_args
    documents = call_args.kwargs["documents"]
    
    doc_metadata = documents[0]["metadata"]
    assert doc_metadata["filename"] == "test.txt"
    assert doc_metadata["file_type"] == "text"
    assert doc_metadata["author"] == "Test Author"
    assert doc_metadata["chunk_index"] == 0
    assert doc_metadata["total_chunks"] == 1
    assert "ingestion_timestamp" in doc_metadata


@pytest.mark.asyncio
async def test_store_chunks_failure(base_state: IngestionWorkflowState, mock_api_client):
    """Test storage failure handling."""
    chunks = [{"content": "Test", "index": 0, "char_count": 4}]
    
    state = {
        **base_state,
        "chunks": chunks,
    }
    
    # Mock API client to raise an exception
    mock_api_client.insert_documents_batch.side_effect = Exception("Storage error")
    
    with patch("workflows.ingestion_workflow.get_api_client", return_value=mock_api_client):
        result = await store_chunks(state)
    
    assert result["error"] is not None
    assert "Storage failed" in result["error"]
    assert result["next_step"] == "handle_error"


@pytest.mark.asyncio
async def test_store_chunks_partial_success(base_state: IngestionWorkflowState, mock_api_client):
    """Test handling of partial insertion."""
    chunks = [
        {"content": "Chunk 1", "index": 0, "char_count": 7},
        {"content": "Chunk 2", "index": 1, "char_count": 7},
        {"content": "Chunk 3", "index": 2, "char_count": 7},
    ]
    
    state = {
        **base_state,
        "chunks": chunks,
    }
    
    # Mock only 2 out of 3 insertions succeeded
    mock_api_client.insert_documents_batch.return_value = {"ids": ["id1", "id2"]}
    
    with patch("workflows.ingestion_workflow.get_api_client", return_value=mock_api_client):
        result = await store_chunks(state)
    
    assert result["error"] is None  # Still succeeds
    assert len(result["inserted_ids"]) == 2
    assert result["document_metadata"]["inserted_count"] == 2
    assert result["document_metadata"]["failed_count"] == 1


# ============================================================================
# 5. Error Handling Tests
# ============================================================================

@pytest.mark.asyncio
async def test_handle_error(base_state: IngestionWorkflowState):
    """Test error handler node."""
    state = {
        **base_state,
        "error": "Test error message",
    }
    
    result = await handle_error(state)
    
    assert result["current_step"] == "handle_error"
    assert result["workflow_complete"] is True
    assert result["error"] == "Test error message"


# ============================================================================
# 6. Integration Tests (Full Workflow)
# ============================================================================

@pytest.mark.asyncio
async def test_full_workflow_text_file(sample_text_file: Path, mock_api_client):
    """Test complete workflow with real text file."""
    initial_state: IngestionWorkflowState = {
        "messages": [],
        "user_query": "",
        "current_step": "start",
        "next_step": None,
        "workflow_complete": False,
        "search_strategy": None,
        "search_query": None,
        "search_results": None,
        "num_results": 0,
        "document_path": None,
        "document_type": None,
        "extracted_content": None,
        "document_metadata": {},
        "response": None,
        "citations": None,
        "error": None,
        "retry_count": 0,
        "file_path": str(sample_text_file),
        "collection_name": "test_collection",
        "chunk_size": 50,
        "chunk_overlap": 10,
        "chunks": None,
        "inserted_ids": None,
    }
    
    with patch("workflows.ingestion_workflow.get_api_client", return_value=mock_api_client):
        result = await ingestion_workflow.ainvoke(initial_state)
    
    # Verify workflow completed successfully
    assert result["workflow_complete"] is True
    assert result["error"] is None
    assert result["inserted_ids"] is not None
    assert len(result["inserted_ids"]) > 0
    
    # Verify all stages completed
    assert result["document_type"] == "text"
    assert result["extracted_content"] is not None
    assert result["chunks"] is not None
    assert len(result["chunks"]) > 0


@pytest.mark.asyncio
async def test_workflow_with_custom_metadata(sample_text_file: Path, mock_api_client):
    """Test workflow preserves custom metadata."""
    custom_metadata = {
        "author": "Test Author",
        "department": "Engineering",
        "year": 2024,
    }
    
    initial_state: IngestionWorkflowState = {
        "messages": [],
        "user_query": "",
        "current_step": "start",
        "next_step": None,
        "workflow_complete": False,
        "search_strategy": None,
        "search_query": None,
        "search_results": None,
        "num_results": 0,
        "document_path": None,
        "document_type": None,
        "extracted_content": None,
        "document_metadata": custom_metadata,
        "response": None,
        "citations": None,
        "error": None,
        "retry_count": 0,
        "file_path": str(sample_text_file),
        "collection_name": "test_collection",
        "chunk_size": 100,
        "chunk_overlap": 20,
        "chunks": None,
        "inserted_ids": None,
    }
    
    with patch("workflows.ingestion_workflow.get_api_client", return_value=mock_api_client):
        result = await ingestion_workflow.ainvoke(initial_state)
    
    assert result["error"] is None
    
    # Check that custom metadata was preserved and passed to storage
    call_args = mock_api_client.insert_documents_batch.call_args
    documents = call_args.kwargs["documents"]
    
    # All documents should have custom metadata
    for doc in documents:
        assert doc["metadata"]["author"] == "Test Author"
        assert doc["metadata"]["department"] == "Engineering"
        assert doc["metadata"]["year"] == 2024


@pytest.mark.asyncio
async def test_workflow_error_handling(tmp_path: Path, mock_api_client):
    """Test workflow handles errors gracefully."""
    # Create an unsupported file
    bad_file = tmp_path / "test.xyz"
    bad_file.write_text("content")
    
    initial_state: IngestionWorkflowState = {
        "messages": [],
        "user_query": "",
        "current_step": "start",
        "next_step": None,
        "workflow_complete": False,
        "search_strategy": None,
        "search_query": None,
        "search_results": None,
        "num_results": 0,
        "document_path": None,
        "document_type": None,
        "extracted_content": None,
        "document_metadata": {},
        "response": None,
        "citations": None,
        "error": None,
        "retry_count": 0,
        "file_path": str(bad_file),
        "collection_name": "test_collection",
        "chunk_size": 100,
        "chunk_overlap": 20,
        "chunks": None,
        "inserted_ids": None,
    }
    
    with patch("workflows.ingestion_workflow.get_api_client", return_value=mock_api_client):
        result = await ingestion_workflow.ainvoke(initial_state)
    
    # Workflow should complete but with error
    assert result["workflow_complete"] is True
    assert result["error"] is not None
    assert "Unsupported file type" in result["error"]


# ============================================================================
# 7. Agent Integration Tests
# ============================================================================

@pytest.mark.asyncio
async def test_agent_ingest_document_success(sample_text_file: Path, mock_api_client):
    """Test agent ingest_document method."""
    agent = IntraMindAgent()
    
    with patch("workflows.ingestion_workflow.get_api_client", return_value=mock_api_client):
        result = await agent.ingest_document(
            file_path=str(sample_text_file),
            collection_name="test_collection",
            chunk_size=100,
            chunk_overlap=20,
        )
    
    assert result["success"] is True
    assert result["file_name"] == "test_document.txt"
    assert result["file_type"] == "text"
    assert result["collection_name"] == "test_collection"
    assert result["chunks_created"] > 0
    assert result["chunks_stored"] > 0
    assert len(result["inserted_ids"]) > 0


@pytest.mark.asyncio
async def test_agent_ingest_document_failure(tmp_path: Path):
    """Test agent handles ingestion failures."""
    agent = IntraMindAgent()
    
    # Try to ingest non-existent file
    result = await agent.ingest_document(
        file_path="/nonexistent/file.txt",
        collection_name="test_collection",
    )
    
    assert result["success"] is False
    assert result["error"] is not None


@pytest.mark.asyncio
async def test_agent_ingest_document_custom_params(sample_text_file: Path, mock_api_client):
    """Test agent with custom parameters."""
    agent = IntraMindAgent()
    
    custom_metadata = {
        "category": "documentation",
        "version": "1.0",
    }
    
    with patch("workflows.ingestion_workflow.get_api_client", return_value=mock_api_client):
        result = await agent.ingest_document(
            file_path=str(sample_text_file),
            collection_name="custom_collection",
            chunk_size=200,
            chunk_overlap=50,
            document_metadata=custom_metadata,
        )
    
    assert result["success"] is True
    assert result["collection_name"] == "custom_collection"
    assert result["chunk_size"] == 200
    assert result["chunk_overlap"] == 50
    
    # Verify custom metadata was passed through
    call_args = mock_api_client.insert_documents_batch.call_args
    documents = call_args.kwargs["documents"]
    assert documents[0]["metadata"]["category"] == "documentation"
    assert documents[0]["metadata"]["version"] == "1.0"


@pytest.mark.asyncio
async def test_agent_ingest_document_with_different_file_types(tmp_path: Path, mock_api_client):
    """Test agent with various file types."""
    agent = IntraMindAgent()
    
    # Test with markdown
    md_file = tmp_path / "test.md"
    md_file.write_text("# Test\n\nMarkdown content")
    
    with patch("workflows.ingestion_workflow.get_api_client", return_value=mock_api_client):
        result = await agent.ingest_document(
            file_path=str(md_file),
            collection_name="test_collection",
        )
    
    assert result["success"] is True
    assert result["file_type"] == "text"
    assert result["file_name"] == "test.md"

