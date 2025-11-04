# Changelog

All notable changes to the IntraMind AI Agent will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- LangGraph state machine architecture for intelligent document search
- Hybrid LLM strategy (local Ollama + cloud LLMs) for cost optimization
- Query classification and multi-query expansion workflows
- Document ingestion pipeline supporting PDF, DOCX, PPTX, TXT, MD, and images
- Comprehensive test suite (94 tests with 67% coverage)
- Search quality filtering with `min_score` parameter
- CLI interface with interactive and single-query modes
- Metrics and observability system with persistent tracking
- Complete documentation (README, QUICKSTART, WORKFLOWS, OBSERVABILITY, PORTFOLIO_WRITEUP)
- Architecture documentation with Mermaid diagrams
- End-to-end integration tests

### Features
- Semantic search across enterprise documents
- Automatic query complexity routing (simple vs. complex)
- Result synthesis with AI-powered natural language responses
- Document citations and metadata preservation
- Streaming search support
- Health checks and monitoring
- Async/await throughout for performance

---

## How to Use This Changelog

When you make your first public release:
1. Move items from `[Unreleased]` to a new version section (e.g., `[1.0.0] - 2025-XX-XX`)
2. Add the release date
3. Keep the `[Unreleased]` section for future changes

For subsequent releases, categorize changes as:
- **Added** for new features
- **Changed** for changes in existing functionality
- **Deprecated** for soon-to-be removed features
- **Removed** for now removed features
- **Fixed** for any bug fixes
- **Security** for vulnerability fixes

---

**Project Status**: Pre-release development  
**Current Version**: Unreleased  
**Maintainer**: Jess Kelly

