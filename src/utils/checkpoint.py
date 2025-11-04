"""Checkpoint management for conversation memory."""

import asyncio
import logging
from pathlib import Path
from typing import Any

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from config import settings

logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages conversation checkpoints using LangGraph's AsyncSqliteSaver."""

    _instance: "CheckpointManager | None" = None
    _checkpointer: AsyncSqliteSaver | None = None
    _conn: aiosqlite.Connection | None = None
    _db_path: str | None = None
    _event_loop: asyncio.AbstractEventLoop | None = None

    def __new__(cls) -> "CheckpointManager":
        """Singleton pattern to ensure one checkpointer instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize checkpoint manager (called once due to singleton)."""
        if self._db_path is None:
            self._initialize_path()

    def _initialize_path(self) -> None:
        """Store the database path for lazy initialization."""
        if not settings.enable_conversation_memory:
            logger.info("Conversation memory disabled in config")
            return

        # Ensure checkpoint directory exists
        checkpoint_path = Path(settings.checkpoint_storage_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self._db_path = str(checkpoint_path)
        logger.info(f"Checkpoint storage path configured: {self._db_path}")

    async def _ensure_checkpointer(self) -> None:
        """Ensure the checkpointer is initialized (lazy async initialization).
        
        Re-initializes if we detect a different event loop (e.g., in tests).
        """
        if self._db_path is None:
            return
        
        # Check if we're in a different event loop than when checkpointer was created
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            # No event loop running
            current_loop = None
        
        # If event loop changed, we need to re-initialize
        if self._checkpointer is not None and self._event_loop is not current_loop:
            logger.info(f"Event loop changed, re-initializing checkpointer")
            # Close old connection if it exists
            if self._conn:
                try:
                    await self._conn.close()
                except Exception:
                    pass
            self._checkpointer = None
            self._conn = None
        
        # If checkpointer exists and event loop matches, we're good
        if self._checkpointer is not None:
            return
        
        try:
            # Create async SQLite connection
            self._conn = await aiosqlite.connect(self._db_path)
            # Create AsyncSqliteSaver with the connection
            self._checkpointer = AsyncSqliteSaver(self._conn)
            # Setup the database schema
            await self._checkpointer.setup()
            # Remember which event loop we initialized in
            self._event_loop = current_loop
            logger.info(f"Async checkpoint storage initialized at {self._db_path}")
        except Exception as e:
            logger.error(f"Failed to initialize checkpoint storage: {e}")
            raise

    async def get_checkpointer(self) -> AsyncSqliteSaver | None:
        """Get the checkpointer instance (async to support lazy initialization).
        
        Returns:
            AsyncSqliteSaver instance or None if memory is disabled
        """
        await self._ensure_checkpointer()
        return self._checkpointer
    
    def get_checkpointer_sync(self) -> AsyncSqliteSaver | None:
        """Get the checkpointer instance synchronously (for workflow compilation).
        
        WARNING: Returns existing checkpointer or None. Does NOT initialize.
        The checkpointer will be lazily initialized on first async use.
        This is safe because LangGraph will create the checkpointer lazily if needed.
        
        Returns:
            AsyncSqliteSaver instance or None if memory is disabled or not yet initialized
        """
        if self._db_path is None:
            return None
        
        # Return existing checkpointer (may be None if not yet initialized)
        # This is OK - LangGraph will handle lazy initialization on first async use
        return self._checkpointer

    def is_enabled(self) -> bool:
        """Check if conversation memory is enabled.
        
        Returns:
            True if memory is enabled (DB path is configured)
        """
        return self._db_path is not None

    async def get_conversation_history(
        self, thread_id: str, limit: int | None = None
    ) -> list[dict[str, Any]]:
        """Get conversation history for a thread.
        
        Args:
            thread_id: Unique identifier for the conversation thread
            limit: Maximum number of messages to retrieve (None = all)
            
        Returns:
            List of conversation messages
        """
        if not self.is_enabled():
            return []

        try:
            checkpointer = await self.get_checkpointer()
            if not checkpointer:
                return []
            
            # Get checkpoint data for thread
            config = {"configurable": {"thread_id": thread_id}}
            checkpoint = await checkpointer.aget(config)
            
            if not checkpoint:
                return []
            
            # Extract messages from checkpoint
            messages = checkpoint.get("channel_values", {}).get("messages", [])
            
            # Apply limit if specified
            if limit and len(messages) > limit:
                messages = messages[-limit:]
            
            return messages
            
        except Exception as e:
            logger.error(f"Failed to retrieve conversation history: {e}")
            return []

    async def clear_conversation(self, thread_id: str) -> bool:
        """Clear conversation history for a thread.
        
        Args:
            thread_id: Unique identifier for the conversation thread
            
        Returns:
            True if cleared successfully, False otherwise
        """
        if not self.is_enabled():
            return False

        try:
            checkpointer = await self.get_checkpointer()
            if not checkpointer:
                return False
            
            # Clear by setting empty checkpoint
            config = {"configurable": {"thread_id": thread_id}}
            await checkpointer.aput(config, {}, {})
            logger.info(f"Cleared conversation history for thread: {thread_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear conversation history: {e}")
            return False

    async def list_threads(self) -> list[str]:
        """List all conversation thread IDs.
        
        Returns:
            List of thread IDs
        """
        if not self.is_enabled():
            return []

        try:
            checkpointer = await self.get_checkpointer()
            if not checkpointer:
                return []
            
            # Query all checkpoints and extract thread IDs
            # This is a simple implementation - production might need pagination
            threads = []
            async for checkpoint in checkpointer.alist({}):
                thread_id = checkpoint.config.get("configurable", {}).get("thread_id")
                if thread_id and thread_id not in threads:
                    threads.append(thread_id)
            
            return threads
            
        except Exception as e:
            logger.error(f"Failed to list threads: {e}")
            return []

    async def aclose(self) -> None:
        """Close the async checkpointer connection."""
        if self._conn:
            try:
                await self._conn.close()
                logger.info("Checkpoint manager closed")
            except Exception as e:
                logger.error(f"Error closing checkpoint connection: {e}")


# Global checkpoint manager instance
checkpoint_manager = CheckpointManager()


def get_checkpointer() -> AsyncSqliteSaver | None:
    """Get the global checkpointer instance synchronously (for workflow compilation).
    
    Returns:
        AsyncSqliteSaver instance or None if memory is disabled
    """
    return checkpoint_manager.get_checkpointer_sync()


async def create_checkpointer() -> AsyncSqliteSaver | None:
    """Create a new checkpointer instance for the current event loop.
    
    This creates a fresh checkpointer that's bound to the current event loop,
    avoiding issues with reusing checkpointers across different event loops.
    
    Returns:
        New AsyncSqliteSaver instance or None if memory is disabled
    """
    if not settings.enable_conversation_memory:
        return None
    
    checkpoint_path = Path(settings.checkpoint_storage_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        # Create fresh connection and checkpointer for this event loop
        conn = await aiosqlite.connect(str(checkpoint_path))
        checkpointer = AsyncSqliteSaver(conn)
        await checkpointer.setup()
        logger.info(f"Created new checkpointer for current event loop at {checkpoint_path}")
        return checkpointer
    except Exception as e:
        logger.error(f"Failed to create checkpointer: {e}")
        return None

