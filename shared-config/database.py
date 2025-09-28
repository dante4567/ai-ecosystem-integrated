#!/usr/bin/env python3

import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy import MetaData
import logging

logger = logging.getLogger(__name__)

class Base(DeclarativeBase):
    metadata = MetaData()

class DatabaseConfig:
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL", "postgresql+asyncpg://user:password@localhost:5432/ai_ecosystem")
        self.engine = None
        self.session_maker = None

    async def initialize(self):
        """Initialize database connection"""
        try:
            self.engine = create_async_engine(
                self.database_url,
                echo=os.getenv("DB_ECHO", "false").lower() == "true",
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
                pool_recycle=3600,
            )

            self.session_maker = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            logger.info("Database connection initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False

    async def create_tables(self):
        """Create all tables"""
        if self.engine:
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)
                logger.info("Database tables created successfully")

    async def get_session(self):
        """Get database session"""
        if not self.session_maker:
            await self.initialize()

        return self.session_maker()

    async def close(self):
        """Close database connection"""
        if self.engine:
            await self.engine.dispose()
            logger.info("Database connection closed")

# Global database instance
db_config = DatabaseConfig()

async def get_db():
    """Dependency for getting database session"""
    async with db_config.get_session() as session:
        try:
            yield session
        finally:
            await session.close()