-- AI Ecosystem Database Initialization
-- This script sets up the initial database structure and extensions

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Create schemas (optional, but good practice)
-- Tables will be created automatically by SQLAlchemy

-- Set default configurations
ALTER DATABASE ai_ecosystem SET timezone TO 'UTC';

-- Create indexes for common query patterns (will be created after tables exist)
-- These will be handled by SQLAlchemy migrations