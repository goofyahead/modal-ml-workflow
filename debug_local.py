#!/usr/bin/env python3
"""
Simple debugging script - just call the functions directly
"""

import asyncio
import os
from dotenv import load_dotenv

load_dotenv()

# Import the raw functions for debugging
from main import _raw_get_all_meetings_with_metadata, _raw_semantic_search

async def test_get_meetings():
    """Test get all meetings"""
    print("Testing get_all_meetings_with_metadata...")
    result = await _raw_get_all_meetings_with_metadata(limit=2)
    print(f"Got {result['total_count']} total meetings")
    if result['meetings']:
        first = result['meetings'][0]
        print(f"First meeting: {first['filename']}")
        print(f"Metadata type: {type(first['meeting_metadata'])}")

async def test_search():
    """Test semantic search"""
    print("\nTesting semantic_search...")
    results = await _raw_semantic_search("client meeting", limit=2)
    print(f"Found {len(results)} results")
    if results:
        print(f"First result: {results[0]['filename']}")

async def main():
    """Simple test runner"""
    print("🔧 Simple Debug Script")
    
    # Quick env check
    db_url = os.environ.get("DATABASE_URL", "")
    print(f"DB: {'✅' if db_url else '❌'}")
    
    if not db_url:
        print("Need DATABASE_URL in .env file")
        return
    
    await test_get_meetings()
    await test_search()

if __name__ == "__main__":
    asyncio.run(main())