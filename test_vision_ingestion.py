#!/usr/bin/env python3
"""
Test script for vision ingestion with per-page processing.
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env if available
try:
    from dotenv import load_dotenv
    env_file = project_root / '.env'
    if env_file.exists():
        load_dotenv(env_file)
except ImportError:
    pass

# Set up environment
os.environ.setdefault('FLASK_APP', 'run_server.py')

from external.ai_bulk_doc_analysis.ingestion_service import IngestionService, DEFAULT_VISION_PROMPT

def test_vision_ingestion():
    """Test vision ingestion with a sample PDF."""
    
    # Find a test PDF
    test_pdfs = [
        project_root / "test_document.pdf",
        project_root / "car.pdf",
        project_root / "data" / "ai_bulk_doc_analysis" / "docs" / "021aa086-5520-437b-b7af-15475effdf0b" / "test_document.pdf",
    ]
    
    pdf_path = None
    for pdf in test_pdfs:
        if pdf.exists():
            pdf_path = pdf
            break
    
    if not pdf_path:
        print("❌ No test PDF found. Please provide a PDF file path.")
        return False
    
    print(f"📄 Using test PDF: {pdf_path}")
    print(f"   Size: {pdf_path.stat().st_size / 1024:.2f} KB")
    
    # Create ingestion service
    service = IngestionService()
    
    # Create vision ingestion profile with config
    print("\n🔧 Creating vision ingestion profile...")
    vision_config = {
        "dpi": 200,
        "image_format": "PNG",
        "jpeg_quality": 85,
        "model": "claude-3-opus-20240229",
        "max_tokens": 4096,
        "temperature": 0.0
    }
    
    try:
        profile = service.create_ingestion_profile(
            name="Test Vision Profile",
            accepted_input_types=["PDF"],
            mode="vision",
            vision_prompt=DEFAULT_VISION_PROMPT,
            vision_config=vision_config
        )
        print(f"✅ Created profile: {profile.ingestion_profile_id}")
    except Exception as e:
        print(f"❌ Failed to create profile: {e}")
        return False
    
    # Get the full profile object from DB
    db_profile = service.get_ingestion_profile(profile.ingestion_profile_id)
    if not db_profile:
        print("❌ Failed to retrieve profile from database")
        return False
    
    # Verify vision_config was stored
    if db_profile.metadata_json and isinstance(db_profile.metadata_json, dict):
        stored_config = db_profile.metadata_json.get("vision_config", {})
        print(f"\n📋 Stored vision config:")
        for key, value in stored_config.items():
            print(f"   {key}: {value}")
    else:
        print("⚠️  Warning: vision_config not found in metadata_json")
    
    # Test ingestion
    print(f"\n🔄 Starting vision ingestion...")
    print(f"   Processing: {pdf_path.name}")
    print(f"   Mode: vision (per-page processing)")
    print(f"   Model: {vision_config['model']}")
    print(f"   DPI: {vision_config['dpi']}")
    print(f"   Format: {vision_config['image_format']}")
    
    try:
        r0_content, metadata = service.ingest_file(pdf_path, db_profile)
        
        print(f"\n✅ Ingestion completed successfully!")
        print(f"\n📊 Metadata:")
        print(f"   Method: {metadata.get('method')}")
        print(f"   File type: {metadata.get('file_type')}")
        print(f"   Page count: {metadata.get('page_count', 0)}")
        print(f"   DPI: {metadata.get('dpi')}")
        print(f"   Image format: {metadata.get('image_format')}")
        print(f"   Model: {metadata.get('model')}")
        print(f"   Input tokens: {metadata.get('input_tokens', 0):,}")
        print(f"   Output tokens: {metadata.get('output_tokens', 0):,}")
        
        if metadata.get('failed_pages'):
            print(f"   ⚠️  Failed pages: {metadata['failed_pages']}")
        
        # Verify per-page processing
        print(f"\n🔍 Verifying per-page processing...")
        page_markers = [f"## Page {i}" for i in range(1, metadata.get('page_count', 0) + 1)]
        found_pages = []
        for marker in page_markers:
            if marker in r0_content:
                found_pages.append(marker)
        
        print(f"   Expected page markers: {len(page_markers)}")
        print(f"   Found page markers: {len(found_pages)}")
        
        if len(found_pages) == len(page_markers):
            print(f"   ✅ All pages processed correctly!")
        else:
            print(f"   ⚠️  Some pages may be missing markers")
        
        # Show content preview
        print(f"\n📝 Content preview (first 500 chars):")
        print("-" * 60)
        preview = r0_content[:500]
        print(preview)
        if len(r0_content) > 500:
            print(f"... ({len(r0_content) - 500} more characters)")
        print("-" * 60)
        
        # Save full output to file
        output_file = project_root / f"vision_ingestion_test_output_{pdf_path.stem}.md"
        output_file.write_text(r0_content, encoding='utf-8')
        print(f"\n💾 Full output saved to: {output_file.name}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Ingestion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("Vision Ingestion Test - Per-Page Processing")
    print("=" * 60)
    
    success = test_vision_ingestion()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ Test completed successfully!")
    else:
        print("❌ Test failed!")
    print("=" * 60)
    
    sys.exit(0 if success else 1)

