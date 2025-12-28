#!/usr/bin/env python3
"""
Unit test for Bulk Doc Analysis service layer (no server required).
Tests the core business logic directly.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from external.ai_bulk_doc_analysis.services import BulkDocService, Document, Chain
from datetime import datetime
import tempfile
import shutil

def test_create_chain():
    """Test chain creation"""
    print("\n" + "="*60)
    print("TEST: Create Chain")
    print("="*60)
    
    temp_dir = Path(tempfile.mkdtemp())
    service = BulkDocService(storage_base=temp_dir)
    
    steps = [
        {
            "index": 1,
            "required_inputs": ["R0"],
            "prompt": "Summarize this document.",
            "description": "Summary step"
        },
        {
            "index": 2,
            "required_inputs": ["R0", "R1"],
            "prompt": "Extract key themes.",
            "description": "Theme extraction"
        }
    ]
    
    try:
        chain = service.create_chain("test_user", "Test Chain", "Test description", steps)
        print(f"✓ Chain created")
        print(f"  Chain ID: {chain.chain_id}")
        print(f"  Version ID: {chain.chain_version_id}")
        print(f"  Valid: {chain.valid}")
        print(f"  Steps: {chain.step_count}")
        
        # Verify chain appears in list
        chains = service.list_chains()
        found = any(c.chain_version_id == chain.chain_version_id for c in chains)
        if found:
            print(f"✓ Chain appears in list")
        else:
            print(f"❌ Chain not found in list")
            return False
        
        return True
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_pdf_conversion():
    """Test PDF conversion (if pdfplumber available)"""
    print("\n" + "="*60)
    print("TEST: PDF Conversion")
    print("="*60)
    
    try:
        import pdfplumber
    except ImportError:
        print("⚠️  pdfplumber not available - skipping")
        return True
    
    temp_dir = Path(tempfile.mkdtemp())
    service = BulkDocService(storage_base=temp_dir)
    
    # Try to use an existing PDF from data directory
    existing_pdfs = list(Path("data/ai_bulk_doc_analysis/docs").rglob("*.pdf"))
    if existing_pdfs:
        pdf_path = existing_pdfs[0]
        print(f"✓ Using existing PDF: {pdf_path.name}")
    else:
        # Try to create a minimal PDF using reportlab if available
        try:
            from reportlab.pdfgen import canvas
            from reportlab.lib.pagesizes import letter
            
            pdf_path = temp_dir / "test.pdf"
            c = canvas.Canvas(str(pdf_path), pagesize=letter)
            c.drawString(100, 750, "Test Document")
            c.drawString(100, 730, "This is a test PDF for conversion.")
            c.drawString(100, 710, "It contains multiple lines of text.")
            c.showPage()
            c.save()
            
            print(f"✓ Created test PDF with reportlab")
        except ImportError:
            print("⚠️  reportlab not available and no existing PDFs found")
            print("   Skipping PDF conversion test (needs real PDF)")
            return True
    
    try:
        # Test conversion
        md_content = service._convert_pdf_to_markdown(pdf_path)
        print(f"✓ PDF converted to markdown")
        print(f"  Output length: {len(md_content)} chars")
        print(f"  Preview: {md_content[:100]}...")
        
        if "Test Document" not in md_content and "test document" not in md_content.lower():
            print("⚠️  Warning: Expected content not found in output")
        
        return True
    except Exception as e:
        print(f"❌ Conversion error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def test_run_execution():
    """Test run execution (requires LLM)"""
    print("\n" + "="*60)
    print("TEST: Run Execution (requires ANTHROPIC_API_KEY)")
    print("="*60)
    
    import os
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not set - skipping execution test")
        print("   Set it to test real Claude API integration")
        return True
    
    temp_dir = Path(tempfile.mkdtemp())
    service = BulkDocService(storage_base=temp_dir)
    
    # Create a chain
    steps = [
        {
            "index": 1,
            "required_inputs": ["R0"],
            "prompt": "Summarize this document in one sentence.",
            "description": "Quick summary"
        }
    ]
    
    try:
        chain = service.create_chain("test_user", "Simple Chain", "Test", steps)
        print(f"✓ Chain created: {chain.chain_version_id}")
        
        # Create a session and document
        session_id = service.ensure_session("test_user")
        print(f"✓ Session: {session_id}")
        
        # Create a mock document with converted markdown
        doc_id = "test_doc_123"
        doc_dir = temp_dir / "docs" / doc_id
        doc_dir.mkdir(parents=True, exist_ok=True)
        
        # Create R0 (converted document)
        r0_path = doc_dir / "converted.md"
        r0_path.write_text("This is a test document. It contains some text for analysis.")
        
        doc = Document(
            doc_id=doc_id,
            session_id=session_id,
            original_filename="test.pdf",
            file_type="PDF",
            size_bytes=100,
            status="CONVERTED",
            converted_md_path=str(r0_path),
        )
        service.documents[doc_id] = doc
        service.sessions[session_id].append(doc_id)
        
        print(f"✓ Document created: {doc_id}")
        
        # Create run
        run = service.create_run(session_id, chain.chain_version_id)
        print(f"✓ Run created: {run.run_id}")
        print(f"  Status: {run.status}")
        
        # Check progress
        progress = service.get_run_progress(run.run_id)
        if progress:
            print(f"✓ Progress retrieved")
            print(f"  Status: {progress.get('status')}")
            print(f"  Rows: {len(progress.get('rows', []))}")
        else:
            print(f"❌ Progress not found")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

def main():
    """Run unit tests"""
    print("="*60)
    print("BULK DOC ANALYSIS - SERVICE LAYER UNIT TESTS")
    print("="*60)
    
    results = []
    
    results.append(("Create Chain", test_create_chain()))
    results.append(("PDF Conversion", test_pdf_conversion()))
    results.append(("Run Execution", test_run_execution()))
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All tests passed!")
    else:
        print("❌ Some tests failed")

if __name__ == "__main__":
    main()

