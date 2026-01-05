"""
Domain configuration parsing tests.
"""

import unittest
from external.agent.sql_gate import get_domain_bool, metrics_empty_allowed
from external.tools.parquet_agent.search_glossary import SearchGlossaryTool


class TestDomainConfig(unittest.TestCase):
    """Test domain configuration parsing."""
    
    def test_get_domain_bool_true(self):
        """Test extracting true boolean from domain.md."""
        domain_md = """
## 3) Listing rules
- listing_allows_empty_metrics: true
- listing_default_limit: 50
"""
        result = get_domain_bool(domain_md, "listing_allows_empty_metrics")
        self.assertTrue(result)
    
    def test_get_domain_bool_false(self):
        """Test extracting false boolean from domain.md."""
        domain_md = """
## 3) Listing rules
- listing_allows_empty_metrics: false
"""
        result = get_domain_bool(domain_md, "listing_allows_empty_metrics")
        self.assertFalse(result)
    
    def test_get_domain_bool_missing(self):
        """Test extracting missing boolean returns None."""
        domain_md = """
## 3) Listing rules
- listing_default_limit: 50
"""
        result = get_domain_bool(domain_md, "listing_allows_empty_metrics")
        self.assertIsNone(result)
    
    def test_metrics_empty_allowed(self):
        """Test metrics_empty_allowed logic."""
        domain_md = """
## 3) Listing rules
- listing_allows_empty_metrics: true
"""
        query_spec = {
            "metrics": [],
            "aggregation_plan": "listing"
        }
        result = metrics_empty_allowed(query_spec, domain_md)
        self.assertTrue(result)
    
    def test_metrics_empty_not_allowed(self):
        """Test metrics_empty_allowed when not configured."""
        domain_md = """
## 3) Listing rules
- listing_default_limit: 50
"""
        query_spec = {
            "metrics": [],
            "aggregation_plan": "listing"
        }
        result = metrics_empty_allowed(query_spec, domain_md)
        self.assertFalse(result)
    
    def test_search_glossary(self):
        """Test search_glossary tool parses domain.md."""
        tool = SearchGlossaryTool({"domain_directory": "external/config/domains"})
        result = tool.execute({"term": "revenue", "domain": "ecomm"})
        self.assertIn("hits", result)
        self.assertIsInstance(result["hits"], list)


if __name__ == "__main__":
    unittest.main()

