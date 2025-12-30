"""
Domain Service - Domain-based access control for workflows.
"""
import logging
from typing import Dict, List, Optional

from .models import Workflow, WorkflowDomain
from .db_service import get_db_session

logger = logging.getLogger(__name__)


class DomainService:
    """Service for domain-based access control."""
    
    @staticmethod
    def can_view_workflow(user_session: Dict, workflow: Workflow) -> bool:
        """
        Check if user can view a workflow.
        
        Super admin: can view all workflows
        Domain user: can view workflows that have at least one domain in common
        
        Args:
            user_session: User session data
            workflow: Workflow object
            
        Returns:
            True if user can view the workflow
        """
        from core.auth_manager import AuthManager
        
        # Super admin can view all
        auth_manager = AuthManager()
        if auth_manager.is_super_admin(user_session):
            return True
        
        # Get user domains
        user_domains = user_session.get("domains", [])
        if not user_domains:
            return False
        
        # Get workflow domains
        with get_db_session() as db:
            workflow_domains = [
                wd.domain for wd in db.query(WorkflowDomain).filter(
                    WorkflowDomain.workflow_id == workflow.workflow_id
                ).all()
            ]
        
        # Check if there's any overlap
        return bool(set(user_domains) & set(workflow_domains))
    
    @staticmethod
    def can_edit_workflow(user_session: Dict, workflow: Workflow) -> bool:
        """
        Check if user can edit a workflow.
        
        Super admin: can edit all workflows
        Domain admin: can edit workflows for their domains
        
        Args:
            user_session: User session data
            workflow: Workflow object
            
        Returns:
            True if user can edit the workflow
        """
        from core.auth_manager import AuthManager
        
        auth_manager = AuthManager()
        
        # Super admin can edit all
        if auth_manager.is_super_admin(user_session):
            return True
        
        # Domain admin can edit workflows in their domains
        if auth_manager.is_domain_admin(user_session):
            return DomainService.can_view_workflow(user_session, workflow)
        
        return False
    
    @staticmethod
    def can_run_workflow(user_session: Dict, workflow: Workflow) -> bool:
        """
        Check if user can run a workflow.
        
        Same as can_view_workflow - if you can view it, you can run it.
        
        Args:
            user_session: User session data
            workflow: Workflow object
            
        Returns:
            True if user can run the workflow
        """
        return DomainService.can_view_workflow(user_session, workflow)
    
    @staticmethod
    def get_workflow_domains(workflow_id: str) -> List[str]:
        """
        Get list of domains for a workflow.
        
        Args:
            workflow_id: Workflow ID
            
        Returns:
            List of domain names
        """
        with get_db_session() as db:
            domains = [
                wd.domain for wd in db.query(WorkflowDomain).filter(
                    WorkflowDomain.workflow_id == workflow_id
                ).all()
            ]
        return domains

