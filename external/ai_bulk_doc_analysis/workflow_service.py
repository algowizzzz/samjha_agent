"""
Workflow Service - Manages workflows, versions, and domain associations.
"""
import logging
import uuid
from datetime import datetime
from typing import List, Optional, Dict
from sqlalchemy import and_, or_
from sqlalchemy.orm import Session as SQLASession

from .models import (
    Workflow, WorkflowVersion, WorkflowDomain,
    IngestionProfile, ExportProfile, ChainVersion
)
from .db_service import get_db_session

logger = logging.getLogger(__name__)


class WorkflowService:
    """Service for managing workflows and their versions."""

    def create_workflow(
        self,
        user_id: str,
        name: str,
        description: str,
        domains: List[str],
        ingestion_profile_id: str,
        chain_version_id: str,
        export_profile_id: str
    ) -> Workflow:
        """
        Create a new workflow with initial version.
        
        Args:
            user_id: User creating the workflow
            name: Workflow name (3-80 chars)
            description: Workflow description (20-240 chars)
            domains: List of domain names (at least 1)
            ingestion_profile_id: ID of ingestion profile
            chain_version_id: ID of chain version
            export_profile_id: ID of export profile
            
        Returns:
            Created Workflow object
            
        Raises:
            ValueError: If validation fails
        """
        # Validate description length (20-240 chars)
        if len(description) < 20 or len(description) > 240:
            raise ValueError("Description must be 20-240 characters")
        
        # Validate name length (3-80 chars)
        if len(name) < 3 or len(name) > 80:
            raise ValueError("Name must be 3-80 characters")
        
        # Validate domains
        if not domains or len(domains) == 0:
            raise ValueError("At least one domain is required")
        
        # Validate that domains are non-empty strings
        domains = [d.strip() for d in domains if d and d.strip()]
        if len(domains) == 0:
            raise ValueError("At least one non-empty domain is required")
        
        with get_db_session() as db:
            # Verify chain_version exists
            chain_version = db.query(ChainVersion).filter(
                ChainVersion.chain_version_id == chain_version_id
            ).first()
            if not chain_version:
                raise ValueError(f"Chain version {chain_version_id} not found")
            
            # Create workflow
            workflow_id = f"wf_{uuid.uuid4().hex[:12]}"
            workflow = Workflow(
                workflow_id=workflow_id,
                name=name,
                description=description,
                visibility_scope="domain",  # or "super" for super admin
                created_by=user_id
            )
            db.add(workflow)
            db.flush()
            
            # Create domain associations
            for domain in domains:
                wf_domain = WorkflowDomain(
                    workflow_id=workflow_id,
                    domain=domain
                )
                db.add(wf_domain)
            
            # Create initial version
            version_number = 1
            workflow_version_id = f"wfv_{workflow_id}-v{version_number}"
            workflow_version = WorkflowVersion(
                workflow_version_id=workflow_version_id,
                workflow_id=workflow_id,
                version_number=version_number,
                ingestion_profile_id=ingestion_profile_id,
                chain_version_id=chain_version_id,
                export_profile_id=export_profile_id
            )
            db.add(workflow_version)
            db.commit()
            
            logger.info(f"Created workflow {workflow_id} with version {workflow_version_id}")
            # Return a simple object with the data (avoiding session binding issues)
            class WorkflowResult:
                pass
            result = WorkflowResult()
            result.workflow_id = workflow_id
            result.name = name
            result.description = description
            result.workflow_version_id = workflow_version_id
            result.version_number = version_number
            result.domains = domains
            return result
    
    def update_workflow(
        self,
        workflow_id: str,
        name: str,
        description: str,
        domains: List[str],
        ingestion_profile_id: str,
        chain_version_id: str,
        export_profile_id: str
    ) -> Dict:
        """
        Update workflow - creates new version (immutable).
        
        Args:
            workflow_id: ID of workflow to update
            name: Updated name
            description: Updated description
            domains: Updated list of domains
            ingestion_profile_id: Updated ingestion profile ID
            chain_version_id: Updated chain version ID
            export_profile_id: Updated export profile ID
            
        Returns:
            New WorkflowVersion object
            
        Raises:
            ValueError: If workflow not found or validation fails
        """
        # Validate description length (20-240 chars)
        if len(description) < 20 or len(description) > 240:
            raise ValueError("Description must be 20-240 characters")
        
        # Validate name length (3-80 chars)
        if len(name) < 3 or len(name) > 80:
            raise ValueError("Name must be 3-80 characters")
        
        # Validate domains
        if not domains or len(domains) == 0:
            raise ValueError("At least one domain is required")
        
        domains = [d.strip() for d in domains if d and d.strip()]
        if len(domains) == 0:
            raise ValueError("At least one non-empty domain is required")
        
        with get_db_session() as db:
            workflow = db.query(Workflow).filter(Workflow.workflow_id == workflow_id).first()
            if not workflow:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            # Verify chain_version exists
            chain_version = db.query(ChainVersion).filter(
                ChainVersion.chain_version_id == chain_version_id
            ).first()
            if not chain_version:
                raise ValueError(f"Chain version {chain_version_id} not found")
            
            # Update workflow metadata
            workflow.name = name
            workflow.description = description
            workflow.updated_at = datetime.utcnow()
            
            # Update domains (delete old, add new)
            db.query(WorkflowDomain).filter(WorkflowDomain.workflow_id == workflow_id).delete()
            for domain in domains:
                wf_domain = WorkflowDomain(workflow_id=workflow_id, domain=domain)
                db.add(wf_domain)
            
            # Get next version number
            latest_version = (
                db.query(WorkflowVersion)
                .filter(WorkflowVersion.workflow_id == workflow_id)
                .order_by(WorkflowVersion.version_number.desc())
                .first()
            )
            version_number = (latest_version.version_number + 1) if latest_version else 1
            
            # Create new version
            workflow_version_id = f"wfv_{workflow_id}-v{version_number}"
            workflow_version = WorkflowVersion(
                workflow_version_id=workflow_version_id,
                workflow_id=workflow_id,
                version_number=version_number,
                ingestion_profile_id=ingestion_profile_id,
                chain_version_id=chain_version_id,
                export_profile_id=export_profile_id
            )
            db.add(workflow_version)
            db.commit()
            
            logger.info(f"Updated workflow {workflow_id}, created version {workflow_version_id}")
            
            # Return dictionary representation (avoid session binding issues)
            return {
                "workflow_version_id": workflow_version_id,
                "workflow_id": workflow_id,
                "version_number": version_number,
                "ingestion_profile_id": ingestion_profile_id,
                "chain_version_id": chain_version_id,
                "export_profile_id": export_profile_id,
            }
    
    def list_workflows(
        self,
        user_id: str,
        user_domains: List[str],
        is_super_admin: bool
    ) -> List[Dict]:
        """
        List workflows visible to user (domain-filtered).
        
        Args:
            user_id: User ID
            user_domains: List of domains user has access to
            is_super_admin: Whether user is super admin
            
        Returns:
            List of workflow dictionaries with metadata
        """
        with get_db_session() as db:
            if is_super_admin:
                # Super admin sees all
                workflows = db.query(Workflow).all()
            else:
                # Domain-scoped: workflows that have at least one domain in user's domains
                if not user_domains:
                    # User has no domains, return empty list
                    return []
                
                workflows = (
                    db.query(Workflow)
                    .join(WorkflowDomain)
                    .filter(WorkflowDomain.domain.in_(user_domains))
                    .distinct()
                    .all()
                )
            
            result = []
            for wf in workflows:
                # Get domains
                domains = [wd.domain for wd in wf.domains]
                
                # Get latest version
                latest_version = (
                    db.query(WorkflowVersion)
                    .filter(WorkflowVersion.workflow_id == wf.workflow_id)
                    .order_by(WorkflowVersion.version_number.desc())
                    .first()
                )
                
                # Get ingestion and export profiles for metadata
                ingestion_profile = None
                export_profile = None
                chain = None
                
                if latest_version:
                    ingestion_profile = db.query(IngestionProfile).filter(
                        IngestionProfile.ingestion_profile_id == latest_version.ingestion_profile_id
                    ).first()
                    
                    export_profile = db.query(ExportProfile).filter(
                        ExportProfile.export_profile_id == latest_version.export_profile_id
                    ).first()
                    
                    chain = db.query(ChainVersion).filter(
                        ChainVersion.chain_version_id == latest_version.chain_version_id
                    ).first()
                
                # Extract step titles from chain snapshot
                step_titles = []
                if chain and chain.snapshot and isinstance(chain.snapshot, dict):
                    steps = chain.snapshot.get("steps", [])
                    if isinstance(steps, list):
                        step_titles = [
                            step.get("title", f"Step {step.get('index', i+1)}") 
                            for i, step in enumerate(steps) 
                            if isinstance(step, dict)
                        ]
                
                result.append({
                    "workflow_id": wf.workflow_id,
                    "name": wf.name,
                    "description": wf.description,
                    "domains": domains,
                    "workflow_version_id": latest_version.workflow_version_id if latest_version else None,
                    "latest_version_id": latest_version.workflow_version_id if latest_version else None,
                    "chain_version_id": latest_version.chain_version_id if latest_version else None,
                    "ingestion_profile_id": latest_version.ingestion_profile_id if latest_version else None,
                    "export_profile_id": latest_version.export_profile_id if latest_version else None,
                    "created_at": wf.created_at.isoformat() if wf.created_at else None,
                    "updated_at": wf.updated_at.isoformat() if wf.updated_at else None,
                    "accepted_input_types": ingestion_profile.accepted_input_types if ingestion_profile else [],
                    "ingestion_mode": ingestion_profile.mode if ingestion_profile else None,
                    "export_format": export_profile.format if export_profile else None,
                    "export_type": export_profile.format if export_profile else None,  # Alias for compatibility
                    "step_count": chain.step_count if chain else 0,
                    "step_titles": step_titles,  # NEW: List of step titles
                })
            
            return result
    
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get workflow by ID."""
        with get_db_session() as db:
            return db.query(Workflow).filter(Workflow.workflow_id == workflow_id).first()
    
    def get_workflow_version(self, workflow_version_id: str) -> Optional[WorkflowVersion]:
        """Get workflow version by ID."""
        with get_db_session() as db:
            return db.query(WorkflowVersion).filter(
                WorkflowVersion.workflow_version_id == workflow_version_id
            ).first()
    
    def delete_workflow(self, workflow_id: str) -> bool:
        """
        Delete workflow (cascades to versions and domains).
        
        Args:
            workflow_id: ID of workflow to delete
            
        Returns:
            True if deleted, False if not found
        """
        with get_db_session() as db:
            workflow = db.query(Workflow).filter(Workflow.workflow_id == workflow_id).first()
            if not workflow:
                return False
            
            db.delete(workflow)
            db.commit()
            logger.info(f"Deleted workflow {workflow_id}")
            return True

