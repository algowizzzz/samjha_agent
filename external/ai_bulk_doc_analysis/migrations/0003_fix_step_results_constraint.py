"""
Migration: Fix step_results unique constraint to include task_id for CSV workflows

Revision ID: 0003_fix_step_results_constraint
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql, sqlite

# Revision identifiers
revision = '0003_fix_step_results_constraint'
down_revision = '0002_workflow_system'

def upgrade():
    """Update unique constraint to include task_id."""
    # Get database connection to check type
    bind = op.get_bind()
    is_postgresql = bind.dialect.name == 'postgresql'
    is_sqlite = bind.dialect.name == 'sqlite'
    
    # Drop old constraint
    constraint_names = ['uq_step_result', 'step_results_run_id_doc_id_step_index_key']
    for constraint_name in constraint_names:
        try:
            op.drop_constraint(constraint_name, 'step_results', type_='unique')
        except Exception:
            pass
    
    if is_postgresql:
        # PostgreSQL: Use a unique index with COALESCE to handle NULL task_id
        # This allows multiple NULLs (non-CSV) but enforces uniqueness when task_id is set (CSV)
        op.execute("""
            CREATE UNIQUE INDEX uq_step_result_with_task 
            ON step_results (run_id, doc_id, step_index, COALESCE(task_id, ''))
        """)
    elif is_sqlite:
        # SQLite: NULLs are considered distinct in unique constraints
        # So we can create a constraint that includes task_id
        # For non-CSV (task_id is NULL), each (run_id, doc_id, step_index) must be unique
        # For CSV (task_id is NOT NULL), (run_id, doc_id, step_index, task_id) must be unique
        # SQLite allows this because NULL != NULL in unique constraints
        op.create_unique_constraint(
            'uq_step_result_with_task',
            'step_results',
            ['run_id', 'doc_id', 'step_index', 'task_id']
        )
    else:
        # Fallback for other databases
        op.create_unique_constraint(
            'uq_step_result_with_task',
            'step_results',
            ['run_id', 'doc_id', 'step_index', 'task_id']
        )

def downgrade():
    """Revert to old constraint."""
    try:
        op.drop_constraint('uq_step_result_with_task', 'step_results', type_='unique')
    except Exception:
        pass
    
    # Restore old constraint
    op.create_unique_constraint(
        'uq_step_result',
        'step_results',
        ['run_id', 'doc_id', 'step_index']
    )

