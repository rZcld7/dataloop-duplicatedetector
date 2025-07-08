from .audit_logger import AuditLogger
from typing import Any, Dict

audit_logger = AuditLogger()

def track_critical_action(action: str, user: str, details: Dict[str, Any]):
    """Registra una acción crítica en el log de auditoría."""
    audit_logger.log(action, user, details) 