import logging
import json
from datetime import datetime
from typing import Any, Dict

class AuditLogger:
    def __init__(self, log_file: str = 'audit.log'):
        self.logger = logging.getLogger('AuditLogger')
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter('%(message)s'))
        self.logger.addHandler(handler)

    def log(self, action: str, user: str, details: Dict[str, Any]):
        entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'action': action,
            'user': user,
            'details': details
        }
        self.logger.info(json.dumps(entry, ensure_ascii=False)) 