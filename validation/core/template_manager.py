"""Template management for validation reports."""
import os
from typing import Dict


class ValidationTemplateManager:
    """Manages template files for validation reports."""
    
    def __init__(self, template_dir: str):
        self.template_dir = template_dir
        # Validate that template directory exists
        if not os.path.exists(template_dir):
            raise FileNotFoundError(f"Template directory not found: {template_dir}")
    
    def load_template(self, template_name: str) -> str:
        """Load a template file."""
        template_path = os.path.join(self.template_dir, template_name)
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Template file not found: {template_path}")
        
        with open(template_path, "r") as f:
            return f.read()
    
    def build_step_template(self, step_name: str, include_validation: bool = False) -> str:
        """Build template for a validation step."""
        template_report = self.load_template("validation_title.txt")
        
        if include_validation:
            template_report += self.load_template("validity_check.txt")
        
        # Add step-specific template (skip for validate_smi since it already has validity_check)
        if step_name != 'validate_smi':
            from ..config.validation_config import ValidationStepConfig
            config = ValidationStepConfig.get_all_configs()[step_name]
            template_report += self.load_template(config.template_file)
        
        return template_report
    
    def add_deduplication_template(self, template_report: str) -> str:
        """Add deduplication section to template."""
        return template_report + self.load_template("deduplicate.txt")
    
    def finalize_template(self, template_report: str) -> str:
        """Add final section to template."""
        return template_report + self.load_template("end.txt")
    
    def build_complete_validation_template(self) -> str:
        """Build template for complete validation."""
        return self.load_template("validation_title.txt")
    
    def add_validation_section(self, template_report: str) -> str:
        """Add validation section."""
        return template_report + self.load_template("validity_check.txt")
    
    def add_mixture_section(self, template_report: str) -> str:
        """Add mixture removal section.""" 
        return template_report + self.load_template("mixture_removal.txt")
    
    def add_inorganic_section(self, template_report: str) -> str:
        """Add inorganic removal section."""
        return template_report + self.load_template("inorganic_removal.txt")
    
    def add_organometallic_section(self, template_report: str) -> str:
        """Add organometallic removal section."""
        return template_report + self.load_template("organometallic_removal.txt")