"""Template management for cleaning reports."""
import os
from typing import Dict


class CleaningTemplateManager:
    """Manages template files for cleaning reports."""
    
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
        """Build template for a cleaning step."""
        template_report = self.load_template("cleaning_title.txt")
        
        if include_validation:
            template_report += self.load_template("validity_check.txt")
        
        # Add step-specific template
        from ..config.cleaning_config import CleaningStepConfig
        config = CleaningStepConfig.get_all_configs()[step_name]
        template_report += self.load_template(config.template_file)
        
        return template_report
    
    def add_deduplication_template(self, template_report: str) -> str:
        """Add deduplication section to template."""
        return template_report + self.load_template("deduplicate.txt")
    
    def finalize_template(self, template_report: str) -> str:
        """Add final section to template."""
        return template_report + self.load_template("end.txt")
    
    def build_complete_cleaning_template(self) -> str:
        """Build template for complete cleaning."""
        return self.load_template("cleaning_title.txt")
    
    def add_validation_section(self, template_report: str) -> str:
        """Add validation section."""
        return template_report + self.load_template("validity_check.txt")
    
    def add_salt_cleaning_section(self, template_report: str) -> str:
        """Add salt cleaning section."""
        return template_report + self.load_template("salt_cleaning.txt")
    
    def add_neutralization_section(self, template_report: str) -> str:
        """Add neutralization section."""
        return template_report + self.load_template("neutralization.txt")

