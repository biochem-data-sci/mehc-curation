"""Template management for refinement reports."""
import os
from typing import Dict


class RefinementTemplateManager:
    """Manages template files for refinement reports."""
    
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
        
        with open(template_path, "r", encoding="utf-8") as f:
            return f.read()
    
    def build_refinement_template(
        self,
        validate: bool = True,
        rm_mixture: bool = True,
        rm_inorganic: bool = True,
        rm_organometallic: bool = True,
        cl_salt: bool = True,
        neutralize: bool = True,
        destereoisomerize: bool = True,
        detautomerize: bool = True,
        rm_dup_after_1st: bool = False,
        rm_dup_after_2nd: bool = False,
        rm_dup_after_3rd: bool = False,
    ) -> str:
        """Build the complete refinement template based on enabled steps."""
        template_report = ""
        
        # Validation stage
        if validate or rm_mixture or rm_inorganic or rm_organometallic:
            template_report += self.load_template("validation_title.txt")
            
            if validate:
                template_report += self.load_template("validity_check.txt")
            
            if rm_mixture:
                template_report += self.load_template("mixture_removal.txt")
            
            if rm_inorganic:
                template_report += self.load_template("inorganic_removal.txt")
            
            if rm_organometallic:
                template_report += self.load_template("organometallic_removal.txt")
            
            if rm_dup_after_1st:
                template_report += self.load_template("deduplicate.txt")
        
        # Cleaning stage
        if cl_salt or neutralize:
            template_report += self.load_template("cleaning_title.txt")
            
            if cl_salt:
                template_report += self.load_template("salt_cleaning.txt")
            
            if neutralize:
                template_report += self.load_template("neutralization.txt")
            
            if rm_dup_after_2nd:
                template_report += self.load_template("deduplicate.txt")
        
        # Normalization stage
        if destereoisomerize or detautomerize:
            template_report += self.load_template("normalization_title.txt")
            
            if destereoisomerize:
                template_report += self.load_template("destereoisomerization.txt")
            
            if detautomerize:
                template_report += self.load_template("detautomerization.txt")
            
            if rm_dup_after_3rd:
                template_report += self.load_template("deduplicate.txt")
        
        # End section
        template_report += self.load_template("end.txt")
        
        return template_report
    
    def format_template(self, template: str, format_data: Dict) -> str:
        """Format template with provided data."""
        return template.format(**format_data)

