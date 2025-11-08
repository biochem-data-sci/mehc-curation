"""Configuration for refinement operations."""
import os


class RefinementConfig:
    """Configuration for refinement pipeline."""
    
    def __init__(self):
        # Get template directory relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(current_dir))
        self.template_dir = os.path.join(project_root, "template_report")
        
        # Report subdirectory name
        self.report_subdir = "refinement"
        
        # Output file names
        self.output_csv = "post_refined_smiles.csv"
        self.report_file = "refinement_report.txt"
        self.dup_1st_csv = "remove_dupls_1st_data.csv"
        self.dup_2nd_csv = "remove_dupls_2nd_data.csv"
        self.dup_3rd_csv = "remove_dupls_3rd_data.csv"

