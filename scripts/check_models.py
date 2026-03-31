
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from src.ml.registry import ModelRegistry
from src.ml.training.model_configs import ModelType

def check_models():
    console = Console()
    registry = ModelRegistry()
    
    console.rule("[bold blue]Model Provenance Check[/bold blue]")
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("League")
    table.add_column("Type")
    table.add_column("Version")
    table.add_column("Date")
    table.add_column("Hash Abbrev")
    table.add_column("Status")
    
    leagues = ['PL', 'PD', 'SA', 'BL1', 'FL1']
    model_types = [ModelType.POISSON, ModelType.NB]
    
    for league in leagues:
        for m_type in model_types:
            try:
                # This loads the production model metadata
                model, meta = registry.get_production_model_for_league(league, m_type)
                
                status = "[green]OK[/green]" if meta else "[red]MISSING[/red]"
                version = meta.version if meta else "-"
                date = meta.training_date.split('T')[0] if meta else "-"
                h = meta.data_hash[:8] if meta and meta.data_hash else "-"
                
                table.add_row(league, m_type.value, version, date, h, status)
                
            except Exception as e:
                table.add_row(league, m_type.value, "ERROR", "-", "-", f"[red]{str(e)}[/red]")

    console.print(table)

if __name__ == "__main__":
    check_models()
