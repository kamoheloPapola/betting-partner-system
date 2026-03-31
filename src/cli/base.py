import typer

# Initialize the Typer app in a base module to avoid double-import/circular issues.
app = typer.Typer(
    name="betting-partner",
    help="ML-powered betting prediction system CLI.",
    pretty_exceptions_show_locals=False,
    add_completion=False,
)
