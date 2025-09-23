# main.py
import typer
from unsupervised_cnn_cli.data_utils import load_and_prepare_data
from unsupervised_cnn_cli.model import run_rbm_model

app = typer.Typer()

@app.command()
def run(
    input_file: str = typer.Option(..., help="Path to tidy CSV"),
    output_prefix: str = typer.Option(..., help="Prefix for output files")
):
    """
    Run unsupervised CNN on tidy input.
    """
    df_matrix = load_and_prepare_data(input_file)
    run_rbm_model(df_matrix, output_prefix)

if __name__ == "__main__":
    app()
