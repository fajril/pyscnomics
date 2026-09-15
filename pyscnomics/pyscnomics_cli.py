"""
CLI entry point for PySCnomics API server.
"""

import click
import uvicorn


@click.group()
def entry_point() -> None:
    """Manage PySCnomics services."""


@entry_point.command()
@click.option(
    "--host",
    default="127.0.0.1",
    show_default=True,
    help="The host address for the API server.",
)
@click.option(
    "--port",
    default=9999,
    show_default=True,
    type=int,
    help="The port number for the API server.",
)
def serve(host: str, port: int) -> None:
    """Run the PySCnomics API server."""
    uvicorn.run(
        "pyscnomics.api.main:app",
        host=host,
        port=port,
        reload=False,
    )


if __name__ == "__main__":
    entry_point()
