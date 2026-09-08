"""
CLI entry point for PySCnomics API server.
"""

import click
import uvicorn


@click.command()
@click.option(
    "-api",
    "--api",
    default=1,
    help="The command for running the API backend. 0 for not activating the API backend. 1 for activating the API backend",
)
@click.option(
    "-port",
    "--port",
    default=9999,
    help="The port number for running the API backend. The default port is 9999",
)
@click.option(
    "--host",
    default="127.0.0.1",
    help="The host address for running the API backend. Defaults to 127.0.0.1",
)
def entry_point(api: int, port: int, host: str) -> None:
    """Manages CLI"""
    if api == 1:
        body = """
                We welcome you to our library, PySCnomics. This package contains tailored functionalities for
                assessing economic feasibility of oil and gas projects following the state-of-the-art Production
                Sharing Contract (PSC) schemes in Indonesia.
                PySCnomics is the product of join research between Indonesia's Special Task Force for Upstream Oil
                and Gas Business Activities (SKK Migas) and the Department of Petroleum Engineering,
                Institut Teknologi Bandung (ITB)
                """
        print(body)
        uvicorn.run(
            "pyscnomics.api.main:app",
            host=host,
            port=port,
            reload=False,
        )


if __name__ == "__main__":
    entry_point()
