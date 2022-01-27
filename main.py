import os
import sys
from pathlib import Path
from typing import Optional

import typer
from mesa.batchrunner import BatchRunnerMP
from tqdm import tqdm
from typer import secho

from experiment import Model, Protocol
from graph import Graph
from viz import start_server


app = typer.Typer()


def file_name_for_params(proto, graph, nodes, view_size, view_to_send_size, delta_t, disaster_intensity, iteration):
    return (f"{proto.value} "
            f"{graph.value} "
            f"{nodes} "
            f"{view_size} "
            f"{view_to_send_size} "
            f"{delta_t} "
            f"{disaster_intensity} "
            f"{iteration}"
            f".csv")


@app.command()
def run_experiment(
        proto: Protocol,
        graph: Graph,
        nodes: int,
        view_size: int,
        view_to_send_size: int,
        delta_t: int,
        disaster_intensity: float,
):
    secho(f"Running experiment: "
          f"{proto=} "
          f"{graph=} "
          f"{nodes=} "
          f"{view_size=} "
          f"{view_to_send_size=} "
          f"{delta_t=} "
          f"{disaster_intensity=}", fg='green')
    start_server({
        "proto": proto,
        "graph": graph,
        "nodes": nodes,
        "view_size": view_size,
        "view_to_send_size": view_to_send_size,
        "delta_t": delta_t,
        "disaster_intensity": disaster_intensity,
    })


@app.command()
def run_all(cores: Optional[int] = None):
    def runner_run_all(runner: BatchRunnerMP):
        run_iter_args, total_iterations = runner._make_model_args_mp()
        # register the process pool and init a queue
        # store results in ordered dictionary
        run_iter_args = filter(
            lambda _: (
                    file_name_for_params(**_[1], iteration=_[3]) not in os.listdir(reports_dir)
            ),
            run_iter_args,
        )

        if runner.processes > 1:
            with tqdm(total_iterations, disable=not runner.display_progress) as pbar:
                for params, model in runner.pool.imap_unordered(
                        runner._run_wrappermp, run_iter_args
                ):
                    # results[params] = model
                    secho(f"\nWriting report for experiment: {params}\n")
                    pbar.update()

        # Close multi-processing
        runner.pool.close()

    reports_dir = Path().cwd() / "runs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    sys.setrecursionlimit(1_000_000)
    secho(f"{sys.getrecursionlimit()=}")
    runner = BatchRunnerMP(
        Model,
        nr_processes=cores,  # None = cores available
        variable_parameters={
            "proto": [Protocol.CYCLON, Protocol.NEWSCAST],
            "graph": [Graph.GEO, Graph.RANDOM, Graph.LATTICE, Graph.STAR],
            "nodes": [1_000],
            "view_size": [20, 50, 100],
            "view_to_send_size": [6, 10, 15],  # shuffle length
            "delta_t": [1, 4, 10],
            "disaster_intensity": [0.50, 0.75, 0.95],
        },
        fixed_parameters={

        },
        max_steps=1_000_000,  # actually stops sooner
    )
    try:
        runner_run_all(runner)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    app()
