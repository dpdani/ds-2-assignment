import datetime
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from mesa.batchrunner import BatchRunnerMP
from pandas import DataFrame
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
        # mesa's implementation is severely bad.
        # to the point of non-functioning.

        run_iter_args, total_iterations = runner._make_model_args_mp()
        # register the process pool and init a queue
        # store results in ordered dictionary
        run_iter_args = filter(
            lambda _: (
                    file_name_for_params(**_[1], iteration=_[3]) not in os.listdir(reports_dir)
            ),
            run_iter_args,
        )

        def wrapper(*args, **kwargs):
            params, model = runner._run_wrappermp(*args, **kwargs)
            # results[params] = model
            df: DataFrame = model.datacollector.get_model_vars_dataframe()
            proto, graph, nodes, view_size, view_to_send_size, delta_t, disaster_intensity, iteration = params
            df.to_csv(
                reports_dir / file_name_for_params(proto, graph, nodes, view_size, view_to_send_size, delta_t,
                                                   disaster_intensity, iteration)
            )
            return params

        if runner.processes > 1:
            with tqdm(total_iterations, disable=not runner.display_progress) as pbar:
                for params in runner.pool.imap_unordered(
                        wrapper, run_iter_args
                ):
                    secho(f"\nWriting report for experiment: {params}\n")
                    pbar.update()

                # runner._result_prep_mp(results)
        # For debugging model due to difficulty of getting errors during multiprocessing
        # else:
        #     for run in run_iter_args:
        #         params, model_data = runner._run_wrappermp(run)
        #         results[params] = model_data
        #
        #     runner._result_prep_mp(results)

        # Close multi-processing
        runner.pool.close()

    start = datetime.datetime.utcnow()
    # reports_dir = Path().cwd() / "runs" / f"{start.strftime('%Y%m%d%H%M%S')}"
    reports_dir = Path().cwd() / "runs"
    # reports_dir.mkdir(parents=True)
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
            "view_to_send_size": [19, 49, 99],
            "delta_t": [1, 4, 10],
            "disaster_intensity": [0.50, 0.65, 0.80, 0.90, 0.95],
        },
        # iterations=5,
        # variable_parameters={
        #     "proto": [Protocol.CYCLON, Protocol.NEWSCAST],
        #     "graph": [Graph.GEO, Graph.RANDOM, Graph.LATTICE],
        #     "nodes": [1_000, 10_000],
        #     "view_size": [20, 50],
        #     "view_to_send_size": [19, 49],
        #     "delta_t": [1, 4, 10],
        #     "disaster_intensity": [0.50, 0.90],
        # },
        fixed_parameters={

        },
        # fixed_parameters={
        #     "proto": Protocol.CYCLON,
        #     "graph": Graph.GEO,
        #     "nodes": 1_000,
        #     "view_size": 20,
        #     "view_to_send_size": 19,
        #     "delta_t": 5,
        #     "disaster_at": 500,
        #     "disaster_intensity": 0.50,
        #     "resurrection_at": 1000,
        # },
        max_steps=1_000_000,  # actually stops sooner
    )
    try:
        runner_run_all(runner)
    except KeyboardInterrupt:
        pass
    # secho("Finished running experiments, saving results...", fg="green")
    # for _ in runner.datacollector_model_reporters:
    #     secho(f"Writing report for experiment: {_}")
    #     df: DataFrame = runner.datacollector_model_reporters[_]
    #     proto, graph, nodes, view_size, view_to_send_size, delta_t, disaster_intensity, iteration = _
    #     df.to_csv(
    #         reports_dir / f"{proto.value} "
    #                       f"{graph.value} "
    #                       f"{nodes} "
    #                       f"{view_size} "
    #                       f"{view_to_send_size} "
    #                       f"{delta_t} "
    #                       f"{disaster_intensity} "
    #                       f"{iteration}"
    #                       f".csv"
    #     )
    # secho("Done.", fg="green")


if __name__ == '__main__':
    app()
