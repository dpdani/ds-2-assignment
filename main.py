import datetime
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
        results = {}

        if runner.processes > 1:
            with tqdm(total_iterations, disable=not runner.display_progress) as pbar:
                for params, model in runner.pool.imap_unordered(
                        runner._run_wrappermp, run_iter_args
                ):
                    # results[params] = model
                    secho(f"\nWriting report for experiment: {params}\n")
                    df: DataFrame = model.datacollector.get_model_vars_dataframe()
                    proto, graph, nodes, view_size, view_to_send_size, delta_t, disaster_intensity, iteration = params
                    df.to_csv(
                        reports_dir / f"{proto.value} "
                                      f"{graph.value} "
                                      f"{nodes} "
                                      f"{view_size} "
                                      f"{view_to_send_size} "
                                      f"{delta_t} "
                                      f"{disaster_intensity} "
                                      f"{iteration}"
                                      f".csv"
                    )
                    pbar.update()

                runner._result_prep_mp(results)
        # For debugging model due to difficulty of getting errors during multiprocessing
        else:
            for run in run_iter_args:
                params, model_data = runner._run_wrappermp(run)
                results[params] = model_data

            runner._result_prep_mp(results)

        # Close multi-processing
        runner.pool.close()

    start = datetime.datetime.utcnow()
    reports_dir = Path().cwd() / "runs" / f"{start.strftime('%Y%m%d%H%M%S')}"
    reports_dir.mkdir(parents=True)
    sys.setrecursionlimit(1_000_000)
    secho(f"{sys.getrecursionlimit()=}")
    runner = BatchRunnerMP(
        Model,
        nr_processes=cores,  # None = cores available
        variable_parameters={
            "proto": [Protocol.CYCLON, Protocol.NEWSCAST],
            "graph": [Graph.GEO, Graph.RANDOM, Graph.LATTICE, Graph.STAR],
            "nodes": [1_000, 10_000],
            "view_size": [20, 50, 100],
            "view_to_send_size": [19, 49, 99],
            "delta_t": [1, 2, 3, 4, 5, 10],
            "disaster_intensity": [0.50, 0.65, 0.80, 0.90, 0.95],
        },
        iterations=5,
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
        # max_steps=1500,
    )
    try:
        runner_run_all(runner)
    except KeyboardInterrupt:
        pass
    secho("Finished running experiments, saving results...", fg="green")
    for _ in runner.datacollector_model_reporters:
        secho(f"Writing report for experiment: {_}")
        df: DataFrame = runner.datacollector_model_reporters[_]
        proto, graph, nodes, view_size, view_to_send_size, delta_t, disaster_intensity, iteration = _
        df.to_csv(
            reports_dir / f"{proto.value} "
                          f"{graph.value} "
                          f"{nodes} "
                          f"{view_size} "
                          f"{view_to_send_size} "
                          f"{delta_t} "
                          f"{disaster_intensity} "
                          f"{iteration}"
                          f".csv"
        )
    secho("Done.", fg="green")


if __name__ == '__main__':
    app()
