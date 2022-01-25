from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import BarChartModule, ChartModule, NetworkModule

from experiment import Model


def agent_portrayal(agent):
    portrayal = {
        "Shape": "circle",
        "Filled": "true",
        "r": 0.05,
        "Color": "grey",
        "Layer": 1,
    }

    # if agent.dead:
    #     portrayal["Color"] = "red"
    #     portrayal["Layer"] = 0
    return portrayal


def network_portrayal(G):
    portrayal = {
        "nodes": [
            {
                'size': .1,
                'color': "green",
                'tooltip': f"id: {i}",
                'id': i,
            }
            for i in G.nodes],
        "edges": [
            {
                'source': source,
                'target': target,
                'color': "grey",
                'width': .001,
                'id': f"({source},{target})",
            }
            for (source, target) in G.edges]
    }

    return portrayal


def start_server(model_params):
    grid = NetworkModule(network_portrayal)

    charts = []

    for _ in [
        # {"Label": "Clustering Coefficient", "Color": "black"},
        {"Label": "Alive Clustering Coefficient", "Color": "black"},
        # {"Label": "Average Path Length", "Color": "black"},
        {"Label": "Alive Average Path Length", "Color": "black"},
        # {"Label": "Unprocessed Messages", "Color": "black"},
        {"Label": "Alive Unprocessed Messages", "Color": "black"},
        # {"Label": "Average Message Latency", "Color": "black"},
        {"Label": "Alive Average Message Latency", "Color": "black"},
        # {"Label": "Partitions", "Color": "black"},
        {"Label": "Alive Partitions", "Color": "black"},
        {"Label": "Pollution", "Color": "black"},
        {"Label": "Alive Agents", "Color": "black"},
        {"Label": "Alive Degree", "Color": "black"},
    ]:
        charts.append(
            ChartModule([_], data_collector_name='datacollector')
        )

    for _ in [
        # {"Label": "Degree", "Color": "black"},
        # {"Label": "Alive Degree", "Color": "black"},
    ]:
        charts.append(
            BarChartModule([_], data_collector_name="datacollector")
        )

    server = ModularServer(Model,
                           # [grid, *charts],
                           charts,
                           "Model",
                           model_params)
    server.port = 8521
    server.launch()
    return server
