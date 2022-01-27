from __future__ import annotations

import heapq
import math
import random
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable

import igraph
import mesa
import numpy as np
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation

from graph import Graph


class Model(mesa.Model):
    def __init__(
            self,
            proto: 'Protocol',
            graph: Graph,
            nodes: int,
            view_size: int,
            view_to_send_size: int,
            delta_t: int,
            disaster_intensity: float,
    ):
        super().__init__()
        self.resurrection_at = 200 * delta_t
        self.end_at = 300 * delta_t
        self.disaster_intensity = disaster_intensity
        self.disaster_at = 100 * delta_t
        self.delta_T = delta_t
        self.graph_type = graph
        self.graph = graph.i_make(nodes)
        self.schedule = RandomActivation(self)
        if view_to_send_size > view_size:
            view_to_send_size = view_size - 1
            self.running = False
        if proto == Protocol.NEWSCAST and view_to_send_size != 10:
            self.running = False
        self.agents = []  # accessing self.schedule.agents takes up ~45% of running time, when there's >1000 agents
        for i in self.graph.vs:
            agent = proto.klass(self, i.index, view_size, view_to_send_size)
            self.schedule.add(agent)
            self.agents.append(agent)
        self.datacollector = DataCollector(
            model_reporters={
                "Clustering Coefficient": self.clustering_coefficient,
                "Alive Clustering Coefficient": self.alive_clustering_coefficient,
                "Average Path Length": self.average_path_length,
                "Alive Average Path Length": self.alive_average_path_length,
                "Degree": self.degree,
                "Alive Degree": self.alive_degree,
                "Unprocessed Messages": self.unprocessed_messages,
                "Alive Unprocessed Messages": self.alive_unprocessed_messages,
                "Average Message Latency": self.average_message_latency,
                "Alive Average Message Latency": self.alive_average_message_latency,
                "Partitions": self.partitions,
                "Alive Partitions": self.alive_partitions,
                "Pollution": self.pollution,
                "Alive Agents": self.alive_agents_num,
            },
            agent_reporters={

            },
        )
        self._overlay_network = igraph.Graph()
        self._overlay_network_time = -1
        self._alive_overlay_network = igraph.Graph()
        self._alive_overlay_network_time = -1
        self._alive_agents: list[Agent] = []
        self._alive_agents_time = -1
        self.latencies = np.array(self.graph.shortest_paths(  # uses Dijkstra
            weights="latency"
        ), dtype=int)
        self.proto = proto
        self.nodes = nodes
        self.view_size = view_size
        self.view_to_send_size = view_to_send_size

    def step(self) -> None:
        if self.schedule.time <= self.end_at:
            self.datacollector.collect(self)
            if self.disaster_at > 0:
                if self.schedule.time == self.disaster_at:
                    self.disaster()
            if self.resurrection_at > 0:
                if self.schedule.time == self.resurrection_at:
                    self.resurrection()
            self.schedule.step()
        else:
            self.running = False
            df = self.datacollector.get_model_vars_dataframe()
            reports_dir = Path().cwd() / "runs"
            df.to_csv(
                reports_dir / f"{self.proto.value} "
                              f"{self.graph_type.value} "
                              f"{self.nodes} "
                              f"{self.view_size} "
                              f"{self.view_to_send_size} "
                              f"{self.delta_T} "
                              f"{self.disaster_intensity} "
                              f"{0}"
                              f".csv"
            )
            # Avoid pickling large objects when using multiprocessing
            del self.graph, self.datacollector, self._overlay_network, \
                self._alive_overlay_network, self.latencies, \
                self.schedule

    def disaster(self):
        for _ in self.agents:
            p = self.disaster_intensity
            match self.graph_type:
                case Graph.GEO:
                    if 0 <= _.unique_id <= 41:
                        p = 0
                case Graph.STAR:
                    if _.unique_id == 0:
                        p = 0
            if random.random() < p:
                _.die()

    def resurrection(self):
        for _ in self.agents:
            _.resurrect()

    # Statistics
    @property
    def overlay_network(self) -> igraph.Graph:
        def edges():
            for agent in self.agents:
                for _ in agent.view:
                    yield agent.unique_id, _.address.unique_id

        if self._overlay_network_time == self.schedule.time:
            return self._overlay_network
        g = igraph.Graph(len(self.agents))
        g.add_edges(edges())
        self._overlay_network = g
        self._overlay_network_time = self.schedule.time
        return g

    @property
    def alive_agents(self) -> list[Agent]:
        if self._alive_agents_time == self.schedule.time:
            return self._alive_agents
        self._alive_agents = list(filter(
            lambda _: not _.dead,
            self.agents,
        ))
        self._alive_agents_time = self.schedule.time
        return self._alive_agents

    @property
    def alive_overlay_network(self) -> igraph.Graph:
        if self._alive_overlay_network_time == self.schedule.time:
            return self._alive_overlay_network
        g: igraph.Graph = self.overlay_network.copy()
        g.delete_vertices(map(
            lambda _: _.unique_id,
            filter(
                lambda _: _.dead,
                self.agents,
            )
        ))
        self._alive_overlay_network = g
        self._alive_overlay_network_time = self.schedule.time
        return g

    def clustering_coefficient(self) -> float:
        return self.overlay_network.transitivity_undirected(
            mode="zero"  # no triangles => return 0
        )

    def alive_clustering_coefficient(self) -> float:
        return self.alive_overlay_network.transitivity_undirected(
            mode="zero"
        )

    def average_path_length(self):
        length: float = self.overlay_network.average_path_length()
        return 0 if math.isnan(length) else length

    def alive_average_path_length(self):
        length: float = self.alive_overlay_network.average_path_length()
        return 0 if math.isnan(length) else length

    def degree(self):
        from igraph import mean
        return mean(
            self.overlay_network.degree()
        )

    def alive_degree(self):
        from igraph import mean
        return mean(
            self.alive_overlay_network.degree()
        )

    def unprocessed_messages(self):
        return sum(
            map(
                lambda _: len(_.incoming_messages),
                self.agents,
            )
        )

    def alive_unprocessed_messages(self):
        return sum(
            map(
                lambda _: len(_.incoming_messages),
                self.alive_agents,
            )
        )

    def average_message_latency(self):
        return sum(
            map(
                lambda _:
                sum([tm[0] - self.schedule.time for tm in _.incoming_messages]) / len(_.incoming_messages)
                if _.incoming_messages else 0,
                self.agents,
            )
        ) / len(self.agents)

    def alive_average_message_latency(self):
        return sum(
            map(
                lambda _:
                sum([tm[0] - self.schedule.time for tm in _.incoming_messages]) / len(_.incoming_messages)
                if _.incoming_messages else 0,
                self.alive_agents,
            )
        ) / len(self.alive_agents)

    def partitions(self):
        if self.schedule.time < 3:
            return 0
        return len(self.overlay_network.clusters())

    def alive_partitions(self):
        if self.schedule.time < 3:
            return 0
        return len(self.alive_overlay_network.clusters())

    def pollution(self):
        if self.schedule.time == 0:
            return 0
        return sum(
            map(
                lambda agent: len(list(
                    filter(
                        lambda _: _.address.dead,
                        agent.view,
                    )
                )),
                self.alive_agents,
            )
        ) / sum(
            map(
                lambda agent: len(agent.view),
                self.alive_agents
            )
        )

    def alive_agents_num(self):
        return len(self.alive_agents)


@dataclass(frozen=True, order=True)
class Descriptor:
    address: 'Agent'
    hop_count: int
    time: int


@dataclass(frozen=True, order=True)
class View:
    _: list[Descriptor]

    def increased_hop_count(self, amount: int):
        return View([
            Descriptor(_.address, _.hop_count + amount, _.time)
            for _ in self._
        ])

    def __len__(self):
        return len(self._)

    def __iter__(self):
        return self._.__iter__()

    def __getitem__(self, item: 'Agent' | int):
        match item:
            case int():
                return self._[item]
            case Agent():
                for descriptor in self._:
                    if descriptor.address == item:
                        return descriptor
                raise IndexError('agent not found.')

    def __sub__(self, other: 'View'):
        return View(list(filter(lambda _: _ is not None, [
            _ if _ not in other else None
            for _ in self._
        ])))

    def __add__(self, other: 'View'):
        v = self.descriptors.copy()
        v.extend(other.descriptors)
        return View(v)

    @property
    def descriptors(self):
        return self._


@dataclass(frozen=True, order=True)
class Message:
    from_: 'Agent'
    view: View
    follow_up: bool


class Agent(mesa.Agent):
    def __init__(self, model: Model, i, view_size: int, view_to_send_size: int):
        self.dead = False
        self.incoming_messages: list[tuple[int, Message]] = []
        self.model = model
        self.view = View([])
        self.view_size = view_size
        self.view_to_send_size = view_to_send_size
        self.bootstrapped = False
        super().__init__(i, model)

    def step(self) -> None:
        if not self.dead:
            if not self.bootstrapped:
                self.bootstrap()
            self.read_messages()
            if (self.model.schedule.time + self.unique_id) % self.model.delta_T == 0:
                self.send_message()

    def read_messages(self):
        # element 0 has the lowest time in the queue; element 0[0] is the time value
        while self.incoming_messages and self.incoming_messages[0][0] <= self.model.schedule.time:
            time, message = heapq.heappop(self.incoming_messages)
            view = message.view
            # if pull:
            if message.follow_up:
                # in theory a ``send'' would cause the receiving agent to follow
                # up to the message by sending its view, but that message would
                # only be sent to the socket that initially sent a message.
                # here, no sockets, so this fields kinda mimics that behavior.
                message.from_.push_message(
                    Message(
                        from_=self,
                        view=self.prepare_reply(message.from_),
                        follow_up=False,
                    ),
                    self.model.schedule.time + self.latency_to(message.from_)
                )
            self.view = self.select_view(
                self.merge_views(self.view, view)
            )

    def send_message(self):
        peer = self.select_peer()
        # if push:
        peer.push_message(
            Message(
                from_=self,
                view=self.prepare_request(peer),
                follow_up=True
            ),
            self.model.schedule.time + self.latency_to(peer)
        )
        # else:
        #     peer.push_message(empty_message)
        # if pull:
        #     subsequent pull is handled in self.read_messages

    def push_message(self, message: Message, time: int):
        # assert time > self.model.schedule.time
        heapq.heappush(
            self.incoming_messages,
            (time, message)
        )

    def latency_to(self, to: 'Agent') -> int:
        return self.model.latencies[self.unique_id][to.unique_id]

    def bootstrap(self):
        self.view = View([])
        match self.model.graph_type:
            case Graph.GEO:
                self.geo_bootstrap()
            case Graph.RANDOM:
                self.random_bootstrap()
            case Graph.LATTICE:
                self.lattice_bootstrap()
            case Graph.STAR:
                self.star_bootstrap()
        self.bootstrapped = True

    def _to_descriptors(self, agent_ids: Iterable[int]):
        return list(map(
            lambda _: Descriptor(_, 1, self.model.schedule.time),
            map(
                lambda _: self.model.agents[_],
                agent_ids,
            ),
        ))

    def _local_broadcast(self, lowest_id: int, size: int):
        local = list(
            filter(
                lambda _: not self.model.agents[_].dead,
                range(lowest_id, lowest_id + size),
            )
        )
        random.shuffle(local)
        return local[:5]

    def _graph_local_broadcast(self, node: int):
        local = list(filter(
            lambda _: not self.model.agents[_].dead,
            map(
                lambda _: _.target if _.source == self.unique_id else _.source,
                self.model.graph.vs[self.unique_id].all_edges(),
            ),
        ))
        random.shuffle(local)
        return local[:5]

    def geo_bootstrap(self):
        if self.unique_id <= 5:
            # An agent in the core hex knows (statically) about:
            #   - every other agent in the core; and
            #   - every agent in the geo hex for which this agent is the gateway
            lowest_geo_id = 6 + 6 * self.unique_id
            self.view = (
                    View(self._to_descriptors(range(6))) +
                    View(self._to_descriptors(range(lowest_geo_id, lowest_geo_id + 6)))
            )
        elif 6 <= self.unique_id <= 41:
            # assert (len(self.model.graph.vs) - 42) / 36 == (len(self.model.graph.vs) - 42) // 36
            size_of_local = (len(self.model.graph.vs) - 42) // 36
            lowest_local_id = 42 + (self.unique_id - 6) * size_of_local
            lowest_geo_id = 6 + 6 * ((self.unique_id - 6) // 6)
            core_gateway = list(filter(
                lambda _: 0 <= _ <= 5,
                map(
                    lambda _: _.target if _.source == self.unique_id else _.source,
                    self.model.graph.vs[self.unique_id].all_edges(),
                ),
            ))
            # assert len(core_gateway) == 1
            core_gateway = core_gateway[0]
            self.view = (
                    View(self._to_descriptors([core_gateway])) +
                    View(self._to_descriptors(range(lowest_geo_id, lowest_geo_id + 6))) +
                    View(self._to_descriptors(self._local_broadcast(lowest_local_id, size_of_local)))
            )
        else:
            # assert (len(self.model.graph.vs) - 42) / 36 == (len(self.model.graph.vs) - 42) // 36
            size_of_local = (len(self.model.graph.vs) - 42) // 36
            lowest_local_id = 42 + size_of_local * ((self.unique_id - 42) // size_of_local)
            geo_gateway = list(filter(
                lambda _: 6 <= _ <= 41,
                map(
                    lambda _: _.target if _.source == self.unique_id else _.source,
                    self.model.graph.vs[self.unique_id].all_edges(),
                ),
            ))
            # assert len(geo_gateway) == 1
            geo_gateway = geo_gateway[0]
            self.view = (
                    View(self._to_descriptors([geo_gateway])) +
                    View(self._to_descriptors(self._local_broadcast(lowest_local_id, size_of_local)))
            )

    def random_bootstrap(self):
        self.view = View(
            self._to_descriptors(self._graph_local_broadcast(self.unique_id))
        )

    def lattice_bootstrap(self):
        self.view = View(
            self._to_descriptors(map(
                lambda _: _.target if _.source == self.unique_id else _.source,
                self.model.graph.vs[self.unique_id].all_edges(),
            ))
        )

    def star_bootstrap(self):
        self.view = View(
            self._to_descriptors(list(map(
                lambda _: _.target if _.source == self.unique_id else _.source,
                self.model.graph.vs[self.unique_id].all_edges(),
            ))[:5])
        )

    def die(self):
        self.dead = True
        self.view = View([])
        self.incoming_messages = []
        self.bootstrapped = False

    def resurrect(self):
        self.dead = False

    # Abstract
    def select_peer(self) -> 'Agent':
        raise NotImplemented

    def merge_views(self, view_1: View, view_2: View) -> View:
        raise NotImplemented

    def _merge_views(self, view_1: View, view_2: View, order_by: Callable[[Descriptor], Any]) -> View:
        this = {
            _.address.unique_id: (order_by(_), _)
            for _ in view_1
        }
        other = {
            _.address.unique_id: (order_by(_), _)
            for _ in view_2
        }
        intersection = set(this.keys()) & set(other.keys())
        intersection = {
            _: this[_]
            if this[_][0] < other[_][0]
            else other[_]
            for _ in intersection
        }
        out = this | other
        out.update(intersection)
        try:
            del out[self.unique_id]
        except KeyError:
            pass
        return View(list(
            map(
                lambda _: out[_][1],
                out,
            )
        ))

    def select_view(self, view) -> View:
        raise NotImplemented

    def _select_view(self, view, order_by: Callable[[Descriptor], Any]) -> View:
        sortable = [
            (order_by(_), _)
            for _ in view
        ]
        return View(list(map(
            lambda _: _[1],
            sorted(sortable)[:self.view_size]
        )))

    def prepare_request(self, peer: 'Agent') -> View:
        raise NotImplemented

    def prepare_reply(self, peer: 'Agent') -> View:
        raise NotImplemented

    def __lt__(self, other: 'Agent'):
        return self.unique_id < other.unique_id

    def __repr__(self):
        return f"<{self.__class__.__qualname__} {self.unique_id}>"


class Newscast(Agent):
    def select_peer(self) -> 'Agent':
        return self.view[random.randrange(0, len(self.view))].address

    def merge_views(self, view_1: View, view_2: View) -> View:
        return self._merge_views(view_1, view_2, lambda _: _.hop_count)

    def select_view(self, view) -> View:
        return self._select_view(view, lambda _: _.hop_count)

    def prepare_request(self, peer: Agent) -> View:
        v = self.view.descriptors
        v.append(
            Descriptor(address=self, hop_count=0, time=self.model.schedule.time)
        )
        return View(v)

    def prepare_reply(self, peer: Agent) -> View:
        return self.prepare_request(peer)


class Cyclon(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cemetery = View([])

    def step(self) -> None:
        self.cemetery = View([])
        super().step()

    def select_peer(self) -> Agent:
        oldest = None
        oldest_age = -1
        for descriptor in self.view:
            age = self.model.schedule.time - descriptor.time
            if age > oldest_age:
                oldest = descriptor.address
                oldest_age = age
        return oldest

    def merge_views(self, view_1: View, view_2: View) -> View:
        return self._merge_views(view_1, view_2, lambda _: self.model.schedule.time - _.time)

    def select_view(self, view) -> View:
        v = self._select_view(view, lambda _: self.model.schedule.time - _.time) + self.cemetery
        return View(v.descriptors[:self.view_size])

    def prepare_request(self, peer: Agent) -> View:
        v = self.view.descriptors.copy()
        random.shuffle(v)
        v = v[:self.view_to_send_size - 1]
        v.append(
            Descriptor(address=self, hop_count=0, time=self.model.schedule.time)
        )
        return View(v)

    def prepare_reply(self, peer: Agent) -> View:
        to_remove = sorted(
            self.view.descriptors.copy(),
            key=lambda _: self.model.schedule.time - _.time
        )
        to_remove = View(to_remove[:self.view_to_send_size])
        self.view = self.view - to_remove
        self.cemetery += to_remove
        return to_remove


class Protocol(Enum):
    NEWSCAST = "newscast"
    CYCLON = "cyclon"

    @property
    def klass(self):
        match self:
            case Protocol.NEWSCAST:
                return Newscast
            case Protocol.CYCLON:
                return Cyclon
