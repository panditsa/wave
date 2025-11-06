from dataclasses import dataclass
from typing import Any, Sequence, TYPE_CHECKING
import torch.fx as fx
from ..ops.wave_ops import (
    get_custom,
    Read,
    Write,
    SharedMemoryBarrier,
    WorkgroupBarrier,
    SchedulingBarrier,
)
from ..wave.constraints import Constraint
from .base import define_schedule_op
import logging

if TYPE_CHECKING:
    from .._support.tracing import CapturedTrace

logger = logging.getLogger(__name__)


def get_proxy_result(proxy):
    """
    Get the real result for a proxy from the current schedule context.

    Args:
        proxy: The proxy to resolve

    Returns:
        The real result if found in the context, otherwise None
    """
    from .._support.tracing import ScheduleContext

    current_context = ScheduleContext.current()
    if current_context is not None:
        return current_context.get_proxy_result(proxy)
    return None


def create_schedule_proxy(
    region_graph,
    real_value: Any,
    op_name: str = "schedule_op",
):
    """
    Create a proxy for a schedule operation that embeds the real value.

    Args:
        region_graph: The region graph to create the proxy in
        real_value: The real value to embed in the proxy
        op_name: The name of the operation for debugging

    Returns:
        The created proxy
    """

    # Create a proxy function that returns the embedded real value
    def proxy_func(*proxy_args, **proxy_kwargs):
        return real_value

    # Set the function name for better debugging
    proxy_func.__name__ = op_name

    try:
        proxy = region_graph.create_proxy(
            "call_function",
            proxy_func,
            (),
            {},
        )

        from .._support.tracing import ScheduleContext

        current_context = ScheduleContext.current()
        if current_context is not None:
            current_context.proxy_to_results[proxy] = real_value

        return proxy

    except Exception as e:
        logger.exception("Failed to create schedule proxy for op '%s': %s", op_name, e)
        raise ValueError(
            f"Failed to create schedule proxy for op '{op_name}': {e}"
        ) from e


def empty_proxy(name: str = "empty_proxy"):
    from .._support.tracing import ScheduleContext

    current_context = ScheduleContext.current()
    if current_context is not None:
        return create_schedule_proxy(
            current_context.region_graph,
            None,
            name,
        )


# Stubs to enable type checking of the custom schedule ops - decorated with @define_op for dispatch
@define_schedule_op
def get_node_by_tag(tag: str): ...


@define_schedule_op
def get_node_by_tag_and_type(tag: str, node_type: Any): ...


@define_schedule_op
def partition_by_address_space(node: Any, address_space: Any): ...


@define_schedule_op
def pipeline(iterate: Sequence[fx.Node]): ...


@define_schedule_op
def getitem(obj: Any, index: int): ...


def get_node_by_tag_helper(kernel_trace, tag: str):
    logger.info(f"Getting node by tag: {tag}")
    nodes = kernel_trace.walk(lambda x: get_custom(x).tag == tag)
    logger.info(f"Found {len(nodes)} nodes by tag: {tag}")
    return nodes


@dataclass
class CustomScheduleOp:
    """Base class for custom schedule operations."""

    pass


@dataclass
class GetNodeByTag(CustomScheduleOp):
    tag: str
    schedule_op_name = "get_node_by_tag"

    @classmethod
    def handle(
        cls, region_graph, kernel_trace, constraints: list[Constraint], tag: str
    ):
        # Always execute the real logic during tracing to apply scheduling
        real_result = get_node_by_tag_helper(kernel_trace, tag)

        # Create a proxy that embeds the real result
        return create_schedule_proxy(
            region_graph,
            real_result,
            cls.schedule_op_name,
        )


@dataclass
class GetNodeByTagAndType(CustomScheduleOp):
    tag: str
    node_type: Any
    schedule_op_name = "get_node_by_tag_and_type"

    @classmethod
    def handle(
        cls,
        region_graph,
        kernel_trace,
        constraints: list[Constraint],
        tag: str,
        node_type: Any,
    ):
        assert constraints is not None, "Constraints are required"

        nodes = get_node_by_tag_helper(kernel_trace, tag)
        nodes = [node for node in nodes if isinstance(get_custom(node), node_type)]
        logger.info(f"Found {len(nodes)} nodes by tag: {tag} and type: {node_type}")

        return create_schedule_proxy(
            region_graph,
            nodes,
            cls.schedule_op_name,
        )


@dataclass
class PartitionByAddressSpace(CustomScheduleOp):
    node: Any
    address_space: Any
    schedule_op_name = "partition_by_address_space"

    @classmethod
    def handle(
        cls,
        region_graph,
        kernel_trace,
        constraints: list[Constraint],
        nodes: Sequence[fx.Node],
        address_space: Any,
    ):
        matched, unmatched = [], []

        assert hasattr(
            nodes, "node"
        ), f"Expected 'nodes' to be a proxy object with a 'node' attribute, but got type: {type(nodes).__name__}"
        nodes = get_proxy_result(nodes.node)
        assert nodes is not None, "Nodes must have a result"
        assert len(nodes) > 0, "Nodes must have at least one element"

        assert all(
            [
                isinstance(get_custom(node), Read)
                or isinstance(get_custom(node), Write)
                for node in nodes
            ]
        ), "Nodes to partition must be Read or Write"
        matched = [
            node
            for node in nodes
            if get_custom(node).memory_type.address_space == address_space
        ]
        unmatched = [node for node in nodes if node not in matched]

        return create_schedule_proxy(
            region_graph,
            (matched, unmatched),
            cls.schedule_op_name,
        )


class PipelinedLoop:
    def __init__(
        self,
        iterate: Sequence[fx.Node],
        kernel_trace: "CapturedTrace",
        constraints: list[Constraint],
    ):
        self.iterate = iterate
        self.kernel_trace = kernel_trace
        self.constraints = constraints
        self.initiation_interval = None
        self.num_stages = 0
        self.barrier_insertions = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        from ..wave.scheduling.schedule import (
            propagate_scheduling_parameters_to_iter_args,
            apply_pipelined_schedule,
        )
        from ..wave.scheduling.schedule_enums import SchedulingType

        result = get_proxy_result(self.iterate)
        assert result is not None, "Iterate must have a result"
        assert isinstance(result, Sequence), "Iterate must be a sequence"
        assert len(result) == 1, "Iterate must have exactly one element"

        custom_iterate = get_custom(result[0])
        subgraph = self.kernel_trace.get_subgraph(custom_iterate.subgraph_name)

        propagate_scheduling_parameters_to_iter_args(
            subgraph,
            self.initiation_interval,
        )

        # Only use scheduling barriers if the user has not provided barrier insertions
        use_scheduling_barriers = len(self.barrier_insertions) == 0

        apply_pipelined_schedule(
            custom_iterate,
            subgraph,
            self.kernel_trace,
            self.constraints,
            use_scheduling_barriers=use_scheduling_barriers,
            num_stages=self.num_stages,
            initiation_interval=self.initiation_interval,
            scheduling_type=SchedulingType.MANUAL,
            visualize=False,
            multi_buffer_count=None,
            barrier_insertions=self.barrier_insertions,
        )

    def set_stage(self, nodes: Sequence[fx.Node]):
        stage = self.num_stages
        if self.initiation_interval is None:
            self.initiation_interval = len(nodes)
        else:
            assert self.initiation_interval == len(
                nodes
            ), "The number of clusters must be the same across stages"
        result_clusters = []
        for cluster in nodes:
            result_nodes = []
            for node in cluster:
                node_result = get_proxy_result(node)
                assert node_result is not None, "Nodes must have a result"
                result_nodes.append(node_result)
            result_clusters.append(tuple(result_nodes))

        node_order = 0
        for i, cluster in enumerate(result_clusters):
            for nodes in cluster:
                for node in nodes:
                    custom = get_custom(node)
                    custom.scheduling_parameters = {
                        "absolute_cycle": stage * self.initiation_interval + i,
                        "stage": stage,
                        "initiation_interval": self.initiation_interval,
                        "prefetch_stage": None,
                        "order": node_order,
                    }
                    custom.scheduling_parameters["cycle"] = (
                        custom.scheduling_parameters["absolute_cycle"]
                        % self.initiation_interval
                    )
                    node_order += 1

        self.num_stages += 1
        # During tracing, return a proxy for the set_stage operation
        return empty_proxy("set_stage")

    def insert_barrier_after(
        self, proxy, barrier_type="shared_memory", pipeline_stage=None
    ):
        """
        Insert a barrier after the operations in proxy.
        Creates the barrier nodes immediately and stores them for use during loop reconstruction.

        Args:
            proxy: Proxy object containing target nodes to insert barriers after
            barrier_type: Type of barrier ("shared_memory", "workgroup", or "scheduling")
            pipeline_stage: Which pipeline stage to insert the barrier in (None for all stages)
        """

        # assert that barrier type is one of "shared_memory" or "workgroup" or "scheduling"
        assert barrier_type in [
            "shared_memory",
            "workgroup",
            "scheduling",
        ], "Invalid barrier type"

        # Get the actual nodes from the proxy
        target_nodes = get_proxy_result(proxy)
        if not target_nodes:
            logger.warning("No target nodes found for barrier insertion")
            return self

        if not isinstance(target_nodes, (list, tuple)):
            target_nodes = [target_nodes]

        # Get the subgraph from the iterate
        result = get_proxy_result(self.iterate)
        assert (
            result is not None and len(result) == 1
        ), "Iterate must have exactly one element"
        custom_iterate = get_custom(result[0])
        subgraph = self.kernel_trace.get_subgraph(custom_iterate.subgraph_name)

        # Only create a barrier after the LAST node in the proxy to avoid duplicates
        last_target_node = target_nodes[-1]

        # Get scheduling parameters and location from the last target node
        custom = get_custom(last_target_node)
        if (
            not hasattr(custom, "scheduling_parameters")
            or custom.scheduling_parameters is None
        ):
            logger.warning(
                f"Node {last_target_node} has no scheduling parameters, skipping barrier insertion"
            )
            return self

        # Create the barrier object (but don't add to graph yet)
        if barrier_type == "scheduling":
            barrier = SchedulingBarrier([])
        elif barrier_type == "workgroup":
            barrier = WorkgroupBarrier()
        else:
            barrier = SharedMemoryBarrier()

        # Store barrier object and metadata keyed by the LAST target node only
        self.barrier_insertions[last_target_node] = {
            "barrier_op": barrier,  # The barrier CustomOp object (not yet added to graph)
            "barrier_type": barrier_type,
            "pipeline_stage": pipeline_stage,
            "location": custom.location,  # Store location for later use
            "scheduling_parameters": custom.scheduling_parameters.copy(),  # Store scheduling params
        }

        return self


@dataclass
class GetItem(CustomScheduleOp):
    obj: Any
    index: int
    schedule_op_name = "getitem"

    @classmethod
    def handle(
        cls,
        region_graph,
        kernel_trace,
        constraints: list[Constraint],
        obj: Any,
        index: int,
    ):
        if hasattr(obj, "node"):
            source_result = get_proxy_result(obj.node)
            if isinstance(source_result, (list, tuple)) and len(source_result) > index:
                real_result = source_result[index]
            else:
                raise ValueError(f"Index {index} out of bounds for {source_result}")
        else:
            raise ValueError(f"Object {obj} is not a proxy")

        return create_schedule_proxy(
            region_graph,
            real_result,
            cls.schedule_op_name,
        )


@dataclass
class Pipeline(CustomScheduleOp):
    schedule_op_name = "pipeline"

    @classmethod
    def handle(
        cls,
        region_graph,
        kernel_trace,
        constraints: list[Constraint],
        iterate: Sequence[fx.Node],
    ):
        real_pipelined_loop = PipelinedLoop(iterate, kernel_trace, constraints)

        create_schedule_proxy(
            region_graph,
            real_pipelined_loop,
            cls.schedule_op_name,
        )

        # For context manager support, return the real object
        return real_pipelined_loop
