"""
Post-scheduling buffer count optimization.

Reduces shared memory buffer counts in software-pipelined loops by detecting
the consume-before-write pattern: when all reads of a buffer are at later
pipeline stages than the write, the buffer is consumed before being refilled,
allowing the count to be reduced by one (e.g., triple → double buffer).
"""

import torch.fx as fx

from wave_lang.support.logging import get_logger
from ...ops.wave_ops import get_custom

logger = get_logger("wave.scheduling.post_scheduling_buffer_opt")


def _has_consume_before_write(node: fx.Node) -> bool:
    """True if all scheduled users of `node` read at a later pipeline stage."""
    custom = get_custom(node)
    if custom.scheduling_parameters is None:
        return False
    write_stage = custom.scheduling_parameters["stage"]
    has_users = False
    for user in custom.users:
        if user.scheduling_parameters is None:
            continue
        has_users = True
        if user.scheduling_parameters["stage"] <= write_stage:
            return False
    return has_users


def optimize_buffer_counts(
    buffer_counts: dict[fx.Node, int],
    graph: fx.Graph,
    node_to_buffer: dict[fx.Node, fx.Node],
) -> dict[fx.Node, int]:
    """Reduce buffer counts where the consume-before-write pattern applies.

    For each allocation that requires >= 2 buffers, if any write node exhibits
    the consume-before-write pattern (all reads at a later stage), reduce the
    count by one.  Each allocation is reduced at most once.
    """
    for alloc_node in list(buffer_counts):
        if buffer_counts[alloc_node] < 2:
            continue
        write_nodes = [w for w, a in node_to_buffer.items() if a is alloc_node]
        if not any(_has_consume_before_write(w) for w in write_nodes):
            continue

        count = buffer_counts[alloc_node]
        new_count = count - 1
        logger.debug(
            f"Buffer count for {alloc_node} reduced {count} -> {new_count} "
            f"(consume-before-write)"
        )
        if new_count < 2:
            del buffer_counts[alloc_node]
        else:
            buffer_counts[alloc_node] = new_count

    return buffer_counts
