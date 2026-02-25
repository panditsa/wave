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
    assert (
        "stage" in custom.scheduling_parameters
    ), f"Node {node} has scheduling_parameters but no 'stage' key"
    write_stage = custom.scheduling_parameters["stage"]
    has_users = False
    for user in custom.users:
        if user.scheduling_parameters is None:
            continue
        assert (
            "stage" in user.scheduling_parameters
        ), f"User {user.fx_node} has scheduling_parameters but no 'stage' key"
        has_users = True
        if user.scheduling_parameters["stage"] <= write_stage:
            return False
    return has_users


def optimize_buffer_counts(
    buffer_counts: dict[fx.Node, int],
    node_to_buffer: dict[fx.Node, fx.Node],
) -> dict[fx.Node, int]:
    """Reduce buffer counts where the consume-before-write pattern applies.

    For each allocation that requires >= 2 buffers, if *all* write nodes exhibit
    the consume-before-write pattern (all reads at a later stage), reduce the
    count by one.  Each allocation is reduced at most once.

    We require ``all`` rather than ``any`` because the buffer count is the max
    across all write nodes.  If even one write lacks the pattern, that write
    still needs the full count and reducing would be unsound.
    """
    for alloc_node in list(buffer_counts):
        if buffer_counts[alloc_node] < 2:
            continue
        write_nodes = [w for w, a in node_to_buffer.items() if a is alloc_node]
        if not write_nodes or not all(
            _has_consume_before_write(w) for w in write_nodes
        ):
            continue

        count = buffer_counts[alloc_node]
        # The buffer count ceil((lifetime+1)/II) includes +1 for the worst
        # case where the write to the new value and the read of the old value
        # overlap within the same II window, requiring both copies to coexist.
        # Consume-before-write proves this overlap cannot happen (all reads
        # complete before the write fires), so exactly one copy is
        # unnecessary.  Reducing by more than 1 is not possible from this
        # pattern alone: there is only one read/write overlap boundary per
        # write, so the pattern can only eliminate one extra buffer slot.
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
