# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: LicenseRef-NVIDIA-SOFTWARE-LICENSE

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from cuda.core.experimental._stream import Stream
from cuda.core.experimental._utils import driver, handle_return, precondition


@dataclass
class DebugPrintOptions:
    """ """

    VERBOSE: bool = False
    RUNTIME_TYPES: bool = False
    KERNEL_NODE_PARAMS: bool = False
    MEMCPY_NODE_PARAMS: bool = False
    MEMSET_NODE_PARAMS: bool = False
    HOST_NODE_PARAMS: bool = False
    EVENT_NODE_PARAMS: bool = False
    EXT_SEMAS_SIGNAL_NODE_PARAMS: bool = False
    EXT_SEMAS_WAIT_NODE_PARAMS: bool = False
    KERNEL_NODE_ATTRIBUTES: bool = False
    HANDLES: bool = False
    MEM_ALLOC_NODE_PARAMS: bool = False
    MEM_FREE_NODE_PARAMS: bool = False
    BATCH_MEM_OP_NODE_PARAMS: bool = False
    EXTRA_TOPO_INFO: bool = False
    CONDITIONAL_NODE_PARAMS: bool = False


class GraphBuilder:
    """TBD

    Directly creating a :obj:`~_graph.GraphBuilder` is not supported due
    to ambiguity. New graph builders should instead be created through a
    :obj:`~_device.Device`, or a :obj:`~_stream.stream` object
    """

    def __init__(self):
        raise NotImplementedError(
            "directly creating a Graph object can be ambiguous. Please either "
            "call Device.create_graph() or stream.creating_graph()"
        )

    @staticmethod
    def _init(stream, is_root=True):
        self = GraphBuilder.__new__(GraphBuilder)
        # TODO: I need to know if we own this stream object.
        #       If from Device(), then we can destroy it on close
        #       If from Stream, then we can't
        self._stream = stream
        self._capturing = False
        self._is_root = is_root  # TODO: Is this info needed?
        return self

    def _check_capture_stream_provided(self, *args, **kwargs):
        if self._stream == None:
            raise RuntimeError("Tried to use a stream capture operation on a graph builder without a stream")

    @property
    def legacy_stream_capture(self) -> Stream:
        return self._stream

    @property
    def is_root_builder(self) -> bool:
        return self._is_root_builder

    @precondition(_check_capture_stream_provided)
    def begin_capture(self, mode="global"):
        # Supports "global", "local" or "relaxed"
        # TODO; Test case for each mode and fail
        if mode == "global":
            capture_mode = driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_GLOBAL
        elif mode == "local":
            capture_mode = driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
        elif mode == "relaxed":
            capture_mode = driver.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_RELAXED
        else:
            raise ValueError(f"Only 'global', 'local' or 'relaxed' capture mode are supported, got {capture_mode}")

        handle_return(driver.cuStreamBeginCapture(self._stream.handle, capture_mode))
        self._capturing = True

    @precondition(_check_capture_stream_provided)
    def is_capture_active(self) -> bool:
        result = handle_return(driver.cuStreamGetCaptureInfo(self._stream.handle))

        capture_status = result[0]
        if capture_status == driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE:
            return False
        elif capture_status == driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE:
            return True
        elif capture_status == driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_INVALIDATED:
            raise RuntimeError(
                "Stream is part of a capture sequence that has been invalidated, but "
                "not terminated. The capture sequence must be terminated with self.`()."
            )
        else:
            raise NotImplementedError(f"Unsupported capture stuse type received: {capture_status}")

    @precondition(_check_capture_stream_provided)
    def end_capture(self):
        if not self._capturing:
            raise RuntimeError("Stream is not capturing")

        self._graph = handle_return(driver.cuStreamEndCapture(self._stream.handle))
        self._capturing = False

    def debug_dot_print(self, path, options: Optional[DebugPrintOptions] = None):
        # TODO: We should be able to print one while the capture is happening right? Just need to make sure driver version is new enough.
        if self._graph == None:
            raise RuntimeError("Graph needs to be built before generating a DOT debug file")

        # TODO: Apply each option to the value
        options_value = 0

        handle_return(driver.cuGraphDebugDotPrint(self._graph, path, options_value))

    def fork(self, count) -> Tuple[Graph, ...]:
        if count <= 1:
            raise ValueError(f"Invalid fork count: expecting >= 2, got {count}")

        # 1. Record an event on our stream
        event = self._stream.record()

        # TODO: Steps 2,3,4 can be combined under a single loop

        # 2. Create a streams for each of the new forks
        # TODO: Optimization where one of the fork stream is allowed to use
        # TODO: Should use the same stream options as initial stream??
        fork_stream = [self._stream.device.create_stream() for i in range(count)]

        # 3. Have each new stream wait on our singular event
        for stream in fork_stream:
            stream.wait(event)

        # 4. Discard the event
        # TODO: Is this actually allowed when using with a graph? Surely, since it just needs to create an edge for us... right?
        event.close()

        # 5. Create new graph builders for each new stream fork
        return [GraphBuilder._init(stream=stream, is_root=False) for stream in fork_stream]

    def join(self, *graph_builders):
        if len(graph_builders) < 1:
            raise ValueError("Must specify which graphs should join but none were given")

        for graph in graph_builders:
            self._stream.wait(graph.legacy_stream_capture)
            # TODO: Can we close each of those new streams? Do we need weakref?
            graph.close()

    def create_conditional_handle(self, default_value=None):
        pass

    def if_cond(self, handle):
        pass

    def if_else(self, handle):
        pass

    def switch(self, handle, count):
        pass

    def close(self):
        if self._capturing:
            raise RuntimeError("Trying to close a graph builder who is still capturing")
        # Can we call this directly? Or relying on weakref enough?
        self._stream.close()


class Graph:
    """ """

    def __init__(self):
        raise RuntimeError("directly constructing a Graph instance is not supported")

    @staticmethod
    def _init(graph):
        self = Graph.__new__(Graph)
        self._graph = graph
        return self
