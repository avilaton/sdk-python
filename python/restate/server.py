#
#  Copyright (c) 2023-2024 - Restate Software, Inc., Restate GmbH
#
#  This file is part of the Restate SDK for Python,
#  which is released under the MIT license.
#
#  You can find a copy of the license in file LICENSE in the root
#  directory of this repository or package, or at
#  https://github.com/restatedev/sdk-typescript/blob/main/LICENSE
#
"""This module contains the ASGI server for the restate framework using Starlette."""

import asyncio
import logging
import signal
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable, Dict, Literal, Optional, Set

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import Route

from restate.discovery import compute_discovery_json
from restate.endpoint import Endpoint
from restate.server_context import ServerInvocationContext, DisconnectedException
from restate.server_types import Receive, ReceiveChannel, RestateAppT, Scope, Send, binary_to_header, header_to_binary  # pylint: disable=line-too-long
from restate.vm import VMWrapper
from restate.aws_lambda import is_running_on_lambda, wrap_asgi_as_lambda_handler

from restate._internal import PyIdentityVerifier, IdentityVerificationException  # pylint: disable=import-error,no-name-in-module
from restate._internal import SDK_VERSION  # pylint: disable=import-error,no-name-in-module

logger = logging.getLogger(__name__)

X_RESTATE_SERVER_STR = f"restate-sdk-python/{SDK_VERSION}"
X_RESTATE_SERVER = header_to_binary([("x-restate-server", X_RESTATE_SERVER_STR)])


async def send_status(send: Send, receive: Receive, status_code: int):
    """respond with a status code"""
    await send({"type": "http.response.start", "status": status_code, "headers": X_RESTATE_SERVER})
    # For more info on why this loop, see ServerInvocationContext.leave()
    # pylint: disable=R0801
    while True:
        event = await receive()
        if event is None:
            break
        if event.get("type") == "http.disconnect":
            break
        if event.get("type") == "http.request" and event.get("more_body", False) is False:
            break
    await send({"type": "http.response.body"})


async def process_invocation_to_completion(
    vm: VMWrapper, handler, attempt_headers: Dict[str, str], receive: ReceiveChannel, send: Send
):
    """Invoke the user code."""
    status, res_headers = vm.get_response_head()
    res_bin_headers = header_to_binary(res_headers)
    res_bin_headers.extend(X_RESTATE_SERVER)
    await send({"type": "http.response.start", "status": status, "headers": res_bin_headers, "trailers": False})
    assert status == 200
    # ========================================
    # Read the input and the journal
    # ========================================
    while True:
        message = await receive()
        if message.get("type") == "http.disconnect":
            # everything ends here really ...
            return
        if message.get("type") == "http.request":
            body = message.get("body", None)
            assert isinstance(body, bytes)
            vm.notify_input(body)
        if not message.get("more_body", False):
            vm.notify_input_closed()
            break
        if vm.is_ready_to_execute():
            break
    # ========================================
    # Execute the user code
    # ========================================
    invocation = vm.sys_input()
    context = ServerInvocationContext(
        vm=vm, handler=handler, invocation=invocation, attempt_headers=attempt_headers, send=send, receive=receive
    )
    try:
        await context.enter()
    except asyncio.exceptions.CancelledError:
        context.on_attempt_finished()
        raise
    except DisconnectedException:
        # The client disconnected before we could send the response
        context.on_attempt_finished()
        return
    # pylint: disable=W0718
    except Exception:
        logger.exception("Exception in Restate handler")
    try:
        await context.leave()
    finally:
        context.on_attempt_finished()


def asgi_app(endpoint: Endpoint, lifespan: Optional[Callable] = None) -> RestateAppT:
    """Create a Starlette ASGI app for the given endpoint."""

    # Prepare request signer
    identity_verifier = PyIdentityVerifier(endpoint.identity_keys)

    active_channels: Set[ReceiveChannel] = set()

    def _on_sigterm() -> None:
        """Notify all active receive channels of graceful shutdown."""
        for ch in active_channels:
            ch.notify_shutdown()

    @asynccontextmanager
    async def _internal_lifespan(app: Starlette) -> AsyncIterator[None]:
        loop = asyncio.get_running_loop()
        try:
            loop.add_signal_handler(signal.SIGTERM, _on_sigterm)
        except (NotImplementedError, RuntimeError):
            pass  # Windows or non-main thread
        if lifespan is not None:
            async with lifespan(app):
                yield
        else:
            yield

    async def health(_request: Request) -> Response:
        """Handle /health requests."""
        return Response(
            content=b'{"status":"ok"}',
            media_type="application/json",
            headers={"x-restate-server": X_RESTATE_SERVER_STR},
        )

    async def discover(request: Request) -> Response:
        """Handle /discover requests."""
        request_headers = [(k.decode(), v.decode()) for k, v in request.headers.raw]
        request_path = request.url.path
        try:
            identity_verifier.verify(request_headers, request_path)
        except IdentityVerificationException:
            return Response(status_code=401, headers={"x-restate-server": X_RESTATE_SERVER_STR})

        discovered_as: Literal["request_response", "bidi"]
        if request.scope["http_version"] == "1.1":
            discovered_as = "request_response"
        else:
            discovered_as = "bidi"

        accept_header = request.headers.get("accept")
        version = 2
        if accept_header:
            if "application/vnd.restate.endpointmanifest.v4+json" in accept_header:
                version = 4
            elif "application/vnd.restate.endpointmanifest.v3+json" in accept_header:
                version = 3
            elif "application/vnd.restate.endpointmanifest.v2+json" in accept_header:
                version = 2
            else:
                return Response(
                    content=f"Unsupported discovery version {accept_header}".encode(),
                    media_type="text/plain",
                    status_code=415,
                    headers={"x-restate-server": X_RESTATE_SERVER_STR},
                )

        try:
            js = compute_discovery_json(endpoint, version, discovered_as)
            return Response(
                content=js.encode("utf-8"),
                media_type=f"application/vnd.restate.endpointmanifest.v{version}+json",
                headers={"x-restate-server": X_RESTATE_SERVER_STR},
            )
        except ValueError as e:
            return Response(
                content=f"Error when computing discovery {e}".encode(),
                media_type="text/plain",
                status_code=500,
                headers={"x-restate-server": X_RESTATE_SERVER_STR},
            )

    class InvokeHandler:
        """Raw ASGI handler for /invoke/{service}/{handler} — supports bidirectional streaming."""

        async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
            service_name = scope["path_params"]["service"]
            handler_name = scope["path_params"]["handler"]
            request_path = scope["path"]

            assert not isinstance(scope["headers"], str)
            assert hasattr(scope["headers"], "__iter__")
            request_headers = binary_to_header(scope["headers"])

            try:
                identity_verifier.verify(request_headers, request_path)
            except IdentityVerificationException:
                await send_status(send, receive, 401)
                return

            service = endpoint.services.get(service_name)
            if not service:
                await send_status(send, receive, 404)
                return
            handler = service.handlers.get(handler_name)
            if not handler:
                await send_status(send, receive, 404)
                return

            receive_channel = ReceiveChannel(receive)
            active_channels.add(receive_channel)
            try:
                await process_invocation_to_completion(
                    VMWrapper(request_headers), handler, dict(request_headers), receive_channel, send
                )
            except Exception:  # pylint: disable=W0718
                logger.exception("Exception in ASGI app")
            finally:
                active_channels.discard(receive_channel)
                await receive_channel.close()

    routes = [
        Route("/health", endpoint=health),
        Route("/discover", endpoint=discover, methods=["GET"]),
        Route("/invoke/{service}/{handler}", endpoint=InvokeHandler()),
    ]

    app = Starlette(routes=routes, lifespan=_internal_lifespan)

    if is_running_on_lambda():
        # If we're on Lambda, just return the adapter
        return wrap_asgi_as_lambda_handler(app)

    return app
