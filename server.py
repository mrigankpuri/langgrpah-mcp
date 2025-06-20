#!/usr/bin/env python3
"""
MCP Server with official streaming capabilities using StreamableHTTPSessionManager
"""

import contextlib
import logging
from collections.abc import AsyncIterator
import anyio
import click
import mcp.types as types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from starlette.applications import Starlette
from starlette.routing import Mount
from starlette.types import Receive, Scope, Send

logger = logging.getLogger(__name__)


@click.command()
@click.option("--port", default=8000, help="Port to listen on for HTTP")
@click.option(
    "--log-level",
    default="INFO",
    help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
)
@click.option(
    "--json-response",
    is_flag=True,
    default=False,
    help="Enable JSON responses instead of SSE streams",
)
def main(
        port: int,
        log_level: str,
        json_response: bool,
) -> int:
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    app = Server("streaming-mcp-server")

    @app.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
        ctx = app.request_context

        if name == "streaming_task":
            task_name = arguments.get("task_name", "default_task")
            steps = arguments.get("steps", 3)
            delay = arguments.get("delay", 1.0)

            logger.info(f"Starting streaming task: {task_name}")

            # Send streaming progress messages
            for i in range(steps):
                await ctx.session.send_log_message(
                    level="info",
                    data=f"Step {i + 1}/{steps}: Processing {task_name}...",
                    logger="streaming_task",
                    related_request_id=ctx.request_id,
                )

                if i < steps - 1:  # Don't wait after the last step
                    await anyio.sleep(delay)

            # Send completion message
            await ctx.session.send_log_message(
                level="info",
                data=f"Task '{task_name}' completed successfully!",
                logger="streaming_task",
                related_request_id=ctx.request_id,
            )

            return [
                types.TextContent(
                    type="text",
                    text=f"Streaming task '{task_name}' completed with {steps} steps",
                )
            ]

        elif name == "data_processing":
            dataset = arguments.get("dataset", "sample_data")
            batch_size = arguments.get("batch_size", 100)
            total_records = arguments.get("total_records", 1000)

            logger.info(f"Starting data processing: {dataset}")

            batches = (total_records + batch_size - 1) // batch_size

            for batch in range(batches):
                start_record = batch * batch_size
                end_record = min((batch + 1) * batch_size, total_records)

                await ctx.session.send_log_message(
                    level="info",
                    data=f"Processing batch {batch + 1}/{batches}: records {start_record}-{end_record}",
                    logger="data_processing",
                    related_request_id=ctx.request_id,
                )

                # Simulate processing time
                await anyio.sleep(0.5)

            # Send final summary
            await ctx.session.send_log_message(
                level="info",
                data=f"Data processing complete! Processed {total_records} records in {batches} batches",
                logger="data_processing",
                related_request_id=ctx.request_id,
            )

            return [
                types.TextContent(
                    type="text",
                    text=f"Successfully processed {total_records} records from {dataset}",
                )
            ]

        elif name == "health_check":
            await ctx.session.send_log_message(
                level="info",
                data="Health check initiated",
                logger="health_check",
                related_request_id=ctx.request_id,
            )

            # Simulate checking various components
            components = ["database", "cache", "external_api", "file_system"]

            for component in components:
                await ctx.session.send_log_message(
                    level="info",
                    data=f"Checking {component}... OK",
                    logger="health_check",
                    related_request_id=ctx.request_id,
                )
                await anyio.sleep(0.3)

            await ctx.session.send_log_message(
                level="info",
                data="All systems operational!",
                logger="health_check",
                related_request_id=ctx.request_id,
            )

            return [
                types.TextContent(
                    type="text",
                    text="Health check completed - all systems operational",
                )
            ]

        else:
            return [
                types.TextContent(
                    type="text",
                    text=f"Unknown tool: {name}",
                )
            ]

    @app.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="streaming_task",
                description="Execute a task with streaming progress updates",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "task_name": {
                            "type": "string",
                            "description": "Name of the task to execute",
                            "default": "default_task"
                        },
                        "steps": {
                            "type": "integer",
                            "description": "Number of steps in the task",
                            "default": 3,
                            "minimum": 1,
                            "maximum": 10
                        },
                        "delay": {
                            "type": "number",
                            "description": "Delay between steps in seconds",
                            "default": 1.0,
                            "minimum": 0.1,
                            "maximum": 5.0
                        }
                    },
                    "required": ["task_name"]
                }
            ),
            types.Tool(
                name="data_processing",
                description="Process data with streaming batch updates",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "dataset": {
                            "type": "string",
                            "description": "Name of the dataset to process",
                            "default": "sample_data"
                        },
                        "batch_size": {
                            "type": "integer",
                            "description": "Number of records per batch",
                            "default": 100,
                            "minimum": 10,
                            "maximum": 1000
                        },
                        "total_records": {
                            "type": "integer",
                            "description": "Total number of records to process",
                            "default": 1000,
                            "minimum": 1,
                            "maximum": 10000
                        }
                    },
                    "required": ["dataset"]
                }
            ),
            types.Tool(
                name="health_check",
                description="Perform system health check with streaming component status",
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False
                }
            )
        ]

    # Create the session manager with streaming capabilities
    session_manager = StreamableHTTPSessionManager(
        app=app,
        event_store=None,  # No event store for stateless operation
        json_response=json_response,
        stateless=True,  # Enable stateless mode
    )

    async def handle_streamable_http(
            scope: Scope, receive: Receive, send: Send
    ) -> None:
        await session_manager.handle_request(scope, receive, send)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        """Context manager for session manager."""
        async with session_manager.run():
            logger.info("MCP Streaming Server started successfully!")
            logger.info(f"Available tools: streaming_task, data_processing, health_check")
            try:
                yield
            finally:
                logger.info("MCP Streaming Server shutting down...")

    # Create ASGI application
    starlette_app = Starlette(
        debug=True,
        routes=[
            Mount("/mcp", app=handle_streamable_http),
        ],
        lifespan=lifespan,
    )

    import uvicorn
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)

    return 0


if __name__ == "__main__":
    main()