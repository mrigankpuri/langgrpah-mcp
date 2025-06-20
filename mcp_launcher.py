#!/usr/bin/env python
"""
MCP Server Launcher - Easily connect to different MCP servers
"""
import asyncio
import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any
import subprocess

def load_config():
    """Load MCP server configurations"""
    config_file = Path("mcp_servers.json")
    
    if not config_file.exists():
        # Create default config
        default_config = {
            "servers": {
                "weather": {
                    "url": "http://localhost:8001/mcp",
                    "name": "Weather Server",
                    "description": "FastMCP weather server with streaming",
                    "model": "gpt-3.5-turbo"
                },
                "example": {
                    "url": "http://localhost:8002/mcp", 
                    "name": "Example Server",
                    "description": "Another MCP server",
                    "model": "gpt-4"
                }
            }
        }
        
        with open(config_file, 'w') as f:
            json.dump(default_config, f, indent=2)
        
        print(f"📁 Created default config at {config_file}")
        print("💡 Edit this file to add your MCP servers")
        
    with open(config_file, 'r') as f:
        return json.load(f)

def list_servers(config):
    """List available MCP servers"""
    print("🌐 Available MCP Servers:")
    print("-" * 50)
    
    for server_id, server_info in config["servers"].items():
        print(f"📛 {server_id}")
        print(f"   URL: {server_info['url']}")
        print(f"   Name: {server_info['name']}")
        print(f"   Description: {server_info['description']}")
        print(f"   Model: {server_info.get('model', 'gpt-3.5-turbo')}")
        print()

async def connect_to_server(server_config, additional_args=None):
    """Connect to a specific MCP server"""
    cmd = [
        "python", "generic_streaming_client.py",
        "--url", server_config["url"],
        "--name", server_config["name"],
        "--model", server_config.get("model", "gpt-3.5-turbo")
    ]
    
    if additional_args:
        cmd.extend(additional_args)
    
    print(f"🚀 Launching connection to {server_config['name']}...")
    print(f"🌐 URL: {server_config['url']}")
    print(f"🤖 Model: {server_config.get('model', 'gpt-3.5-turbo')}")
    print("-" * 50)
    
    # Launch the generic client
    process = await asyncio.create_subprocess_exec(*cmd)
    await process.wait()

def add_server(config):
    """Add a new MCP server to config"""
    print("➕ Add New MCP Server")
    print("-" * 30)
    
    server_id = input("Server ID (e.g., 'myserver'): ").strip()
    if server_id in config["servers"]:
        print(f"❌ Server '{server_id}' already exists!")
        return
    
    url = input("Server URL (e.g., 'http://localhost:8001/mcp'): ").strip()
    name = input("Display Name: ").strip()
    description = input("Description: ").strip()
    model = input("OpenAI Model (default: gpt-3.5-turbo): ").strip() or "gpt-3.5-turbo"
    
    config["servers"][server_id] = {
        "url": url,
        "name": name,
        "description": description,
        "model": model
    }
    
    # Save updated config
    with open("mcp_servers.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Added server '{server_id}' to configuration!")

def remove_server(config):
    """Remove an MCP server from config"""
    list_servers(config)
    
    server_id = input("Enter server ID to remove: ").strip()
    if server_id not in config["servers"]:
        print(f"❌ Server '{server_id}' not found!")
        return
    
    confirm = input(f"Are you sure you want to remove '{server_id}'? (y/N): ").strip().lower()
    if confirm == 'y':
        del config["servers"][server_id]
        
        # Save updated config
        with open("mcp_servers.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Removed server '{server_id}' from configuration!")
    else:
        print("❌ Cancelled")

async def main():
    parser = argparse.ArgumentParser(description='MCP Server Launcher')
    parser.add_argument('action', nargs='?', choices=['list', 'connect', 'add', 'remove'], 
                       default='list', help='Action to perform')
    parser.add_argument('server_id', nargs='?', help='Server ID to connect to')
    parser.add_argument('--url', help='Direct URL to connect to (bypasses config)')
    parser.add_argument('--name', help='Server name (used with --url)')
    parser.add_argument('--model', help='OpenAI model to use')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config()
    
    if args.action == 'list':
        list_servers(config)
        
    elif args.action == 'add':
        add_server(config)
        
    elif args.action == 'remove':
        remove_server(config)
        
    elif args.action == 'connect':
        if args.url:
            # Direct connection
            server_config = {
                "url": args.url,
                "name": args.name or "Direct Connection",
                "model": args.model or "gpt-3.5-turbo"
            }
            await connect_to_server(server_config)
            
        elif args.server_id:
            # Connect to configured server
            if args.server_id not in config["servers"]:
                print(f"❌ Server '{args.server_id}' not found!")
                print("\nAvailable servers:")
                list_servers(config)
                return
            
            server_config = config["servers"][args.server_id]
            
            # Override model if provided
            if args.model:
                server_config = server_config.copy()
                server_config["model"] = args.model
                
            await connect_to_server(server_config)
        else:
            print("❌ Please specify a server_id or use --url for direct connection")
            print("\nAvailable servers:")
            list_servers(config)

if __name__ == "__main__":
    asyncio.run(main()) 