#!/usr/bin/env python3
"""
freeswitch_amd.py - FreeSWITCH AMD Integration via ESL

This script provides integration between FreeSWITCH and the voicemail detection
server via the Event Socket Library (ESL).

Features:
    - Originate calls with AMD detection
    - Stream audio to AMD server via WebSocket
    - Handle AMD results (route to agent or play message)
    - Event-driven architecture using ESL

Requirements:
    pip install python-ESL websockets asyncio

Usage:
    # As a daemon
    python scripts/freeswitch_amd.py --daemon

    # Single call test
    python scripts/freeswitch_amd.py --originate 15551234567 --caller_id 15559876543

    # Monitor mode (watch for AMD events)
    python scripts/freeswitch_amd.py --monitor

Configuration:
    Set environment variables or edit the CONFIG section below.
"""
import os
import sys
import json
import time
import asyncio
import logging
import argparse
from typing import Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor

try:
    import websockets
except ImportError:
    print("Please install websockets: pip install websockets")
    sys.exit(1)

# Try to import ESL - may need custom installation
ESL_AVAILABLE = False
try:
    import ESL
    ESL_AVAILABLE = True
except ImportError:
    print("Warning: python-ESL not available. Install from FreeSWITCH source.")
    print("Some features will be simulated.")

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # FreeSWITCH ESL settings
    "fs_host": os.getenv("FS_HOST", "localhost"),
    "fs_port": int(os.getenv("FS_PORT", "8021")),
    "fs_password": os.getenv("FS_PASSWORD", "ClueCon"),

    # AMD server settings
    "amd_url": os.getenv("AMD_URL", "ws://localhost:8080/ws/amd"),

    # Audio settings
    "sample_rate": 16000,
    "chunk_ms": 20,  # 20ms chunks

    # AMD decision settings
    "amd_timeout_ms": 5000,
    "min_audio_ms": 500,

    # Call routing
    "human_destination": "agent_queue",
    "machine_action": "voicemail",  # 'voicemail', 'hangup', or 'message'
    "voicemail_file": "voicemail_message.wav",

    # Gateway for outbound calls
    "gateway": os.getenv("FS_GATEWAY", "your_gateway"),
}

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("freeswitch_amd")


# ============================================================================
# ESL CONNECTION MANAGEMENT
# ============================================================================

class FreeSwitchConnection:
    """Manage FreeSWITCH ESL connection."""

    def __init__(self, host: str, port: int, password: str):
        self.host = host
        self.port = port
        self.password = password
        self.con = None

    def connect(self) -> bool:
        """Establish ESL connection."""
        if not ESL_AVAILABLE:
            logger.warning("ESL not available - running in simulation mode")
            return True

        try:
            self.con = ESL.ESLconnection(self.host, str(self.port), self.password)
            if self.con.connected():
                logger.info(f"Connected to FreeSWITCH at {self.host}:{self.port}")
                return True
            else:
                logger.error("Failed to connect to FreeSWITCH")
                return False
        except Exception as e:
            logger.error(f"ESL connection error: {e}")
            return False

    def disconnect(self):
        """Close ESL connection."""
        if self.con:
            self.con = None
            logger.info("Disconnected from FreeSWITCH")

    def api(self, command: str) -> str:
        """Execute FreeSWITCH API command."""
        if not ESL_AVAILABLE or not self.con:
            logger.info(f"[SIMULATED] API: {command}")
            return "OK"

        result = self.con.api(command)
        return result.getBody() if result else ""

    def bgapi(self, command: str) -> str:
        """Execute FreeSWITCH background API command."""
        if not ESL_AVAILABLE or not self.con:
            logger.info(f"[SIMULATED] BGAPI: {command}")
            return "OK"

        result = self.con.bgapi(command)
        return result.getBody() if result else ""

    def subscribe_events(self, events: str = "all"):
        """Subscribe to FreeSWITCH events."""
        if not ESL_AVAILABLE or not self.con:
            logger.info(f"[SIMULATED] Subscribe: {events}")
            return

        self.con.events("plain", events)

    def wait_for_event(self, timeout_ms: int = 1000) -> Optional[Dict]:
        """Wait for next event."""
        if not ESL_AVAILABLE or not self.con:
            return None

        event = self.con.recvEventTimed(timeout_ms)
        if event:
            return self._event_to_dict(event)
        return None

    def _event_to_dict(self, event) -> Dict:
        """Convert ESL event to dictionary."""
        result = {}
        for i in range(event.getHeaders()):
            name = event.getHeaderName(i)
            result[name] = event.getHeader(name)
        return result


# ============================================================================
# AMD CLIENT
# ============================================================================

class AMDClient:
    """Client for communicating with AMD WebSocket server."""

    def __init__(self, url: str):
        self.url = url
        self.sample_rate = CONFIG["sample_rate"]
        self.chunk_size = int(self.sample_rate * CONFIG["chunk_ms"] / 1000) * 2  # bytes

    async def classify(self, audio_bytes: bytes, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Send audio to AMD server and get classification.

        Args:
            audio_bytes: PCM16 16kHz mono audio
            timeout: Maximum time to wait for result

        Returns:
            Classification result dictionary
        """
        try:
            async with websockets.connect(self.url) as ws:
                # Send audio in chunks
                for i in range(0, len(audio_bytes), self.chunk_size):
                    chunk = audio_bytes[i:i+self.chunk_size]
                    await ws.send(chunk)
                    await asyncio.sleep(CONFIG["chunk_ms"] / 1000)

                # Request final decision
                await ws.send(json.dumps({"type": "flush"}))

                # Wait for response
                response = await asyncio.wait_for(ws.recv(), timeout=timeout)
                return json.loads(response)

        except asyncio.TimeoutError:
            return {"error": "timeout", "label": "unknown"}
        except Exception as e:
            return {"error": str(e), "label": "unknown"}

    async def stream_classify(self, audio_generator, timeout: float = 5.0) -> Dict[str, Any]:
        """
        Stream audio to AMD server.

        Args:
            audio_generator: Async generator yielding audio chunks
            timeout: Maximum time for final result

        Returns:
            Classification result dictionary
        """
        result = {"label": "unknown"}

        try:
            async with websockets.connect(self.url) as ws:
                # Task to receive responses
                async def receive_responses():
                    nonlocal result
                    while True:
                        try:
                            response = await ws.recv()
                            data = json.loads(response)
                            result = data
                            if data.get("type") == "final":
                                break
                        except:
                            break

                receive_task = asyncio.create_task(receive_responses())

                # Send audio chunks
                async for chunk in audio_generator:
                    await ws.send(chunk)

                # Request final decision
                await ws.send(json.dumps({"type": "flush"}))

                # Wait for final response
                await asyncio.wait_for(receive_task, timeout=timeout)

        except Exception as e:
            result["error"] = str(e)

        return result


# ============================================================================
# CALL HANDLER
# ============================================================================

class CallHandler:
    """Handle calls with AMD detection."""

    def __init__(self, fs_con: FreeSwitchConnection, amd_client: AMDClient):
        self.fs = fs_con
        self.amd = amd_client
        self.executor = ThreadPoolExecutor(max_workers=4)

    def originate_with_amd(self, destination: str, caller_id: str,
                          context: str = "default") -> Dict[str, Any]:
        """
        Originate a call with AMD detection.

        Args:
            destination: Phone number to call
            caller_id: Caller ID to display
            context: Dialplan context

        Returns:
            Result dictionary with call UUID and AMD result
        """
        # Generate UUID
        uuid = self.fs.api("create_uuid").strip()
        logger.info(f"Originating call {uuid} to {destination}")

        # Build dial string
        dial_string = (
            f"{{origination_uuid={uuid},"
            f"origination_caller_id_number={caller_id},"
            f"ignore_early_media=true,"
            f"amd_enabled=true}}"
            f"sofia/gateway/{CONFIG['gateway']}/{destination}"
        )

        # Originate call (parks it for AMD)
        result = self.fs.bgapi(f"originate {dial_string} &park()")
        logger.info(f"Originate result: {result}")

        return {
            "uuid": uuid,
            "destination": destination,
            "status": "originated"
        }

    def handle_amd_result(self, uuid: str, result: Dict[str, Any]):
        """
        Handle AMD classification result.

        Args:
            uuid: Call UUID
            result: AMD classification result
        """
        label = result.get("label", "unknown")
        confidence = result.get("confidence", 0)

        logger.info(f"Call {uuid}: AMD result = {label} ({confidence:.2%})")

        if label == "human":
            # Transfer to agent queue
            logger.info(f"Call {uuid}: Transferring to agent")
            self.fs.api(f"uuid_transfer {uuid} {CONFIG['human_destination']} XML {context}")

        elif label == "machine":
            action = CONFIG["machine_action"]

            if action == "voicemail":
                # Play voicemail message then hangup
                logger.info(f"Call {uuid}: Playing voicemail message")
                self.fs.api(f"uuid_broadcast {uuid} {CONFIG['voicemail_file']} aleg")
                time.sleep(10)  # Wait for message to play
                self.fs.api(f"uuid_kill {uuid}")

            elif action == "hangup":
                # Just hangup
                logger.info(f"Call {uuid}: Hanging up (machine detected)")
                self.fs.api(f"uuid_kill {uuid}")

            elif action == "message":
                # Play short beep then hangup
                self.fs.api(f"uuid_broadcast {uuid} tone_stream://%(100,0,800) aleg")
                time.sleep(0.2)
                self.fs.api(f"uuid_kill {uuid}")

        else:
            # Unknown - treat as human
            logger.warning(f"Call {uuid}: Unknown AMD result, treating as human")
            self.fs.api(f"uuid_transfer {uuid} {CONFIG['human_destination']} XML default")


# ============================================================================
# EVENT MONITOR
# ============================================================================

class EventMonitor:
    """Monitor FreeSWITCH events for AMD integration."""

    def __init__(self, fs_con: FreeSwitchConnection, call_handler: CallHandler):
        self.fs = fs_con
        self.handler = call_handler
        self.running = False

    def start(self):
        """Start event monitoring."""
        self.running = True

        # Subscribe to relevant events
        self.fs.subscribe_events("CHANNEL_ANSWER CHANNEL_HANGUP CUSTOM")

        logger.info("Event monitor started")

        while self.running:
            event = self.fs.wait_for_event(1000)
            if event:
                self._handle_event(event)

    def stop(self):
        """Stop event monitoring."""
        self.running = False
        logger.info("Event monitor stopped")

    def _handle_event(self, event: Dict):
        """Handle incoming event."""
        event_name = event.get("Event-Name", "")
        uuid = event.get("Unique-ID", "")

        if event_name == "CHANNEL_ANSWER":
            # Call answered - could start AMD here
            amd_enabled = event.get("variable_amd_enabled", "false")
            if amd_enabled == "true":
                logger.info(f"Call {uuid} answered with AMD enabled")
                # Would trigger AMD processing here

        elif event_name == "CHANNEL_HANGUP":
            logger.debug(f"Call {uuid} hung up")


# ============================================================================
# DIALPLAN HELPER
# ============================================================================

def generate_dialplan():
    """Generate FreeSWITCH dialplan XML for AMD integration."""

    dialplan = f'''<?xml version="1.0" encoding="UTF-8"?>
<!--
    AMD Dialplan for FreeSWITCH
    Place this in /etc/freeswitch/dialplan/default/

    This extension enables AMD for outbound calls with the amd_ prefix.
-->
<include>
  <!-- AMD-enabled outbound calls -->
  <extension name="amd_outbound">
    <condition field="destination_number" expression="^amd_(.+)$">
      <!-- Extract actual destination -->
      <action application="set" data="actual_destination=$1"/>
      <action application="set" data="amd_enabled=true"/>
      <action application="set" data="amd_url={CONFIG['amd_url']}"/>

      <!-- Answer and wait for media -->
      <action application="answer"/>
      <action application="sleep" data="500"/>

      <!-- Start audio streaming to AMD server -->
      <action application="set" data="RECORD_READ_ONLY=true"/>
      <action application="record_session" data="${{base_dir}}/recordings/amd_${{uuid}}.wav"/>

      <!-- Wait for AMD decision (external script handles this) -->
      <action application="sleep" data="{CONFIG['amd_timeout_ms']}"/>

      <!-- Branch based on AMD result (set by external script) -->
      <action application="execute_extension" data="amd_route"/>
    </condition>
  </extension>

  <!-- AMD routing based on result -->
  <extension name="amd_route">
    <condition field="${{amd_result}}" expression="^human$">
      <action application="log" data="INFO AMD detected human"/>
      <action application="transfer" data="{CONFIG['human_destination']}"/>
    </condition>
    <condition field="${{amd_result}}" expression="^machine$">
      <action application="log" data="INFO AMD detected machine"/>
      <action application="playback" data="{CONFIG['voicemail_file']}"/>
      <action application="hangup"/>
    </condition>
    <condition field="${{amd_result}}" expression="^(.*)$" break="always">
      <!-- Default to human if unknown -->
      <action application="log" data="WARNING AMD result unknown, treating as human"/>
      <action application="transfer" data="{CONFIG['human_destination']}"/>
    </condition>
  </extension>
</include>
'''
    return dialplan


def generate_lua_script():
    """Generate FreeSWITCH Lua script for AMD integration."""

    lua_script = f'''--[[
    AMD Integration Script for FreeSWITCH
    Place in /etc/freeswitch/scripts/amd.lua

    Usage in dialplan:
    <action application="lua" data="amd.lua"/>
]]

-- Configuration
local AMD_URL = "{CONFIG['amd_url']}"
local SAMPLE_RATE = {CONFIG['sample_rate']}
local CHUNK_MS = {CONFIG['chunk_ms']}
local TIMEOUT_MS = {CONFIG['amd_timeout_ms']}

-- Load required modules
local socket = require("socket")
local json = require("cjson")

function amd_detect(session)
    local uuid = session:getVariable("uuid")
    freeswitch.consoleLog("INFO", "Starting AMD for " .. uuid .. "\\n")

    -- This is a simplified example
    -- In production, you would:
    -- 1. Connect to WebSocket server
    -- 2. Stream audio frames
    -- 3. Parse JSON response

    -- For now, use HTTP fallback or external script
    local result = "human"  -- Default

    -- Set result variable
    session:setVariable("amd_result", result)
    freeswitch.consoleLog("INFO", "AMD result for " .. uuid .. ": " .. result .. "\\n")

    return result
end

-- Main entry point
function main(session)
    if session:ready() then
        local result = amd_detect(session)

        if result == "human" then
            session:transfer("{CONFIG['human_destination']}", "XML", "default")
        elseif result == "machine" then
            session:streamFile("{CONFIG['voicemail_file']}")
            session:hangup()
        end
    end
end

main(session)
'''
    return lua_script


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='FreeSWITCH AMD Integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
    # Monitor mode
    python freeswitch_amd.py --monitor

    # Originate call with AMD
    python freeswitch_amd.py --originate 15551234567 --caller_id 15559876543

    # Generate dialplan
    python freeswitch_amd.py --generate-dialplan > amd_dialplan.xml

    # Generate Lua script
    python freeswitch_amd.py --generate-lua > amd.lua
        '''
    )

    parser.add_argument('--host', default=CONFIG['fs_host'],
                        help='FreeSWITCH host')
    parser.add_argument('--port', type=int, default=CONFIG['fs_port'],
                        help='FreeSWITCH ESL port')
    parser.add_argument('--password', default=CONFIG['fs_password'],
                        help='FreeSWITCH ESL password')
    parser.add_argument('--amd-url', default=CONFIG['amd_url'],
                        help='AMD WebSocket server URL')

    parser.add_argument('--monitor', action='store_true',
                        help='Run in monitor mode')
    parser.add_argument('--originate', type=str, metavar='NUMBER',
                        help='Originate call to number')
    parser.add_argument('--caller_id', type=str, default='15551234567',
                        help='Caller ID for originated calls')

    parser.add_argument('--generate-dialplan', action='store_true',
                        help='Generate FreeSWITCH dialplan XML')
    parser.add_argument('--generate-lua', action='store_true',
                        help='Generate FreeSWITCH Lua script')

    parser.add_argument('--test-amd', type=str, metavar='AUDIO_FILE',
                        help='Test AMD with audio file')

    args = parser.parse_args()

    # Update config from args
    CONFIG['fs_host'] = args.host
    CONFIG['fs_port'] = args.port
    CONFIG['fs_password'] = args.password
    CONFIG['amd_url'] = args.amd_url

    # Generate dialplan
    if args.generate_dialplan:
        print(generate_dialplan())
        return

    # Generate Lua script
    if args.generate_lua:
        print(generate_lua_script())
        return

    # Test AMD
    if args.test_amd:
        async def test():
            import soundfile as sf
            import librosa

            # Load audio
            audio, sr = librosa.load(args.test_amd, sr=16000, mono=True)

            # Convert to PCM16
            audio_int16 = (audio * 32767).astype('int16')
            audio_bytes = audio_int16.tobytes()

            # Classify
            client = AMDClient(CONFIG['amd_url'])
            result = await client.classify(audio_bytes)

            print(f"AMD Result: {json.dumps(result, indent=2)}")

        asyncio.run(test())
        return

    # Connect to FreeSWITCH
    fs_con = FreeSwitchConnection(
        CONFIG['fs_host'],
        CONFIG['fs_port'],
        CONFIG['fs_password']
    )

    if not fs_con.connect():
        logger.error("Could not connect to FreeSWITCH")
        sys.exit(1)

    amd_client = AMDClient(CONFIG['amd_url'])
    call_handler = CallHandler(fs_con, amd_client)

    try:
        # Originate mode
        if args.originate:
            result = call_handler.originate_with_amd(
                args.originate,
                args.caller_id
            )
            print(f"Call originated: {json.dumps(result, indent=2)}")

        # Monitor mode
        elif args.monitor:
            monitor = EventMonitor(fs_con, call_handler)
            print("Starting event monitor (Ctrl+C to stop)...")
            try:
                monitor.start()
            except KeyboardInterrupt:
                monitor.stop()

        else:
            parser.print_help()

    finally:
        fs_con.disconnect()


if __name__ == "__main__":
    main()
