"""Live data feed options: Network (A), File Replay (B), UDP Sensor (C)"""
import socket
import json
import time
import threading
from typing import Callable, Optional, List, Dict, Any

class LiveDataFeed:
    """Base class for live target data feeds."""
    
    def __init__(self):
        self.running = False
        self.data_buffer: List[Dict[str, Any]] = []
    
    def start(self):
        raise NotImplementedError
    
    def stop(self):
        self.running = False
    
    def get_data(self, t: float) -> List[Dict[str, Any]]:
        """Return list of {tid, x, y, z, vx, vy, vz} for current time."""
        return self.data_buffer


class WebSocketFeed(LiveDataFeed):
    """Option A: WebSocket network stream"""
    
    def __init__(self, host='localhost', port=5000):
        super().__init__()
        self.host = host
        self.port = port
        self.sock = None
        self.client_thread = None
    
    def start(self):
        self.running = True
        self.data_buffer = []
        
        # Try to import websocket, fall back to socket
        try:
            import websocket
            self._start_websocket_client()
        except ImportError:
            print("[LIVE FEED] websocket library not available, using TCP socket fallback")
            self._start_tcp_client()
    
    def _start_websocket_client(self):
        import websocket
        
        def on_message(ws, message):
            data = json.loads(message)
            self.data_buffer.append({
                'tid': data['tid'],
                'x': data['x'], 'y': data['y'], 'z': data['z'],
                'vx': data['vx'], 'vy': data['vy'], 'vz': data['vz'],
                'timestamp': data.get('timestamp', time.time())
            })
        
        self.ws = websocket.WebSocketApp(f"ws://{self.host}:{self.port}",
                                          on_message=on_message)
        self.client_thread = threading.Thread(target=self.ws.run_forever, daemon=True)
        self.client_thread.start()
    
    def _start_tcp_client(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        self.client_thread = threading.Thread(target=self._tcp_reader, daemon=True)
        self.client_thread.start()
    
    def _tcp_reader(self):
        while self.running:
            try:
                data = self.sock.recv(4096)
                if data:
                    parsed = json.loads(data.decode())
                    if isinstance(parsed, list):
                        self.data_buffer.extend(parsed)
                    else:
                        self.data_buffer.append(parsed)
            except:
                break
    
    def stop(self):
        self.running = False
        if hasattr(self, 'ws'):
            self.ws.close()
        if self.sock:
            self.sock.close()


class FileReplayFeed(LiveDataFeed):
    """Option B: CSV/JSON log file replay"""
    
    def __init__(self, filepath: str, realtime: bool = True):
        super().__init__()
        self.filepath = filepath
        self.realtime = realtime  # Match playback speed to real-time
        self.log_data = []
        self.current_index = 0
        self.start_time = None
        self.load_file()
    
    def load_file(self):
        import csv
        
        if filepath.endswith('.csv'):
            with open(filepath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.log_data.append({
                        'tid': int(row['tid']),
                        'x': float(row['x']), 'y': float(row['y']), 'z': float(row['z']),
                        'vx': float(row['vx']), 'vy': float(row['vy']), 'vz': float(row['vz']),
                        'timestamp': float(row.get('t', row.get('timestamp', 0)))
                    })
        elif filepath.endswith('.json'):
            with open(filepath, 'r') as f:
                self.log_data = json.load(f)
        
        print(f"[FILE REPLAY] Loaded {len(self.log_data)} entries from {filepath}")
    
    def get_data(self, t: float) -> List[Dict[str, Any]]:
        """Return data entries closest to simulation time t."""
        if self.realtime:
            # Match real-time: return entries at current sim time t
            entries = [e for e in self.log_data if abs(e['timestamp'] - t) < 0.05]
            return [{'tid': e['tid'], 'x': e['x'], 'y': e['y'], 'z': e['z'],
                     'vx': e.get('vx', 0), 'vy': e.get('vy', 0), 'vz': e.get('vz', 0)}
                    for e in entries]
        else:
            # Sequential: advance through file regardless of sim time
            while self.current_index < len(self.log_data) and \
                  self.log_data[self.current_index]['timestamp'] <= t:
                self.current_index += 1
            
            if self.current_index < len(self.log_data):
                e = self.log_data[self.current_index]
                return [{'tid': e['tid'], 'x': e['x'], 'y': e['y'], 'z': e['z'],
                         'vx': e.get('vx', 0), 'vy': e.get('vy', 0), 'vz': e.get('vz', 0)}]
            return []


class UDPFeed(LiveDataFeed):
    """Option C: UDP sensor integration"""
    
    def __init__(self, host='localhost', port=6000):
        super().__init__()
        self.host = host
        self.port = port
        self.sock = None
    
    def start(self):
        self.running = True
        self.data_buffer = []
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind((self.host, self.port))
        self.sock.settimeout(0.1)
        print(f"[UDP SENSOR] Listening on {self.host}:{self.port}")
        
        self.client_thread = threading.Thread(target=self._udp_reader, daemon=True)
        self.client_thread.start()
    
    def _udp_reader(self):
        while self.running:
            try:
                data, addr = self.sock.recvfrom(4096)
                if data:
                    parsed = self._parse_packet(data)
                    if parsed:
                        self.data_buffer.append(parsed)
            except socket.timeout:
                continue
            except:
                break
    
    def _parse_packet(self, raw_data):
        """Parse binary/JSON packet from sensor."""
        try:
            # Try JSON first
            data = json.loads(raw_data.decode())
            if isinstance(data, list):
                return [{'tid': d['tid'], 'x': d['x'], 'y': d['y'], 'z': d['z'],
                         'vx': d.get('vx', 0), 'vy': d.get('vy', 0), 'vz': d.get('vz', 0)}
                        for d in data]
            else:
                return [{'tid': data['tid'], 'x': data['x'], 'y': data['y'], 'z': data['z'],
                         'vx': data.get('vx', 0), 'vy': data.get('vy', 0), 'vz': data.get('vz', 0)}]
        except:
            # Binary parsing (custom protocol) - implement per sensor spec
            return None
    
    def stop(self):
        self.running = False
        if self.sock:
            self.sock.close()


def create_live_feed(feed_type: str, **kwargs) -> LiveDataFeed:
    """Factory function for creating live data feeds."""
    feed_map = {
        'network': WebSocketFeed,
        'file': FileReplayFeed,
        'udp': UDPFeed,
    }
    
    feed_class = feed_map.get(feed_type.lower())
    if not feed_class:
        raise ValueError(f"Unknown feed type: {feed_type}. Options: network, file, udp")
    
    return feed_class(**kwargs)
