# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
A Ray logger will receive logging info from different processes.
"""
import numbers
import json
import os
from datetime import datetime
from typing import Dict, Any, Optional


def concat_dict_to_str(dict: Dict, step):
    output = [f'step:{step}']
    for k, v in dict.items():
        if isinstance(v, numbers.Number):
            output.append(f'{k}:{v:.3f}')
    output_str = ' - '.join(output)
    return output_str


# class LocalLogger:

#     def __init__(self, remote_logger=None, enable_wandb=False, print_to_console=False):
#         self.print_to_console = print_to_console
#         if print_to_console:
#             print('Using LocalLogger is deprecated. The constructor API will change ')

#     def flush(self):
#         pass

#     def log(self, data, step):
#         if self.print_to_console:
#             print(concat_dict_to_str(data, step=step), flush=True)

class LocalLogger:
    """A logger that writes data to a local file and optionally prints to console."""
    
    def __init__(self, 
                 remote_logger=None, 
                 enable_wandb=False, 
                 print_to_console=False,
                 log_dir: str = "logs",
                 filename_prefix: str = "run"):
        """
        Initialize the LocalLogger.
        
        Args:
            remote_logger: Legacy parameter (not used)
            enable_wandb: Legacy parameter (not used)
            print_to_console: Whether to print logs to console
            log_dir: Directory where log files will be stored
            filename_prefix: Prefix for the log filename
        """
        self.print_to_console = print_to_console
        if print_to_console:
            print('Using LocalLogger is deprecated. The constructor API will change')
        
        # Create logs directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        
        # Generate unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_prefix}_{timestamp}.log"
        self.log_path = os.path.join(log_dir, filename)
        
        # Initialize the log file with a header
        with open(self.log_path, 'w') as f:
            f.write(f"# Log started at {datetime.now().isoformat()}\n")
            f.write("# Format: {timestamp}\t{step}\t{data_json}\n")
    
    def flush(self):
        """Implement flush method for compatibility."""
        pass
    
    def log(self, data: Dict[str, Any], step: int) -> None:
        """
        Log data to file and optionally console.
        
        Args:
            data: Dictionary containing metrics/data to log
            step: Current step number
        """
        # Console output
        if self.print_to_console:
            print(concat_dict_to_str(data, step=step), flush=True)
        
        # File output
        timestamp = datetime.now().isoformat()
        data_str = json.dumps(data)
        log_line = f"{timestamp}\t{step}\t{data_str}\n"
        
        try:
            with open(self.log_path, 'a') as f:
                f.write(log_line)
        except IOError as e:
            if self.print_to_console:
                print(f"Error writing to log file: {e}")
    
    def get_log_path(self) -> str:
        """Return the path to the log file."""
        return self.log_path