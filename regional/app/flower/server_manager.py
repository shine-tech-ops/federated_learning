"""
Flower Server Manager
"""

import threading
import time
import sys
import os
from loguru import logger
from typing import Dict, Any, Optional
import flwr as fl

# Configure Flower server specific logger
flower_logger = logger.bind(component="FlowerServer")

# Add shared module path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'shared'))

from mnist_model import create_model, get_model_parameters, set_model_parameters


class FlowerServerManager:
    """Flower Server Manager"""
    
    def __init__(self, region_id: str):
        self.region_id = region_id
        self.server_thread: Optional[threading.Thread] = None
        self.server_running = False
        self.current_task = None
        self.model = None
        self.server_config = None
    
    def start_server(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Start Flower server"""
        try:
            if self.server_running:
                flower_logger.warning("Fed Server is already running")
                return self._get_server_info()
            
            # Save task data
            self.current_task = task_data
            
            # Log received task data
            flower_logger.info(f"Received task from central server: {task_data}")
            
            # Create model
            self.model = create_model()
            
            # Configure server
            self.server_config = {
                'host': 'localhost',
                'port': 8080,
                'server_id': f"federated_server_{task_data['task_id']}"
            }
            
            # Start server in background thread
            self.server_thread = threading.Thread(
                target=self._run_server,
                daemon=True
            )
            self.server_thread.start()
            
            # Wait for server to start
            time.sleep(2)
            
            flower_logger.info("Flower server started successfully")
            return self._get_server_info()
            
        except Exception as e:
            flower_logger.error(f"Failed to start Flower server: {e}")
            raise
    
    def _run_server(self):
        """Run Flower server"""
        try:
            # Create strategy
            strategy = fl.server.strategy.FedAvg(
                fraction_fit=1.0,  # 100% clients participate in training
                fraction_evaluate=1.0,  # 100% clients participate in evaluation
                min_fit_clients=1,  # Minimum 1 client
                min_evaluate_clients=1,  # Minimum 1 client
                min_available_clients=1,  # Minimum 1 available client
                evaluate_fn=self._evaluate_fn,
                on_fit_config_fn=self._fit_config_fn,
                on_evaluate_config_fn=self._evaluate_config_fn,
            )
            
            flower_logger.info(f"Starting federated learning server for task: {self.current_task['task_id']}")
            
            # Start server
            self.server_running = True

            fl.server.start_server(
                server_address=f"{self.server_config['host']}:{self.server_config['port']}",
                config=fl.server.ServerConfig(num_rounds=self.current_task['rounds']),
                strategy=strategy
            )
            
            flower_logger.info(f"Federated learning completed for task: {self.current_task['task_id']}")
            
        except Exception as e:
            flower_logger.error(f"Failed to run federated learning server: {e}")
        finally:
            self.server_running = False
    
    def _evaluate_fn(self, server_round: int, parameters, config):
        """Server-side evaluation function with parameter storage"""
        try:
            # Set model parameters
            set_model_parameters(self.model, parameters)
            
            # Store parameters based on strategy
            self._store_parameters(server_round, parameters)
            
            # Server-side evaluation logic
            # Currently returns default values
            loss = 0.0
            accuracy = 0.0
            metrics = {"loss": loss, "accuracy": accuracy}
            
            return loss, metrics
            
        except Exception as e:
            flower_logger.error(f"Server-side evaluation failed: {e}")
            return 0.0, {}
    
    def _fit_config_fn(self, server_round: int):
        """Training configuration function"""
        return {
            "server_round": server_round,
            "local_epochs": 3,
            "learning_rate": 0.01
        }
    
    def _evaluate_config_fn(self, server_round: int):
        """Evaluation configuration function"""
        return {
            "server_round": server_round
        }
    
    def stop_server(self):
        """Stop Flower server"""
        try:
            if self.server_running:
                flower_logger.info("Stopping federated learning server...")
                self.server_running = False
                
                # Wait for server thread to end
                if self.server_thread and self.server_thread.is_alive():
                    self.server_thread.join(timeout=5)
                
                flower_logger.info("Flower server stopped")
            else:
                flower_logger.info("Flower server is not running")
                
        except Exception as e:
            flower_logger.error(f"Failed to stop Flower server: {e}")
    
    def _get_server_info(self) -> Dict[str, Any]:
        """Get server information"""
        return {
            "host": self.server_config['host'],
            "port": self.server_config['port'],
            "server_id": self.server_config['server_id'],
            "running": self.server_running
        }
    
    def is_running(self) -> bool:
        """Check if server is running"""
        return self.server_running
    
    def _store_parameters(self, server_round: int, parameters):
        """Store parameters based on storage strategy"""
        try:
            # Create parameters directory if not exists
            import os
            params_dir = "parameters"
            if not os.path.exists(params_dir):
                os.makedirs(params_dir)
            
            # Storage strategy: checkpoint every 5 rounds + final round
            should_save = False
            filename = ""
            
            # Check if it's a checkpoint round
            if server_round % 5 == 0:
                should_save = True
                filename = f"{params_dir}/checkpoint_round_{server_round:03d}.npz"
                flower_logger.info(f"Saving checkpoint at round {server_round}")
            
            # Check if it's the final round
            if self.current_task and server_round == self.current_task.get('rounds', 0):
                should_save = True
                filename = f"{params_dir}/final_model_round_{server_round:03d}.npz"
                flower_logger.info(f"Saving final model at round {server_round}")
            
            # Save parameters if needed
            if should_save:
                self._save_parameters_to_file(parameters, filename, server_round)
                
        except Exception as e:
            flower_logger.error(f"Failed to store parameters at round {server_round}: {e}")
    
    def _save_parameters_to_file(self, parameters, filepath: str, server_round: int):
        """Save parameters to file with metadata"""
        try:
            import numpy as np
            import json
            from datetime import datetime
            
            # Save parameters
            np.savez(filepath, *parameters)
            
            # Create metadata
            import os
            metadata = {
                "server_round": server_round,
                "task_id": self.current_task.get('task_id', 'unknown') if self.current_task else 'unknown',
                "timestamp": datetime.now().isoformat(),
                "parameter_count": len(parameters),
                "parameter_shapes": [list(p.shape) for p in parameters],
                "total_parameters": sum(p.size for p in parameters),
                "file_size_bytes": os.path.getsize(filepath) if os.path.exists(filepath) else 0
            }
            
            # Save metadata
            metadata_file = filepath.replace('.npz', '_metadata.json')
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            flower_logger.info(f"Parameters saved: {filepath}")
            flower_logger.info(f"Metadata saved: {metadata_file}")
            
        except Exception as e:
            flower_logger.error(f"Failed to save parameters to file {filepath}: {e}")
    
    def save_current_parameters(self, custom_filename: str = None) -> Dict[str, Any]:
        """Manually save current parameters"""
        try:
            if self.model is None:
                return {"error": "No model available"}
            
            # Create parameters directory
            import os
            params_dir = "parameters"
            if not os.path.exists(params_dir):
                os.makedirs(params_dir)
            
            # Generate filename
            if custom_filename:
                filename = f"{params_dir}/{custom_filename}"
            else:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{params_dir}/manual_save_{timestamp}.npz"
            
            # Get current parameters
            current_parameters = get_model_parameters(self.model)
            
            # Save parameters
            self._save_parameters_to_file(current_parameters, filename, 0)
            
            return {
                "success": True,
                "filepath": filename,
                "message": f"Parameters saved to {filename}"
            }
            
        except Exception as e:
            flower_logger.error(f"Failed to save current parameters: {e}")
            return {"error": str(e)}
