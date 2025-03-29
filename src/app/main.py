# src/app/main.py
"""
ComfyUI Server Main Entry Point
Handles server initialization, configuration, and execution
"""

import os
import asyncio
import threading
import gc
import importlib.util
import itertools
import logging
import shutil
import time
from typing import Optional, Tuple

# Core ComfyUI imports
import comfy.options
import comfy.utils
import folder_paths
from comfy.cli_args import args
from app.logger import setup_logger
import utils.extra_config

# Initialize command line argument parsing
comfy.options.enable_args_parsing()

class ComfyUIServer:
    """
    Core server configuration and initialization handler
    Manages environment setup, paths, and pre-startup scripts
    """
    def __init__(self):
        """Initialize server environment and configurations"""
        self._setup_environment()
        setup_logger(log_level=args.verbose, use_stdout=args.log_stdout)
        self._apply_custom_paths()
        self._execute_prestartup_scripts()

    def _setup_environment(self):
        """Configure required environment variables"""
        os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
        os.environ['DO_NOT_TRACK'] = '1'
        
        if args.deterministic:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    def _apply_custom_paths(self):
        """Configure all model and directory paths"""
        config_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), 
            "extra_model_paths.yaml"
        )
        if os.path.isfile(config_path):
            utils.extra_config.load_extra_path_config(config_path)

        if args.extra_model_paths_config:
            for config_path in itertools.chain(*args.extra_model_paths_config):
                utils.extra_config.load_extra_path_config(config_path)

        if args.output_directory:
            self._configure_output_directory(args.output_directory)

        if args.input_directory:
            folder_paths.set_input_directory(os.path.abspath(args.input_directory))
        if args.user_directory:
            folder_paths.set_user_directory(os.path.abspath(args.user_directory))

    def _configure_output_directory(self, output_dir):
        """Initialize output directory structure"""
        output_dir = os.path.abspath(output_dir)
        logging.info(f"Configuring output directory: {output_dir}")
        folder_paths.set_output_directory(output_dir)
        
        model_folders = {
            "checkpoints": "checkpoints",
            "clip": "clip", 
            "vae": "vae",
            "diffusion_models": "diffusion_models",
            "loras": "loras"
        }
        
        for name, folder in model_folders.items():
            full_path = os.path.join(output_dir, folder)
            folder_paths.add_model_folder_path(name, full_path)

    def _execute_prestartup_scripts(self):
        """Execute custom node initialization scripts"""
        if args.disable_all_custom_nodes:
            return

        node_paths = folder_paths.get_folder_paths("custom_nodes")
        startup_times = []
        
        for custom_node_path in node_paths:
            for possible_module in os.listdir(custom_node_path):
                module_path = os.path.join(custom_node_path, possible_module)
                
                if (os.path.isfile(module_path) or \
                   module_path.endswith(".disabled") or \
                   module_path == "__pycache__":
                    continue
                
                script_path = os.path.join(module_path, "prestartup_script.py")
                if os.path.exists(script_path):
                    start_time = time.perf_counter()
                    success = self._execute_script(script_path)
                    startup_times.append((
                        time.perf_counter() - start_time,
                        module_path,
                        success
                    ))
        
        if startup_times:
            self._log_startup_times(startup_times)

    def _execute_script(self, script_path):
        """Execute individual startup script"""
        try:
            module_name = os.path.splitext(script_path)[0]
            spec = importlib.util.spec_from_file_location(module_name, script_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return True
        except Exception as e:
            logging.error(f"Startup script execution failed: {script_path}: {e}")
            return False

    def _log_startup_times(self, times):
        """Log custom node initialization times"""
        logging.info("Custom node initialization times:")
        for time_taken, path, success in sorted(times):
            status = "" if success else " (FAILED)"
            logging.info(f"{time_taken:6.1f} seconds{status}: {path}")

class PromptWorker:
    """
    Handles prompt execution queue processing
    Manages prompt execution and resource cleanup
    """
    def __init__(self, queue, server_instance):
        self.queue = queue
        self.server = server_instance
        self.executor = execution.PromptExecutor(server_instance, lru_size=args.cache_lru)
        self.last_gc_collect = 0
        self.need_gc = False
        self.gc_collect_interval = 10.0

    def run(self):
        """Main processing loop for prompt queue"""
        while True:
            timeout = self._calculate_timeout()
            queue_item = self.queue.get(timeout=timeout)
            
            if queue_item is not None:
                self._process_queue_item(queue_item)
            
            self._handle_queue_flags()

    def _calculate_timeout(self):
        """Determine next timeout based on GC needs"""
        if self.need_gc:
            return max(self.gc_collect_interval - (time.perf_counter() - self.last_gc_collect), 0.0)
        return 1000.0

    def _process_queue_item(self, queue_item):
        """Process individual queue item"""
        item, item_id = queue_item
        execution_start_time = time.perf_counter()
        prompt_id = item[1]
        self.server.last_prompt_id = prompt_id

        self.executor.execute(item[2], prompt_id, item[3], item[4])
        self.need_gc = True
        
        self.queue.task_done(
            item_id,
            self.executor.history_result,
            status=execution.PromptQueue.ExecutionStatus(
                status_str='success' if self.executor.success else 'error',
                completed=self.executor.success,
                messages=self.executor.status_messages)
        )
        
        if self.server.client_id is not None:
            self.server.send_sync("executing", 
                               {"node": None, "prompt_id": prompt_id}, 
                               self.server.client_id)

        execution_time = time.perf_counter() - execution_start_time
        logging.info(f"Prompt execution time: {execution_time:.2f} seconds")

    def _handle_queue_flags(self):
        """Handle queue flags for memory management"""
        flags = self.queue.get_flags()
        free_memory = flags.get("free_memory", False)

        if flags.get("unload_models", free_memory):
            comfy.model_management.unload_all_models()
            self.need_gc = True
            self.last_gc_collect = 0

        if free_memory:
            self.executor.reset()
            self.need_gc = True
            self.last_gc_collect = 0

        if self.need_gc:
            current_time = time.perf_counter()
            if (current_time - self.last_gc_collect) > self.gc_collect_interval:
                self._perform_garbage_collection()
                self.last_gc_collect = current_time
                self.need_gc = False

    def _perform_garbage_collection(self):
        """Execute garbage collection and cache cleanup"""
        gc.collect()
        comfy.model_management.soft_empty_cache()

class ComfyUIStarter:
    """
    Server startup and lifecycle management
    Handles server initialization and execution
    """
    @staticmethod
    def start(asyncio_loop=None):
        """
        Initialize and start server components
        Returns event loop, server instance, and start function
        """
        ComfyUIStarter._setup_temp_directory()
        ComfyUIStarter._handle_windows_updates()
        
        if not asyncio_loop:
            asyncio_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(asyncio_loop)
            
        prompt_server = server.PromptServer(asyncio_loop)
        queue = execution.PromptQueue(prompt_server)

        ComfyUIStarter._initialize_nodes()
        ComfyUIStarter._check_cuda_malloc()
        ComfyUIStarter._setup_server_routes(prompt_server)
        ComfyUIStarter._start_prompt_worker(queue, prompt_server)

        if args.quick_test_for_ci:
            exit(0)

        return asyncio_loop, prompt_server, ComfyUIStarter._create_start_function(prompt_server)

    @staticmethod
    def _setup_temp_directory():
        """Configure temporary directory"""
        if args.temp_directory:
            temp_dir = os.path.join(os.path.abspath(args.temp_directory), "temp")
            logging.info(f"Setting temp directory: {temp_dir}")
            folder_paths.set_temp_directory(temp_dir)
        ComfyUIStarter._cleanup_temp()

    @staticmethod
    def _cleanup_temp():
        """Clean up temporary directory contents"""
        temp_dir = folder_paths.get_temp_directory()
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)

    @staticmethod
    def _handle_windows_updates():
        """Handle Windows-specific updates"""
        if args.windows_standalone_build:
            try:
                import new_updater
                new_updater.update_windows_updater()
            except Exception as e:
                logging.warning(f"Windows updater error: {e}")

    @staticmethod
    def _initialize_nodes():
        """Initialize all node types"""
        nodes.init_extra_nodes(init_custom_nodes=not args.disable_all_custom_nodes)

    @staticmethod
    def _check_cuda_malloc():
        """Verify CUDA malloc compatibility"""
        device = comfy.model_management.get_torch_device()
        device_name = comfy.model_management.get_torch_device_name(device)
        
        if "cudaMallocAsync" in device_name:
            for b in cuda_malloc.blacklist:
                if b in device_name:
                    logging.warning(
                        "CUDA malloc may not be supported on this device. "
                        "Use --disable-cuda-malloc if encountering CUDA errors"
                    )
                    break

    @staticmethod
    def _setup_server_routes(prompt_server):
        """Configure server routes and hooks"""
        prompt_server.add_routes()
        ComfyUIStarter._hijack_progress(prompt_server)

    @staticmethod
    def _hijack_progress(server_instance):
        """Configure progress reporting system"""
        def hook(value, total, preview_image):
            comfy.model_management.throw_exception_if_processing_interrupted()
            progress = {
                "value": value,
                "max": total,
                "prompt_id": server_instance.last_prompt_id,
                "node": server_instance.last_node_id
            }
            server_instance.send_sync("progress", progress, server_instance.client_id)
            if preview_image is not None:
                server_instance.send_sync(
                    BinaryEventTypes.UNENCODED_PREVIEW_IMAGE,
                    preview_image,
                    server_instance.client_id
                )
        comfy.utils.set_progress_bar_global_hook(hook)

    @staticmethod
    def _start_prompt_worker(queue, prompt_server):
        """Initialize prompt worker thread"""
        threading.Thread(
            target=PromptWorker(queue, prompt_server).run,
            daemon=True
        ).start()

    @staticmethod
    def _create_start_function(prompt_server):
        """Create server start function with optional auto-launch"""
        call_on_start = None
        if args.auto_launch:
            def startup_server(scheme, address, port):
                import webbrowser
                if os.name == 'nt' and address == '0.0.0.0':
                    address = '127.0.0.1'
                if ':' in address:
                    address = f"[{address}]"
                webbrowser.open(f"{scheme}://{address}:{port}")
            call_on_start = startup_server

        async def start_all():
            await prompt_server.setup()
            await ComfyUIStarter._run_server(
                prompt_server,
                address=args.listen,
                port=args.port,
                verbose=not args.dont_print_server,
                call_on_start=call_on_start
            )
        return start_all

    @staticmethod
    async def _run_server(server_instance, address='', port=8188, verbose=True, call_on_start=None):
        """Execute server run loop"""
        addresses = [(addr, port) for addr in address.split(",")]
        await asyncio.gather(
            server_instance.start_multi_address(addresses, call_on_start, verbose),
            server_instance.publish_loop()
        )

def main():
    """Primary execution entry point"""
    logging.info(f"Initializing ComfyUI version {comfyui_version.__version__}")
    
    # Initialize server environment
    ComfyUIServer()
    
    # Start server components
    event_loop, _, start_func = ComfyUIStarter.start()
    
    try:
        logging.info("Starting server execution")
        event_loop.run_until_complete(start_func())
    except KeyboardInterrupt:
        logging.info("Server shutdown requested")
    finally:
        ComfyUIStarter._cleanup_temp()

if __name__ == "__main__":
    # Configure xformers logging filter
    if os.name == "nt":
        logging.getLogger("xformers").addFilter(
            lambda record: 'A matching Triton is not available' not in record.getMessage()
        )
    
    # Configure device visibility
    if args.cuda_device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)
        os.environ['HIP_VISIBLE_DEVICES'] = str(args.cuda_device)
        logging.info(f"CUDA device configured: {args.cuda_device}")
    
    if args.oneapi_device_selector is not None:
        os.environ['ONEAPI_DEVICE_SELECTOR'] = args.oneapi_device_selector
        logging.info(f"OneAPI device configured: {args.oneapi_device_selector}")
    
    # Initialize CUDA memory management
    import cuda_malloc
    
    # Apply Windows-specific fixes
    if args.windows_standalone_build:
        try:
            from fix_torch import fix_pytorch_libomp
            fix_pytorch_libomp()
        except Exception as e:
            logging.warning(f"Torch library fix failed: {e}")
    
    # Begin execution
    main()
