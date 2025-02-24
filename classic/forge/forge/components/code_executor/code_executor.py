"""Code execution component for AutoGPT."""
import logging
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class CodeExecutor:
    """Component for executing code in various languages."""
    
    def __init__(self, config):
        """Initialize the code executor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.timeout = config.get("execution_timeout", 30)  # Default 30 seconds
        self.workspace_dir = config.get("workspace_dir", "workspace")
        self.max_output_size = config.get("max_output_size", 4096)
        self.supported_languages = {
            "python": self._execute_python,
            "javascript": self._execute_javascript,
            "node": self._execute_javascript,
            "bash": self._execute_bash,
            "sh": self._execute_bash,
            "powershell": self._execute_powershell,
            "batch": self._execute_batch,
        }
        
        # Ensure workspace exists
        if not os.path.exists(self.workspace_dir):
            os.makedirs(self.workspace_dir)
    
    def execute_code(self, code: str, language: str) -> Dict[str, Union[str, int]]:
        """Execute code in the specified language.
        
        Args:
            code: Code to execute
            language: Programming language
            
        Returns:
            Dictionary with execution results
        """
        language = language.lower().strip()
        
        if language not in self.supported_languages:
            return {
                "success": False,
                "output": f"Unsupported language: {language}. Supported languages: {', '.join(self.supported_languages.keys())}",
                "exit_code": 1
            }
        
        executor = self.supported_languages[language]
        return executor(code)
    
    def _execute_with_timeout(self, func, *args, **kwargs) -> Tuple[bool, str, int]:
        """Execute a function with a timeout.
        
        Args:
            func: Function to execute
            *args: Arguments to pass to func
            **kwargs: Keyword arguments to pass to func
            
        Returns:
            Tuple of (success, output, exit_code)
        """
        result = {"success": False, "output": "", "exit_code": -1}
        completed = threading.Event()
        
        def target():
            try:
                success, output, exit_code = func(*args, **kwargs)
                result["success"] = success
                result["output"] = output
                result["exit_code"] = exit_code
            except Exception as e:
                result["output"] = f"Execution error: {str(e)}"
            finally:
                completed.set()
        
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        
        completed.wait(self.timeout)
        
        if not completed.is_set():
            return False, f"Execution timed out after {self.timeout} seconds", -1
        
        return result["success"], result["output"], result["exit_code"]
    
    def _execute_python(self, code: str) -> Dict[str, Union[str, int]]:
        """Execute Python code.
        
        Args:
            code: Python code to execute
            
        Returns:
            Dictionary with execution results
        """
        with tempfile.NamedTemporaryFile(suffix=".py", delete=False, dir=self.workspace_dir, mode="w") as f:
            f.write(code)
            temp_file = f.name
        
        try:
            success, output, exit_code = self._execute_with_timeout(
                self._run_process, ["python", temp_file]
            )
            
            return {
                "success": success,
                "output": output,
                "exit_code": exit_code
            }
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except Exception:
                pass
    
    def _execute_javascript(self, code: str) -> Dict[str, Union[str, int]]:
        """Execute JavaScript code using Node.js.
        
        Args:
            code: JavaScript code to execute
            
        Returns:
            Dictionary with execution results
        """
        with tempfile.NamedTemporaryFile(suffix=".js", delete=False, dir=self.workspace_dir, mode="w") as f:
            f.write(code)
            temp_file = f.name
        
        try:
            success, output, exit_code = self._execute_with_timeout(
                self._run_process, ["node", temp_file]
            )
            
            return {
                "success": success,
                "output": output,
                "exit_code": exit_code
            }
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except Exception:
                pass
    
    def _execute_bash(self, code: str) -> Dict[str, Union[str, int]]:
        """Execute Bash code.
        
        Args:
            code: Bash code to execute
            
        Returns:
            Dictionary with execution results
        """
        with tempfile.NamedTemporaryFile(suffix=".sh", delete=False, dir=self.workspace_dir, mode="w") as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Make the script executable
            os.chmod(temp_file, 0o755)
            
            success, output, exit_code = self._execute_with_timeout(
                self._run_process, ["bash", temp_file]
            )
            
            return {
                "success": success,
                "output": output,
                "exit_code": exit_code
            }
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except Exception:
                pass
    
    def _execute_powershell(self, code: str) -> Dict[str, Union[str, int]]:
        """Execute PowerShell code.
        
        Args:
            code: PowerShell code to execute
            
        Returns:
            Dictionary with execution results
        """
        with tempfile.NamedTemporaryFile(suffix=".ps1", delete=False, dir=self.workspace_dir, mode="w") as f:
            f.write(code)
            temp_file = f.name
        
        try:
            success, output, exit_code = self._execute_with_timeout(
                self._run_process, ["powershell", "-ExecutionPolicy", "Bypass", "-File", temp_file]
            )
            
            return {
                "success": success,
                "output": output,
                "exit_code": exit_code
            }
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except Exception:
                pass
    
    def _execute_batch(self, code: str) -> Dict[str, Union[str, int]]:
        """Execute Windows Batch code.
        
        Args:
            code: Batch code to execute
            
        Returns:
            Dictionary with execution results
        """
        with tempfile.NamedTemporaryFile(suffix=".bat", delete=False, dir=self.workspace_dir, mode="w") as f:
            f.write(code)
            temp_file = f.name
        
        try:
            success, output, exit_code = self._execute_with_timeout(
                self._run_process, [temp_file]
            )
            
            return {
                "success": success,
                "output": output,
                "exit_code": exit_code
            }
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_file)
            except Exception:
                pass
    
    def _run_process(self, command: List[str]) -> Tuple[bool, str, int]:
        """Run a subprocess with the given command.
        
        Args:
            command: Command to run
            
        Returns:
            Tuple of (success, output, exit_code)
        """
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                cwd=self.workspace_dir
            )
            
            stdout, stderr = process.communicate()
            
            # Combine stdout and stderr, truncate if too long
            output = stdout + ("\n" + stderr if stderr else "")
            if len(output) > self.max_output_size:
                output = output[:self.max_output_size] + f"\n... (output truncated, exceeded {self.max_output_size} characters)"
            
            return process.returncode == 0, output, process.returncode
        except Exception as e:
            return False, f"Error executing command: {str(e)}", 1
