import logging
import os
import random
import shlex
import string
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterator, Literal, Optional, Union

try:
    import docker
    from docker.errors import DockerException, ImageNotFound, NotFound
    from docker.models.containers import Container as DockerContainer
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

from pydantic import BaseModel, Field

from forge.agent.components import ConfigurableComponent
from forge.agent.protocols import CommandProvider
from forge.command import Command, command
from forge.file_storage import FileStorage
from forge.models.json_schema import JSONSchema
from forge.utils.exceptions import (
    CommandExecutionError,
    InvalidArgumentError,
    OperationNotAllowedError,
)

logger = logging.getLogger(__name__)


def we_are_running_in_a_docker_container() -> bool:
    """Check if we are running in a Docker container

    Returns:
        bool: True if we are running in a Docker container, False otherwise
    """
    return os.path.exists("/.dockerenv")


def is_docker_available() -> bool:
    """Check if Docker is available and supports Linux containers

    Returns:
        bool: True if Docker is available and supports Linux containers, False otherwise
    """
    if not DOCKER_AVAILABLE:
        logger.debug("Docker package not available")
        return False
        
    try:
        client = docker.from_env()
        docker_info = client.info()
        return docker_info.get("OSType", "") == "linux"
    except Exception as e:
        logger.debug(f"Docker not available: {e}")
        return False


class CodeExecutionError(CommandExecutionError):
    """The operation (an attempt to run arbitrary code) returned an error"""


class CodeExecutorConfiguration(BaseModel):
    execute_local_commands: bool = False
    """Enable shell command execution"""
    allow_native_execution: bool = True
    """Allow executing Python code outside of Docker when Docker is unavailable"""
    shell_command_control: Literal["allowlist", "denylist"] = "allowlist"
    """Controls which list is used"""
    shell_allowlist: list[str] = Field(default_factory=list)
    """List of allowed shell commands"""
    shell_denylist: list[str] = Field(default_factory=list)
    """List of prohibited shell commands"""
    docker_container_name: str = "agent_sandbox"
    """Name of the Docker container used for code execution"""
    docker_timeout: int = 60
    """Timeout for Docker commands in seconds"""
    execution_timeout: int = 30
    """Timeout for code execution in seconds (when not using Docker)"""


class CodeExecutorComponent(
    CommandProvider, ConfigurableComponent[CodeExecutorConfiguration]
):
    """Provides commands to execute Python code and shell commands."""

    config_class = CodeExecutorConfiguration

    def __init__(
        self,
        workspace: FileStorage,
        config: Optional[CodeExecutorConfiguration] = None,
    ):
        ConfigurableComponent.__init__(self, config)
        self.workspace = workspace

        # Change container name if it's empty or default to prevent different agents
        # from using the same container
        default_container_name = self.config.model_fields[
            "docker_container_name"
        ].default
        if (
            not self.config.docker_container_name
            or self.config.docker_container_name == default_container_name
        ):
            random_suffix = "".join(random.choices(string.ascii_lowercase, k=8))
            self.config.docker_container_name = (
                f"{default_container_name}_{random_suffix}"
            )

        self.docker_available = False
        self.running_in_docker = we_are_running_in_a_docker_container()
        
        if not self.running_in_docker:
            self.docker_available = is_docker_available()

        if not self.docker_available and not self.running_in_docker:
            if self.config.allow_native_execution:
                logger.info(
                    "Docker is not available. Code execution will be performed "
                    "directly in the current environment, which may pose security risks."
                )
            else:
                logger.warning(
                    "Docker is not available and native execution is disabled. "
                    "Code execution commands will not be available."
                )

        if not self.config.execute_local_commands:
            logger.info(
                "Local shell commands are disabled. To enable them,"
                " set EXECUTE_LOCAL_COMMANDS to 'True' in your config file."
            )

    def get_commands(self) -> Iterator[Command]:
        if self.running_in_docker or self.docker_available or self.config.allow_native_execution:
            yield self.execute_python_code
            yield self.execute_python_file

        if self.config.execute_local_commands:
            yield self.execute_shell
            yield self.execute_shell_popen

    @command(
        ["execute_python_code"],
        "Executes the given Python code inside a single-use Docker container"
        " with access to your workspace folder (if Docker is available)"
        " or directly in the current environment",
        {
            "code": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The Python code to run",
                required=True,
            ),
        },
    )
    async def execute_python_code(self, code: str) -> str:
        """
        Create and execute a Python file in a Docker container (if available)
        or directly in the current environment, and return the output.

        If the code generates any data that needs to be captured,
        use a print statement.

        Args:
            code (str): The Python code to run.

        Returns:
            str: The output captured from the code execution.
        """
        # Generate a random filename to prevent conflicts
        temp_path = ""
        while True:
            temp_path = f"temp{self._generate_random_string()}.py"
            if not self.workspace.exists(temp_path):
                break
        await self.workspace.write_file(temp_path, code)

        try:
            return self.execute_python_file(temp_path)
        except Exception as e:
            raise CommandExecutionError(*e.args)
        finally:
            self.workspace.delete_file(temp_path)

    @command(
        ["execute_python_file"],
        "Execute an existing Python file inside a single-use Docker container"
        " with access to your workspace folder (if Docker is available)"
        " or directly in the current environment",
        {
            "filename": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The name of the file to execute",
                required=True,
            ),
            "args": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                description="The (command line) arguments to pass to the script",
                required=False,
                items=JSONSchema(type=JSONSchema.Type.STRING),
            ),
        },
    )
    def execute_python_file(self, filename: Union[str, Path], args: list[str] = []) -> str:
        """Execute a Python file and return the output

        Args:
            filename (Path): The name of the file to execute
            args (list, optional): The arguments with which to run the python script

        Returns:
            str: The output of the file
        """
        logger.info(f"Executing python file '{filename}'")

        if not str(filename).endswith(".py"):
            raise InvalidArgumentError("Invalid file type. Only .py files are allowed.")

        file_path = self.workspace.get_path(filename)
        if not self.workspace.exists(file_path):
            # Mimic the response that you get from the command line to make it
            # intuitively understandable for the LLM
            raise FileNotFoundError(
                f"python: can't open file '{filename}': "
                f"[Errno 2] No such file or directory"
            )

        # Check execution options in order of preference
        if self.running_in_docker:
            logger.debug(
                "App is running in a Docker container; "
                f"executing {file_path} directly..."
            )
            with self.workspace.mount() as local_path:
                result = subprocess.run(
                    ["python", "-B", str(file_path.relative_to(self.workspace.root))]
                    + args,
                    capture_output=True,
                    encoding="utf8",
                    cwd=str(local_path),
                    timeout=self.config.execution_timeout,
                )
                if result.returncode == 0:
                    return result.stdout
                else:
                    raise CodeExecutionError(result.stderr)
        elif self.docker_available:
            logger.debug("Docker is available; using container for execution...")
            return self._run_python_code_in_docker(file_path, args)
        elif self.config.allow_native_execution:
            logger.debug(
                "Docker not available but native execution allowed; "
                f"executing {file_path} directly..."
            )
            try:
                # Create a temporary directory for execution to contain any file operations
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Copy the workspace files to the temp directory
                    import shutil
                    temp_workspace = Path(temp_dir)
                    for item in self.workspace.list_files():
                        dest_path = temp_workspace / item.relative_to(self.workspace.root)
                        if not dest_path.parent.exists():
                            dest_path.parent.mkdir(parents=True)
                        if item.is_file():
                            shutil.copy2(item, dest_path)
                    
                    # Get the relative path from workspace root
                    rel_path = file_path.relative_to(self.workspace.root)
                    exec_path = temp_workspace / rel_path
                    
                    # Execute in the temp directory
                    result = subprocess.run(
                        [sys.executable, "-B", str(exec_path)] + args,
                        capture_output=True,
                        encoding="utf8",
                        cwd=str(temp_workspace),
                        timeout=self.config.execution_timeout,
                    )
                    
                    # Copy any created/modified files back to the workspace
                    for item in Path(temp_dir).rglob("*"):
                        if item.is_file():
                            rel_item = item.relative_to(temp_workspace)
                            workspace_path = self.workspace.root / rel_item
                            if not workspace_path.parent.exists():
                                workspace_path.parent.mkdir(parents=True, exist_ok=True)
                            # Only copy if the file is new or modified
                            if not workspace_path.exists() or not self._files_are_identical(item, workspace_path):
                                shutil.copy2(item, workspace_path)
                    
                    if result.returncode == 0:
                        return result.stdout
                    else:
                        raise CodeExecutionError(result.stderr)
            except subprocess.TimeoutExpired:
                return "Error: Code execution exceeded the time limit."
            except Exception as e:
                raise CodeExecutionError(f"Native execution failed: {str(e)}")
        else:
            raise CommandExecutionError(
                "Docker is not available and native execution is disabled. "
                "Cannot execute Python code."
            )

    def _files_are_identical(self, file1: Path, file2: Path) -> bool:
        """Check if two files have identical content"""
        if not file1.exists() or not file2.exists():
            return False
        
        if file1.stat().st_size != file2.stat().st_size:
            return False
        
        # For small files, compare content directly
        if file1.stat().st_size < 1024 * 1024:  # 1MB
            return file1.read_bytes() == file2.read_bytes()
        
        # For larger files, compare in chunks
        with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
            while True:
                chunk1 = f1.read(4096)
                chunk2 = f2.read(4096)
                if chunk1 != chunk2:
                    return False
                if not chunk1:  # EOF
                    return True

    def validate_command(self, command_line: str) -> tuple[bool, bool]:
        """Check whether a command is allowed and whether it may be executed in a shell.

        If shell command control is enabled, we disallow executing in a shell, because
        otherwise the model could circumvent the command filter using shell features.

        Args:
            command_line (str): The command line to validate

        Returns:
            bool: True if the command is allowed, False otherwise
            bool: True if the command may be executed in a shell, False otherwise
        """
        if not command_line:
            return False, False

        command_name = shlex.split(command_line)[0]

        if self.config.shell_command_control == "allowlist":
            return command_name in self.config.shell_allowlist, False
        elif self.config.shell_command_control == "denylist":
            return command_name not in self.config.shell_denylist, False
        else:
            return True, True

    @command(
        ["execute_shell"],
        "Execute a Shell Command, non-interactive commands only",
        {
            "command_line": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The command line to execute",
                required=True,
            )
        },
    )
    def execute_shell(self, command_line: str) -> str:
        """Execute a shell command and return the output

        Args:
            command_line (str): The command line to execute

        Returns:
            str: The output of the command
        """
        allow_execute, allow_shell = self.validate_command(command_line)
        if not allow_execute:
            logger.info(f"Command '{command_line}' not allowed")
            raise OperationNotAllowedError("This shell command is not allowed.")

        current_dir = Path.cwd()
        # Change dir into workspace if necessary
        if not current_dir.is_relative_to(self.workspace.root):
            os.chdir(self.workspace.root)

        logger.info(
            f"Executing command '{command_line}' in working directory '{os.getcwd()}'"
        )

        try:
            result = subprocess.run(
                command_line if allow_shell else shlex.split(command_line),
                capture_output=True,
                shell=allow_shell,
                timeout=self.config.execution_timeout,
            )
            output = f"STDOUT:\n{result.stdout.decode()}\nSTDERR:\n{result.stderr.decode()}"
        except subprocess.TimeoutExpired:
            output = "Error: Command execution exceeded the time limit."
        finally:
            # Change back to whatever the prior working dir was
            os.chdir(current_dir)

        return output

    @command(
        ["execute_shell_popen"],
        "Execute a Shell Command, non-interactive commands only",
        {
            "command_line": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The command line to execute",
                required=True,
            )
        },
    )
    def execute_shell_popen(self, command_line: str) -> str:
        """Execute a shell command with Popen and returns an english description
        of the event and the process id

        Args:
            command_line (str): The command line to execute

        Returns:
            str: Description of the fact that the process started and its id
        """
        allow_execute, allow_shell = self.validate_command(command_line)
        if not allow_execute:
            logger.info(f"Command '{command_line}' not allowed")
            raise OperationNotAllowedError("This shell command is not allowed.")

        current_dir = Path.cwd()
        # Change dir into workspace if necessary
        if not current_dir.is_relative_to(self.workspace.root):
            os.chdir(self.workspace.root)

        logger.info(
            f"Executing command '{command_line}' in working directory '{os.getcwd()}'"
        )

        do_not_show_output = subprocess.DEVNULL
        try:
            process = subprocess.Popen(
                command_line if allow_shell else shlex.split(command_line),
                shell=allow_shell,
                stdout=do_not_show_output,
                stderr=do_not_show_output,
            )
        finally:
            # Change back to whatever the prior working dir was
            os.chdir(current_dir)

        return f"Subprocess started with PID:'{str(process.pid)}'"

    def _run_python_code_in_docker(self, filename: Union[str, Path], args: list[str]) -> str:
        """Run a Python script in a Docker container"""
        if not DOCKER_AVAILABLE:
            raise CommandExecutionError("Docker package is not installed")
            
        file_path = self.workspace.get_path(filename)
        try:
            client = docker.from_env()
            image_name = "python:3-alpine"
            container_is_fresh = False
            container_name = self.config.docker_container_name
            with self.workspace.mount() as local_path:
                try:
                    container: DockerContainer = client.containers.get(
                        container_name
                    )  # type: ignore
                except NotFound:
                    try:
                        client.images.get(image_name)
                        logger.debug(f"Image '{image_name}' found locally")
                    except ImageNotFound:
                        logger.info(
                            f"Image '{image_name}' not found locally,"
                            " pulling from Docker Hub..."
                        )
                        # Use the low-level API to stream the pull response
                        low_level_client = docker.APIClient()
                        for line in low_level_client.pull(
                            image_name, stream=True, decode=True
                        ):
                            # Print the status and progress, if available
                            status = line.get("status")
                            progress = line.get("progress")
                            if status and progress:
                                logger.info(f"{status}: {progress}")
                            elif status:
                                logger.info(status)

                    logger.debug(f"Creating new {image_name} container...")
                    container: DockerContainer = client.containers.run(
                        image_name,
                        ["sleep", str(self.config.docker_timeout)],  # Configurable timeout
                        volumes={
                            str(local_path.resolve()): {
                                "bind": "/workspace",
                                "mode": "rw",
                            }
                        },
                        working_dir="/workspace",
                        stderr=True,
                        stdout=True,
                        detach=True,
                        name=container_name,
                    )  # type: ignore
                    container_is_fresh = True

                if not container.status == "running":
                    container.start()
                elif not container_is_fresh:
                    container.restart()

                logger.debug(f"Running {file_path} in container {container.name}...")

                exec_result = container.exec_run(
                    [
                        "python",
                        "-B",
                        file_path.relative_to(self.workspace.root).as_posix(),
                    ]
                    + args,
                    stderr=True,
                    stdout=True,
                    demux=True,  # Split stdout and stderr
                )

                # Handle stdout and stderr separately
                stdout, stderr = exec_result.output
                stdout = stdout.decode("utf-8") if stdout else ""
                stderr = stderr.decode("utf-8") if stderr else ""

                if exec_result.exit_code != 0:
                    raise CodeExecutionError(stderr or "Unknown error during execution")

                return stdout

        except DockerException as e:
            logger.warning(
                "Could not run the script in a container. "
                "If you haven't already, please install Docker: "
                "https://docs.docker.com/get-docker/"
            )
            raise CommandExecutionError(f"Could not run the script in a container: {e}")

    def _generate_random_string(self, length: int = 8):
        # Create a string of all letters and digits
        characters = string.ascii_letters + string.digits
        # Use random.choices to generate a random string
        random_string = "".join(random.choices(characters, k=length))
        return random_string
