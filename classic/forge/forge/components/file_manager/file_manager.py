import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterator, Optional, List, Union

from pydantic import BaseModel, ConfigDict, Field

from forge.agent import BaseAgentSettings
from forge.agent.components import ConfigurableComponent
from forge.agent.protocols import CommandProvider, DirectiveProvider
from forge.command import Command, command
from forge.file_storage.base import FileStorage
from forge.models.json_schema import JSONSchema
from forge.utils.file_operations import decode_textual_file

logger = logging.getLogger(__name__)


class FileManagerConfiguration(BaseModel):
    storage_path: str
    """Path to agent files, e.g. state"""
    workspace_path: str
    """Path to files that agent has access to"""
    max_file_size: int = Field(default=10 * 1024 * 1024)  # 10MB
    """Maximum file size in bytes"""
    allowed_extensions: List[str] = Field(default=[
        ".txt", ".md", ".json", ".yaml", ".yml", ".csv", ".py", ".js", 
        ".html", ".css", ".jsx", ".tsx", ".ts", ".c", ".cpp", ".h", 
        ".java", ".go", ".rs", ".php", ".rb", ".pl", ".sh"
    ])
    """List of allowed file extensions"""
    backup_enabled: bool = Field(default=True)
    """Enable file backups before overwriting"""
    backup_dir: str = Field(default="backups")
    """Directory for file backups relative to workspace"""

    model_config = ConfigDict(
        # Prevent mutation of the configuration
        # as this wouldn't be reflected in the file storage
        frozen=False
    )


class FileManagerComponent(
    DirectiveProvider, CommandProvider, ConfigurableComponent[FileManagerConfiguration]
):
    """
    Adds general file manager (e.g. Agent state),
    workspace manager (e.g. Agent output files) support and
    commands to perform operations on files and folders.
    """

    config_class = FileManagerConfiguration

    STATE_FILE = "state.json"
    """The name of the file where the agent's state is stored."""

    def __init__(
        self,
        file_storage: FileStorage,
        agent_state: BaseAgentSettings,
        config: Optional[FileManagerConfiguration] = None,
    ):
        """Initialise the FileManagerComponent.
        Either `agent_id` or `config` must be provided.

        Args:
            file_storage (FileStorage): The file storage instance to use.
            state (BaseAgentSettings): The agent's state.
            config (FileManagerConfiguration, optional): The configuration for
            the file manager. Defaults to None.
        """
        if not agent_state.agent_id:
            raise ValueError("Agent must have an ID.")

        self.agent_state = agent_state

        if not config:
            storage_path = f"agents/{self.agent_state.agent_id}/"
            workspace_path = f"agents/{self.agent_state.agent_id}/workspace"
            ConfigurableComponent.__init__(
                self,
                FileManagerConfiguration(
                    storage_path=storage_path, workspace_path=workspace_path
                ),
            )
        else:
            ConfigurableComponent.__init__(self, config)

        self.storage = file_storage.clone_with_subroot(self.config.storage_path)
        """Agent-related files, e.g. state, logs.
        Use `workspace` to access the agent's workspace files."""
        self.workspace = file_storage.clone_with_subroot(self.config.workspace_path)
        """Workspace that the agent has access to, e.g. for reading/writing files.
        Use `storage` to access agent-related files, e.g. state, logs."""
        self._file_storage = file_storage
        
        # Create backup directory if enabled
        if self.config.backup_enabled:
            backup_path = Path(self.config.workspace_path) / self.config.backup_dir
            self._file_storage.make_dir(backup_path)

    async def save_state(self, save_as_id: Optional[str] = None) -> None:
        """Save the agent's data and state."""
        if save_as_id:
            self._file_storage.make_dir(f"agents/{save_as_id}")
            # Save state
            await self._file_storage.write_file(
                f"agents/{save_as_id}/{self.STATE_FILE}",
                self.agent_state.model_dump_json(),
            )
            # Copy workspace
            self._file_storage.copy(
                self.config.workspace_path,
                f"agents/{save_as_id}/workspace",
            )
        else:
            await self.storage.write_file(
                self.storage.root / self.STATE_FILE, self.agent_state.model_dump_json()
            )

    def get_resources(self) -> Iterator[str]:
        yield "The ability to read and write files."

    def get_commands(self) -> Iterator[Command]:
        yield self.read_file
        yield self.write_to_file
        yield self.list_folder
        yield self.delete_file
        yield self.rename_file
        yield self.copy_file
        yield self.create_folder

    @command(
        parameters={
            "filename": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The path of the file to read",
                required=True,
            )
        },
    )
    def read_file(self, filename: Union[str, Path]) -> str:
        """Read a file and return the contents

        Args:
            filename (str): The name of the file to read

        Returns:
            str: The contents of the file
        """
        # Validate file extension
        self._validate_file_extension(filename)
        
        try:
            file = self.workspace.open_file(filename, binary=True)
            content = decode_textual_file(file, os.path.splitext(filename)[1], logger)
            return content
        except FileNotFoundError:
            raise FileNotFoundError(f"File '{filename}' does not exist")
        except UnicodeDecodeError:
            return f"Cannot read '{filename}' as text - it appears to be a binary file."
        except Exception as e:
            raise RuntimeError(f"Error reading file '{filename}': {str(e)}")

    @command(
        ["write_file", "create_file"],
        "Write a file, creating it if necessary. "
        "If the file exists, it is overwritten.",
        {
            "filename": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The name of the file to write to",
                required=True,
            ),
            "contents": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The contents to write to the file",
                required=True,
            ),
        },
    )
    async def write_to_file(self, filename: Union[str, Path], contents: str) -> str:
        """Write contents to a file

        Args:
            filename (str): The name of the file to write to
            contents (str): The contents to write to the file

        Returns:
            str: A message indicating success or failure
        """
        # Validate file extension
        self._validate_file_extension(filename)
        
        # Check content size
        if len(contents.encode('utf-8')) > self.config.max_file_size:
            raise ValueError(
                f"File content exceeds maximum size of {self.config.max_file_size} bytes"
            )
        
        try:
            # Create directory if needed
            if directory := os.path.dirname(filename):
                self.workspace.make_dir(directory)
            
            # Backup existing file if enabled
            if self.config.backup_enabled and self.workspace.exists(filename):
                await self._backup_file(filename)
            
            # Write the file
            await self.workspace.write_file(filename, contents)
            return f"File {filename} has been written successfully."
        except Exception as e:
            raise RuntimeError(f"Error writing to file '{filename}': {str(e)}")

    @command(
        parameters={
            "folder": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The folder to list files in",
                required=True,
            )
        },
    )
    def list_folder(self, folder: Union[str, Path]) -> list[str]:
        """Lists files in a folder recursively

        Args:
            folder (str): The folder to search in

        Returns:
            list[str]: A list of files found in the folder
        """
        try:
            paths = self.workspace.list_files(folder)
            return [str(p) for p in paths]
        except FileNotFoundError:
            return []  # Return empty list for non-existent directory
        except Exception as e:
            raise RuntimeError(f"Error listing folder '{folder}': {str(e)}")

    @command(
        parameters={
            "filename": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The name of the file to delete",
                required=True,
            )
        },
    )
    def delete_file(self, filename: Union[str, Path]) -> str:
        """Delete a file

        Args:
            filename (str): The name of the file to delete

        Returns:
            str: A message indicating success or failure
        """
        try:
            # Check if the file exists
            if not self.workspace.exists(filename):
                return f"File '{filename}' does not exist."
            
            # Backup file if enabled
            if self.config.backup_enabled:
                # Create backup in async way
                import asyncio
                loop = asyncio.get_event_loop()
                loop.create_task(self._backup_file(filename))
            
            # Delete the file
            self.workspace.delete_file(filename)
            return f"File '{filename}' has been deleted successfully."
        except Exception as e:
            raise RuntimeError(f"Error deleting file '{filename}': {str(e)}")

    @command(
        parameters={
            "old_path": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The current path of the file or folder",
                required=True,
            ),
            "new_path": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The new path for the file or folder",
                required=True,
            ),
        },
    )
    def rename_file(self, old_path: Union[str, Path], new_path: Union[str, Path]) -> str:
        """Rename or move a file or folder

        Args:
            old_path (str): The current path of the file or folder
            new_path (str): The new path for the file or folder

        Returns:
            str: A message indicating success or failure
        """
        try:
            # Check if source exists
            if not self.workspace.exists(old_path):
                return f"File or folder '{old_path}' does not exist."
            
            # Validate new file extension if it's a file
            if os.path.isfile(str(self.workspace.get_path(old_path))):
                self._validate_file_extension(new_path)
            
            # Create directory for destination if needed
            new_dir = os.path.dirname(new_path)
            if new_dir:
                self.workspace.make_dir(new_dir)
            
            # Perform the rename/move
            old_full_path = self.workspace.get_path(old_path)
            new_full_path = self.workspace.get_path(new_path)
            
            if os.path.exists(new_full_path):
                return f"Destination '{new_path}' already exists. Please delete it first or choose a different name."
            
            os.rename(old_full_path, new_full_path)
            return f"Successfully renamed/moved '{old_path}' to '{new_path}'."
        except Exception as e:
            raise RuntimeError(f"Error renaming '{old_path}' to '{new_path}': {str(e)}")

    @command(
        parameters={
            "source": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The source file or folder to copy",
                required=True,
            ),
            "destination": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The destination path",
                required=True,
            ),
        },
    )
    def copy_file(self, source: Union[str, Path], destination: Union[str, Path]) -> str:
        """Copy a file or folder

        Args:
            source (str): The source file or folder to copy
            destination (str): The destination path

        Returns:
            str: A message indicating success or failure
        """
        try:
            # Check if source exists
            source_path = self.workspace.get_path(source)
            if not os.path.exists(source_path):
                return f"Source '{source}' does not exist."
            
            # Get destination path
            destination_path = self.workspace.get_path(destination)
            
            # Create directory for destination if needed
            destination_dir = os.path.dirname(destination_path)
            if destination_dir:
                os.makedirs(os.path.dirname(destination_path), exist_ok=True)
            
            # Copy file or directory
            if os.path.isdir(source_path):
                if os.path.exists(destination_path):
                    return f"Destination '{destination}' already exists. Please delete it first or choose a different name."
                shutil.copytree(source_path, destination_path)
                return f"Successfully copied directory '{source}' to '{destination}'."
            else:
                # Validate file extension
                self._validate_file_extension(destination)
                
                # Copy the file
                shutil.copy2(source_path, destination_path)
                return f"Successfully copied file '{source}' to '{destination}'."
        except Exception as e:
            raise RuntimeError(f"Error copying '{source}' to '{destination}': {str(e)}")

    @command(
        parameters={
            "folder_path": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The path of the folder to create",
                required=True,
            )
        },
    )
    def create_folder(self, folder_path: Union[str, Path]) -> str:
        """Create a new folder

        Args:
            folder_path (str): The path of the folder to create

        Returns:
            str: A message indicating success or failure
        """
        try:
            # Check if folder already exists
            if self.workspace.exists(folder_path):
                return f"Folder '{folder_path}' already exists."
            
            # Create the folder
            self.workspace.make_dir(folder_path)
            return f"Folder '{folder_path}' created successfully."
        except Exception as e:
            raise RuntimeError(f"Error creating folder '{folder_path}': {str(e)}")

    async def _backup_file(self, filename: Union[str, Path]) -> None:
        """Create a backup of a file before modifying it"""
        if not self.workspace.exists(filename):
            return
            
        try:
            source_path = self.workspace.get_path(filename)
            if not os.path.isfile(source_path):
                return  # Only backup files, not directories
                
            # Create backup filename with timestamp
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_str = str(filename)
            base_name = os.path.basename(filename_str)
            backup_name = f"{base_name}.{timestamp}.bak"
            
            # Create backup path
            backup_dir = Path(self.config.backup_dir)
            backup_path = backup_dir / backup_name
            
            # Ensure backup directory exists
            self.workspace.make_dir(backup_dir)
            
            # Read source file
            content = self.workspace.read_binary(filename)
            
            # Write to backup location
            await self.workspace.write_binary(backup_path, content)
            logger.debug(f"Created backup of '{filename}' at '{backup_path}'")
        except Exception as e:
            logger.warning(f"Failed to create backup of '{filename}': {e}")

    def _validate_file_extension(self, filename: Union[str, Path]) -> None:
        """Validate that the file extension is allowed"""
        filename_str = str(filename)
        
        # Allow files without extension
        if '.' not in filename_str:
            return
            
        # Get extension with the dot
        ext = os.path.splitext(filename_str)[1].lower()
        
        # Check if extension is allowed
        if self.config.allowed_extensions and ext not in self.config.allowed_extensions:
            allowed_exts = ", ".join(self.config.allowed_extensions)
            raise ValueError(
                f"File extension '{ext}' is not allowed. Allowed extensions: {allowed_exts}"
            )
