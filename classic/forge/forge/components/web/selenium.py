import asyncio
import logging
import os
import platform
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from sys import platform as sys_platform
from typing import Iterator, Literal, Optional, Type
from urllib.request import urlretrieve

from bs4 import BeautifulSoup
from pydantic import BaseModel
from selenium.common.exceptions import WebDriverException, TimeoutException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeDriverService
from selenium.webdriver.chrome.webdriver import WebDriver as ChromeDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.edge.options import Options as EdgeOptions
from selenium.webdriver.edge.service import Service as EdgeDriverService
from selenium.webdriver.edge.webdriver import WebDriver as EdgeDriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.firefox.service import Service as GeckoDriverService
from selenium.webdriver.firefox.webdriver import WebDriver as FirefoxDriver
from selenium.webdriver.remote.webdriver import WebDriver
from selenium.webdriver.safari.options import Options as SafariOptions
from selenium.webdriver.safari.webdriver import WebDriver as SafariDriver
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

# Import driver managers conditionally to handle missing dependencies gracefully
try:
    from webdriver_manager.chrome import ChromeDriverManager
    from webdriver_manager.firefox import GeckoDriverManager
    from webdriver_manager.microsoft import EdgeChromiumDriverManager as EdgeDriverManager
    WEBDRIVER_MANAGER_AVAILABLE = True
except ImportError:
    WEBDRIVER_MANAGER_AVAILABLE = False

from forge.agent.components import ConfigurableComponent
from forge.agent.protocols import CommandProvider, DirectiveProvider
from forge.command import Command, command
from forge.content_processing.html import extract_hyperlinks, format_hyperlinks
from forge.content_processing.text import extract_information, summarize_text
from forge.llm.providers import MultiProvider
from forge.llm.providers.multi import ModelName
from forge.llm.providers.openai import OpenAIModelName
from forge.models.json_schema import JSONSchema
from forge.utils.exceptions import CommandExecutionError, TooMuchOutputError
from forge.utils.url_validator import validate_url

logger = logging.getLogger(__name__)

FILE_DIR = Path(__file__).parent.parent
MAX_RAW_CONTENT_LENGTH = 500
LINKS_TO_RETURN = 20


BrowserOptions = ChromeOptions | EdgeOptions | FirefoxOptions | SafariOptions


class BrowsingError(CommandExecutionError):
    """An error occurred while trying to browse the page"""


class WebSeleniumConfiguration(BaseModel):
    llm_name: ModelName = OpenAIModelName.GPT3
    """Name of the llm model used to read websites"""
    web_browser: Literal["chrome", "firefox", "safari", "edge"] = "chrome"
    """Web browser used by Selenium"""
    headless: bool = True
    """Run browser in headless mode"""
    user_agent: str = (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_4) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/83.0.4103.97 Safari/537.36"
    )
    """User agent used by the browser"""
    browse_spacy_language_model: str = "en_core_web_sm"
    """Spacy language model used for chunking text"""
    selenium_proxy: Optional[str] = None
    """Http proxy to use with Selenium"""
    driver_install_timeout: int = 30
    """Timeout for driver installation in seconds"""
    page_load_timeout: int = 30
    """Timeout for page loading in seconds"""
    local_driver_path: Optional[str] = None
    """Path to local browser driver executable (overrides automatic download)"""
    max_retries: int = 3
    """Maximum number of retries for browser operations"""
    retry_delay: int = 2
    """Delay between retries in seconds"""


class WebSeleniumComponent(
    DirectiveProvider, CommandProvider, ConfigurableComponent[WebSeleniumConfiguration]
):
    """Provides commands to browse the web using Selenium."""

    config_class = WebSeleniumConfiguration

    def __init__(
        self,
        llm_provider: MultiProvider,
        data_dir: Path,
        config: Optional[WebSeleniumConfiguration] = None,
    ):
        ConfigurableComponent.__init__(self, config)
        self.llm_provider = llm_provider
        self.data_dir = data_dir
        
        # Create directories for drivers and extensions if they don't exist
        self.drivers_dir = data_dir / "drivers"
        self.drivers_dir.mkdir(exist_ok=True, parents=True)
        
        self.extensions_dir = data_dir / "assets" / "crx"
        self.extensions_dir.mkdir(exist_ok=True, parents=True)
        
        # Try to find local Chrome/Chromium binary
        self.local_browser_path = self._find_local_browser()
        if self.local_browser_path:
            logger.info(f"Found local browser at: {self.local_browser_path}")

    def get_resources(self) -> Iterator[str]:
        yield "Ability to read websites."

    def get_commands(self) -> Iterator[Command]:
        yield self.read_webpage

    def _find_local_browser(self) -> Optional[str]:
        """Find local Chrome/Chromium binary"""
        if sys_platform == "linux" or sys_platform == "linux2":
            # Check common Linux locations
            linux_paths = [
                "/usr/bin/google-chrome",
                "/usr/bin/google-chrome-stable",
                "/usr/bin/chrome",
                "/usr/bin/chromium",
                "/usr/bin/chromium-browser",
                "/snap/bin/chromium",
                "/snap/bin/google-chrome",
            ]
            for path in linux_paths:
                if os.path.exists(path):
                    return path
                    
        elif sys_platform == "darwin":
            # Check macOS locations
            mac_paths = [
                "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
                "/Applications/Chromium.app/Contents/MacOS/Chromium",
            ]
            for path in mac_paths:
                if os.path.exists(path):
                    return path
                    
        elif sys_platform == "win32":
            # Check Windows locations
            windows_paths = [
                os.path.expandvars(r"%ProgramFiles%\Google\Chrome\Application\chrome.exe"),
                os.path.expandvars(r"%ProgramFiles(x86)%\Google\Chrome\Application\chrome.exe"),
                os.path.expandvars(r"%LocalAppData%\Google\Chrome\Application\chrome.exe"),
            ]
            for path in windows_paths:
                if os.path.exists(path):
                    return path
        
        return None

    @command(
        ["read_webpage"],
        (
            "Read a webpage, and extract specific information from it."
            " You must specify either topics_of_interest,"
            " a question, or get_raw_content."
        ),
        {
            "url": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The URL to visit",
                required=True,
            ),
            "topics_of_interest": JSONSchema(
                type=JSONSchema.Type.ARRAY,
                items=JSONSchema(type=JSONSchema.Type.STRING),
                description=(
                    "A list of topics about which you want to extract information "
                    "from the page."
                ),
                required=False,
            ),
            "question": JSONSchema(
                type=JSONSchema.Type.STRING,
                description=(
                    "A question you want to answer using the content of the webpage."
                ),
                required=False,
            ),
            "get_raw_content": JSONSchema(
                type=JSONSchema.Type.BOOLEAN,
                description=(
                    "If true, the unprocessed content of the webpage will be returned. "
                    "This consumes a lot of tokens, so use it with caution."
                ),
                required=False,
            ),
        },
    )
    @validate_url
    async def read_webpage(
        self,
        url: str,
        *,
        topics_of_interest: list[str] = [],
        get_raw_content: bool = False,
        question: str = "",
    ) -> str:
        """Browse a website and return the answer and links to the user

        Args:
            url (str): The url of the website to browse
            question (str): The question to answer using the content of the webpage

        Returns:
            str: The answer and links to the user and the webdriver
        """
        driver = None
        
        # Add retry logic for page loading
        retry_count = 0
        while retry_count < self.config.max_retries:
            try:
                driver = await self.open_page_in_browser(url)
                
                # If we got here, the page loaded successfully
                break
                
            except Exception as e:
                retry_count += 1
                if retry_count >= self.config.max_retries:
                    # This is our last attempt, re-raise the exception
                    msg = str(e)
                    if "net::" in msg:
                        raise BrowsingError(
                            "A networking error occurred while trying to load the page: "
                            f"{re.sub(r'^unknown error: ', '', msg)}"
                        )
                    raise CommandExecutionError(f"Failed to load page after {self.config.max_retries} attempts: {e}")
                
                logger.warning(f"Error loading page (attempt {retry_count}/{self.config.max_retries}): {e}")
                await asyncio.sleep(self.config.retry_delay)
        
        try:
            try:
                # Wait for any remaining dynamic content to load
                await asyncio.sleep(2)
                
                # Get the text and links
                text = self.scrape_text_with_selenium(driver)
                links = self.scrape_links_with_selenium(driver, url)
                
                return_literal_content = True
                summarized = False
                
                if not text:
                    return f"Website did not contain any text.\n\nLinks: {links}"
                elif get_raw_content:
                    if (
                        output_tokens := self.llm_provider.count_tokens(
                            text, self.config.llm_name
                        )
                    ) > MAX_RAW_CONTENT_LENGTH:
                        oversize_factor = round(output_tokens / MAX_RAW_CONTENT_LENGTH, 1)
                        raise TooMuchOutputError(
                            f"Page content is {oversize_factor}x the allowed length "
                            "for `get_raw_content=true`"
                        )
                    return text + (f"\n\nLinks: {links}" if links else "")
                else:
                    text = await self.summarize_webpage(
                        text, question or None, topics_of_interest
                    )
                    return_literal_content = bool(question)
                    summarized = True
                
                # Limit links to LINKS_TO_RETURN
                if len(links) > LINKS_TO_RETURN:
                    links = links[:LINKS_TO_RETURN]
                
                text_fmt = f"'''{text}'''" if "\n" in text else f"'{text}'"
                links_fmt = "\n".join(f"- {link}" for link in links)
                
                return (
                    f"Page content{' (summary)' if summarized else ''}:"
                    if return_literal_content
                    else "Answer gathered from webpage:"
                ) + f" {text_fmt}\n\nLinks:\n{links_fmt}"
                
            except TooMuchOutputError:
                raise
            except Exception as e:
                logger.error(f"Error processing webpage content: {e}")
                # If content processing fails, try to return at least the raw page source
                try:
                    if driver:
                        page_source = driver.page_source
                        soup = BeautifulSoup(page_source, "html.parser")
                        basic_text = soup.get_text()
                        truncated_text = basic_text[:1000] + "..." if len(basic_text) > 1000 else basic_text
                        return f"Error processing page content: {str(e)}\n\nPartial raw content:\n{truncated_text}"
                except Exception as inner_e:
                    logger.error(f"Failed to extract basic text: {inner_e}")
                    return f"Failed to extract any content from the page. Error: {str(e)}"
                
        except WebDriverException as e:
            # These errors are often quite long and include lots of context.
            # Just grab the first line.
            msg = e.msg.split("\n")[0] if e.msg else str(e)
            if "net::" in msg:
                raise BrowsingError(
                    "A networking error occurred while trying to load the page: %s"
                    % re.sub(r"^unknown error: ", "", msg)
                )
            raise CommandExecutionError(msg)
        finally:
            if driver:
                try:
                    driver.quit()  # quit() is more thorough than close()
                except Exception as e:
                    logger.warning(f"Error closing webdriver: {e}")

    def scrape_text_with_selenium(self, driver: WebDriver) -> str:
        """Scrape text from a browser window using selenium

        Args:
            driver (WebDriver): A driver object representing
            the browser window to scrape

        Returns:
            str: the text scraped from the website
        """
        try:
            # Get the HTML content directly from the browser's DOM
            page_source = driver.execute_script("return document.body.outerHTML;")
            soup = BeautifulSoup(page_source, "html.parser")

            for script in soup(["script", "style"]):
                script.extract()

            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = "\n".join(chunk for chunk in chunks if chunk)
            return text
        except Exception as e:
            logger.warning(f"Error scraping text: {e}")
            # Fallback method if JavaScript execution fails
            try:
                page_source = driver.page_source
                soup = BeautifulSoup(page_source, "html.parser")
                for script in soup(["script", "style"]):
                    script.extract()
                text = soup.get_text()
                return text
            except Exception as e2:
                logger.error(f"Fallback text scraping failed: {e2}")
                return ""

    def scrape_links_with_selenium(self, driver: WebDriver, base_url: str) -> list[str]:
        """Scrape links from a website using selenium

        Args:
            driver (WebDriver): A driver object representing
            the browser window to scrape
            base_url (str): The base URL to use for resolving relative links

        Returns:
            List[str]: The links scraped from the website
        """
        try:
            page_source = driver.page_source
            soup = BeautifulSoup(page_source, "html.parser")

            # Process the page to extract links
            for script in soup(["script", "style"]):
                script.extract()

            hyperlinks = extract_hyperlinks(soup, base_url)
            
            # Filter out unwanted links (e.g., social media, tracking)
            filtered_links = []
            blacklist_patterns = [
                r'facebook\.com', r'twitter\.com', r'instagram\.com', 
                r'linkedin\.com', r'pinterest\.com', r'youtube\.com',
                r'google\.com/analytics', r'googleads', r'doubleclick\.net',
                r'javascript:void', r'mailto:', r'whatsapp:', r'tel:'
            ]
            
            for link in hyperlinks:
                if not any(re.search(pattern, link[1]) for pattern in blacklist_patterns):
                    filtered_links.append(link)

            return format_hyperlinks(filtered_links)
        except Exception as e:
            logger.warning(f"Error scraping links: {e}")
            return []

    async def open_page_in_browser(self, url: str) -> WebDriver:
        """Open a browser window and load a web page using Selenium

        Params:
            url (str): The URL of the page to load

        Returns:
            driver (WebDriver): A driver object representing
            the browser window to scrape
        """
        # Silence selenium logger
        selenium_logger = logging.getLogger("selenium")
        selenium_logger.setLevel(logging.CRITICAL)
        webdriver_manager_logger = logging.getLogger("webdriver_manager")
        webdriver_manager_logger.setLevel(logging.CRITICAL)
        
        if not WEBDRIVER_MANAGER_AVAILABLE and not self.config.local_driver_path:
            logger.warning(
                "WebDriver Manager not available and no local driver specified. "
                "Install webdriver-manager package or specify a local driver path."
            )

        options_available: dict[str, Type[BrowserOptions]] = {
            "chrome": ChromeOptions,
            "edge": EdgeOptions,
            "firefox": FirefoxOptions,
            "safari": SafariOptions,
        }

        options: BrowserOptions = options_available[self.config.web_browser]()
        options.add_argument(f"user-agent={self.config.user_agent}")

        if isinstance(options, FirefoxOptions):
            if self.config.headless:
                options.headless = True  # type: ignore
                options.add_argument("--disable-gpu")
            
            if self.config.local_driver_path:
                gecko_service = GeckoDriverService(executable_path=self.config.local_driver_path)
            elif WEBDRIVER_MANAGER_AVAILABLE:
                gecko_service = GeckoDriverService(GeckoDriverManager().install())
            else:
                # Look for geckodriver in PATH
                gecko_path = self._find_driver_in_path("geckodriver")
                if not gecko_path:
                    raise CommandExecutionError("Cannot find geckodriver. Please install webdriver-manager or specify a local driver path.")
                gecko_service = GeckoDriverService(executable_path=gecko_path)
                
            driver = FirefoxDriver(service=gecko_service, options=options)
            
        elif isinstance(options, EdgeOptions):
            if self.config.local_driver_path:
                edge_service = EdgeDriverService(executable_path=self.config.local_driver_path)
            elif WEBDRIVER_MANAGER_AVAILABLE:
                edge_service = EdgeDriverService(EdgeDriverManager().install())
            else:
                # Look for msedgedriver in PATH
                edge_path = self._find_driver_in_path("msedgedriver")
                if not edge_path:
                    raise CommandExecutionError("Cannot find msedgedriver. Please install webdriver-manager or specify a local driver path.")
                edge_service = EdgeDriverService(executable_path=edge_path)
                
            driver = EdgeDriver(service=edge_service, options=options)
            
        elif isinstance(options, SafariOptions):
            # Requires a bit more setup on the users end.
            # See https://developer.apple.com/documentation/webkit/testing_with_webdriver_in_safari  # noqa: E501
            driver = SafariDriver(options=options)
            
        elif isinstance(options, ChromeOptions):
            if sys_platform == "linux" or sys_platform == "linux2":
                options.add_argument("--disable-dev-shm-usage")
                options.add_argument("--remote-debugging-port=9222")

            options.add_argument("--no-sandbox")
            if self.config.headless:
                options.add_argument("--headless=new")
                options.add_argument("--disable-gpu")

            if self.config.selenium_proxy:
                options.add_argument(f"--proxy-server={self.config.selenium_proxy}")
            
            # Add extensions for cookie handling and ad blocking
            self._sideload_chrome_extensions(options, self.extensions_dir)
            
            # Additional options to help with stability
            options.add_argument("--disable-extensions")
            options.add_argument("--disable-infobars")
            options.add_argument("--disable-notifications")
            options.add_argument("--disable-popup-blocking")
            
            # Set browser binary location if found
            if self.local_browser_path:
                options.binary_location = self.local_browser_path

            # Use specified local driver or try various methods to find one
            if self.config.local_driver_path:
                chrome_service = ChromeDriverService(executable_path=self.config.local_driver_path)
            elif os.path.exists("/usr/bin/chromedriver"):
                chrome_service = ChromeDriverService(executable_path="/usr/bin/chromedriver")
            elif WEBDRIVER_MANAGER_AVAILABLE:
                try:
                    chrome_driver = ChromeDriverManager().install()
                    chrome_service = ChromeDriverService(executable_path=chrome_driver)
                except Exception as e:
                    logger.warning(f"Error installing ChromeDriver: {e}")
                    # Try fallback methods
                    chrome_path = self._find_driver_in_path("chromedriver")
                    if chrome_path:
                        chrome_service = ChromeDriverService(executable_path=chrome_path)
                    else:
                        # Try to download chromedriver manually
                        chrome_path = self._download_chromedriver_manually()
                        if chrome_path:
                            chrome_service = ChromeDriverService(executable_path=chrome_path)
                        else:
                            raise CommandExecutionError(
                                "Could not find or install ChromeDriver. "
                                "Please install it manually and specify the path."
                            )
            else:
                chrome_path = self._find_driver_in_path("chromedriver")
                if chrome_path:
                    chrome_service = ChromeDriverService(executable_path=chrome_path)
                else:
                    chrome_path = self._download_chromedriver_manually()
                    if chrome_path:
                        chrome_service = ChromeDriverService(executable_path=chrome_path)
                    else:
                        raise CommandExecutionError(
                            "Could not find ChromeDriver and webdriver-manager is not installed. "
                            "Please install webdriver-manager package or specify a local driver path."
                        )

            driver = ChromeDriver(service=chrome_service, options=options)

        # Set timeouts
        driver.set_page_load_timeout(self.config.page_load_timeout)
        driver.set_script_timeout(self.config.page_load_timeout)
        
        try:
            driver.get(url)

            # Wait for page to be ready
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Wait a moment for any cookie banners or anti-bot systems to appear
            await asyncio.sleep(2)
            
            # Wait again for body to ensure the page is fully loaded
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )

            return driver
        except Exception as e:
            # Make sure to quit the driver if an error occurs during loading
            try:
                driver.quit()
            except:
                pass
            raise e

    def _find_driver_in_path(self, driver_name: str) -> Optional[str]:
        """Find a driver executable in PATH"""
        # Add .exe extension for Windows
        if sys_platform == "win32":
            driver_name = f"{driver_name}.exe"
            
        # Check if the driver is in PATH
        driver_path = shutil.which(driver_name)
        if driver_path:
            return driver_path
            
        return None

    def _download_chromedriver_manually(self) -> Optional[str]:
        """Download ChromeDriver manually as a fallback"""
        try:
            import requests
            import zipfile
            import io
            
            # Determine the ChromeDriver version needed
            chrome_version = self._get_chrome_version()
            if not chrome_version:
                logger.warning("Could not determine Chrome version")
                return None
                
            major_version = chrome_version.split(".")[0]
            
            # Get the corresponding ChromeDriver version
            version_url = f"https://chromedriver.storage.googleapis.com/LATEST_RELEASE_{major_version}"
            response = requests.get(version_url)
            if response.status_code != 200:
                logger.warning(f"Failed to get ChromeDriver version: HTTP {response.status_code}")
                return None
                
            driver_version = response.text.strip()
            
            # Determine platform
            if sys_platform == "win32":
                platform_name = "win32"
            elif sys_platform == "darwin":
                if platform.machine() == "arm64":
                    platform_name = "mac_arm64"
                else:
                    platform_name = "mac64"
            else:
                platform_name = "linux64"
                
            # Download ChromeDriver
            download_url = f"https://chromedriver.storage.googleapis.com/{driver_version}/chromedriver_{platform_name}.zip"
            logger.info(f"Downloading ChromeDriver from {download_url}")
            
            response = requests.get(download_url)
            if response.status_code != 200:
                logger.warning(f"Failed to download ChromeDriver: HTTP {response.status_code}")
                return None
                
            # Extract the zip file
            with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
                # Create the drivers directory if it doesn't exist
                driver_dir = self.drivers_dir
                driver_dir.mkdir(exist_ok=True, parents=True)
                
                # Find the chromedriver executable in the zip
                chromedriver_path = None
                for filename in zip_file.namelist():
                    if filename.endswith('.exe') or filename == 'chromedriver':
                        # Extract the file
                        source = zip_file.open(filename)
                        target_path = driver_dir / filename
                        target = open(target_path, "wb")
                        with source, target:
                            shutil.copyfileobj(source, target)
                        
                        # Make it executable
                        target_path.chmod(0o755)
                        chromedriver_path = str(target_path)
                        break
                
                return chromedriver_path
        except Exception as e:
            logger.warning(f"Error downloading ChromeDriver manually: {e}")
            return None

    def _get_chrome_version(self) -> Optional[str]:
        """Get the installed Chrome/Chromium version"""
        try:
            if self.local_browser_path:
                # Run the browser with --version flag
                cmd = [self.local_browser_path, "--version"]
                output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
                
                # Parse the version from output like "Google Chrome 94.0.4606.81"
                match = re.search(r'(\d+\.\d+\.\d+(\.\d+)?)', output)
                if match:
                    return match.group(1)
            
            # Fallback methods based on platform
            if sys_platform == "win32":
                # Check registry for Chrome version
                import winreg
                try:
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Google\Chrome\BLBeacon") as key:
                        version = winreg.QueryValueEx(key, "version")[0]
                        return version
                except:
                    pass
            elif sys_platform == "darwin":
                # Check for Chrome on macOS
                chrome_path = "/Applications/Google Chrome.app/Contents/Frameworks/Google Chrome Framework.framework/Versions"
                if os.path.exists(chrome_path):
                    versions = [v for v in os.listdir(chrome_path) if re.match(r'\d+', v)]
                    if versions:
                        return max(versions)
            elif sys_platform.startswith("linux"):
                # Try various Linux Chrome/Chromium locations
                for cmd in [
                    ["google-chrome", "--version"],
                    ["chromium", "--version"],
                    ["chromium-browser", "--version"],
                ]:
                    try:
                        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)
                        match = re.search(r'(\d+\.\d+\.\d+(\.\d+)?)', output)
                        if match:
                            return match.group(1)
                    except:
                        continue
            
            return None
        except Exception as e:
            logger.warning(f"Error determining Chrome version: {e}")
            return None

    def _sideload_chrome_extensions(
        self, options: ChromeOptions, dl_folder: Path
    ) -> None:
        """Download and load Chrome extensions for improved browsing"""
        crx_download_url_template = "https://clients2.google.com/service/update2/crx?response=redirect&prodversion=99.0&acceptformat=crx3&x=id%3D{crx_id}%26installsource%3Dondemand%26uc"  # noqa
        
        # Extension IDs for useful tools
        extensions = {
            # Cookie Auto-Delete
            "edibdbjcniadpccecjdfdjjppcpchdlm": "Cookie Auto-Delete",
            # uBlock Origin
            "cjpalhdlnbpafiamejdnhcphjbkeiagm": "uBlock Origin",
            # I don't care about cookies
            "fihnjjcciajhdojfnbdddfaoknhalnja": "I don't care about cookies",
        }

        # Make sure the target folder exists
        dl_folder.mkdir(parents=True, exist_ok=True)

        for crx_id, name in extensions.items():
            crx_path = dl_folder / f"{crx_id}.crx"
            if not crx_path.exists():
                try:
                    logger.debug(f"Downloading {name} extension...")
                    crx_download_url = crx_download_url_template.format(crx_id=crx_id)
                    urlretrieve(crx_download_url, crx_path)
                    logger.debug(f"Downloaded {crx_path.name}")
                except Exception as e:
                    logger.warning(f"Failed to download {name} extension: {e}")
                    continue
            
            try:
                options.add_extension(str(crx_path))
            except Exception as e:
                logger.warning(f"Failed to add {name} extension: {e}")

    async def summarize_webpage(
        self,
        text: str,
        question: str | None,
        topics_of_interest: list[str],
    ) -> str:
        """Summarize text using the LLM provider

        Args:
            text (str): The text to summarize
            question (str): The question to ask the model
            topics_of_interest (list): List of topics to extract information about

        Returns:
            str: The summary of the text
        """
        if not text:
            raise ValueError("No text to summarize")

        text_length = len(text)
        logger.debug(f"Web page content length: {text_length} characters")

        # Implement retry logic for summarization
        max_retries = 2
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                result = None
                information = None
                
                if topics_of_interest:
                    information = await extract_information(
                        text,
                        topics_of_interest=topics_of_interest,
                        llm_provider=self.llm_provider,
                        model_name=self.config.llm_name,
                        spacy_model=self.config.browse_spacy_language_model,
                    )
                    return "\n".join(f"* {i}" for i in information)
                else:
                    result, _ = await summarize_text(
                        text,
                        question=question,
                        llm_provider=self.llm_provider,
                        model_name=self.config.llm_name,
                        spacy_model=self.config.browse_spacy_language_model,
                    )
                    return result
            except Exception as e:
                retry_count += 1
                logger.warning(f"Error during text summarization (attempt {retry_count}/{max_retries+1}): {e}")
                
                if retry_count <= max_retries:
                    # Try with a smaller chunk of text on retries
                    if len(text) > 10000:
                        text = text[:10000] + "..."
                    await asyncio.sleep(1)
                else:
                    # Return a simple extraction of the text on failure
                    logger.error(f"Failed to summarize text after {max_retries+1} attempts: {e}")
                    # Extract some key parts of the text as a fallback
                    text_parts = text.split('\n')
                    important_parts = []
                    
                    # Try to find headings and paragraphs
                    for i, part in enumerate(text_parts):
                        part = part.strip()
                        if not part:
                            continue
                            
                        # If it looks like a heading
                        if (len(part) < 100 and part.endswith(':')) or (part.upper() == part and len(part) < 100):
                            important_parts.append(part)
                            # Include the following paragraph if available
                            if i + 1 < len(text_parts) and text_parts[i + 1].strip():
                                next_part = text_parts[i + 1].strip()
                                if len(next_part) > 50:  # Only if it looks like a paragraph
                                    important_parts.append(next_part)
                    
                    # If we found structured content
                    if important_parts:
                        return "Failed to generate a proper summary. Here are some key parts extracted from the text:\n\n" + "\n\n".join(important_parts[:5])
                    else:
                        # Just return the beginning of the text
                        return "Failed to generate a summary. Here's the beginning of the content:\n\n" + text[:1000] + "..."
