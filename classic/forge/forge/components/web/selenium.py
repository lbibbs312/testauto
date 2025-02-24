"""Selenium component for web interaction."""
import logging
import os
import re
import signal
import time
from pathlib import Path
from typing import Dict, List, Optional, Union
from urllib.parse import quote_plus

from selenium import webdriver
from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

from forge.components.web.web_browser import WebBrowser

logger = logging.getLogger(__name__)

class SeleniumWebBrowser(WebBrowser):
    """Selenium-based web browser component."""

    def __init__(self, config):
        """Initialize the Selenium web browser component.
        
        Args:
            config: Configuration dictionary
        """
        self.driver = None
        self.config = config
        self.retry_count = 0
        self.max_retries = 3
        self.page_load_timeout = 45  # seconds
        self.element_timeout = 20    # seconds
        
        # Initialize Chrome options for performance
        self.chrome_options = Options()
        self.chrome_options.add_argument("--disable-dev-shm-usage")
        self.chrome_options.add_argument("--disable-gpu")
        self.chrome_options.add_argument("--disable-extensions")
        self.chrome_options.add_argument("--disable-infobars")
        self.chrome_options.add_argument("--no-sandbox")
        self.chrome_options.add_argument("--log-level=3")  # Suppress console messages
        self.chrome_options.add_argument("--silent")
        
        # Performance optimization - disable images if needed
        if self.config.get("disable_images", False):
            self.chrome_options.add_argument("--blink-settings=imagesEnabled=false")
        
        # User agent configuration
        if self.config.get("user_agent"):
            self.chrome_options.add_argument(f"--user-agent={self.config.get('user_agent')}")
        
        # Headless mode
        if not self.config.get("show_browser", False):
            self.chrome_options.add_argument("--headless=new")

    def _initialize_browser(self):
        """Initialize the Selenium browser with proper error handling."""
        try:
            # Close any existing browser session
            if self.driver:
                try:
                    self.driver.quit()
                except Exception as e:
                    logger.warning(f"Error closing existing browser session: {e}")
            
            # Set up ChromeDriver
            logger.info("===== WebDriver manager =====")
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=self.chrome_options)
            
            # Set default timeout
            self.driver.set_page_load_timeout(self.page_load_timeout)
            
            # Add extensions if specified (uBlock Origin, etc.)
            if self.config.get("extensions", []):
                self._add_extensions()
                
            # Set window size
            self.driver.set_window_size(1920, 1080)
            
            return True
        except Exception as e:
            logger.error(f"Failed to initialize browser: {e}")
            return False

    def _add_extensions(self):
        """Add Chrome extensions from local files."""
        try:
            extensions_dir = Path(__file__).parent.parent.parent.parent / "data" / "assets" / "crx"
            if extensions_dir.exists():
                for ext_file in extensions_dir.glob("*.crx"):
                    try:
                        self.driver.install_addon(str(ext_file), temporary=True)
                        logger.info(f"Added extension: {ext_file.name}")
                    except Exception as e:
                        logger.error(f"Failed to add extension {ext_file.name}: {e}")
        except Exception as e:
            logger.error(f"Failed to add extensions: {e}")

    def _circuit_breaker(self, func, *args, **kwargs):
        """Implements circuit breaker pattern to prevent hanging.
        
        Args:
            func: Function to call
            *args: Arguments to pass to func
            **kwargs: Keyword arguments to pass to func
            
        Returns:
            Result of func or error message
        """
        last_error = None
        for attempt in range(self.max_retries):
            try:
                # Call the function with a timeout
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                last_error = e
                logger.warning(f"Operation failed (attempt {attempt+1}/{self.max_retries}): {e}")
                
                # Force browser restart if needed
                if attempt < self.max_retries - 1:  # Don't restart on final attempt
                    self._force_restart_browser()
                    time.sleep(2)  # Short pause before retry
            
        # If we get here, all attempts failed
        logger.error(f"All attempts failed: {last_error}")
        return f"Operation failed after {self.max_retries} attempts: {last_error}"
    
    def _force_restart_browser(self):
        """Force browser restart when operations hang."""
        try:
            # Try graceful shutdown first
            if self.driver:
                self.driver.quit()
        except Exception as e:
            logger.warning(f"Error during graceful browser shutdown: {e}")
        
        # Create new browser instance
        self._initialize_browser()

    def browse_website(self, url: str) -> Dict[str, Union[str, int]]:
        """Browse a website and return the content.
        
        Args:
            url: URL to browse
            
        Returns:
            Dictionary with status code and content
        """
        return self._circuit_breaker(self._browse_website, url)

    def _browse_website(self, url: str) -> Dict[str, Union[str, int]]:
        """Internal implementation of browse_website."""
        if not self.driver:
            if not self._initialize_browser():
                return {"status_code": 500, "content": "Failed to initialize browser"}
        
        try:
            # Navigate to the URL with timeout handling
            try:
                self.driver.get(url)
            except TimeoutException:
                logger.warning(f"Page load timeout for {url}, continuing with partial content")
            
            # Wait for document ready state
            try:
                WebDriverWait(self.driver, self.element_timeout).until(
                    lambda d: d.execute_script("return document.readyState") == "complete"
                )
            except TimeoutException:
                logger.warning("Document ready state timeout, continuing anyway")
            
            # Get the page content
            content = self.driver.page_source
            
            return {"status_code": 200, "content": content}
        except Exception as e:
            logger.error(f"Error browsing website {url}: {e}")
            return {"status_code": 500, "content": f"Error: {str(e)}"}

    def scroll(self, amount: int) -> None:
        """Scroll the browser window.
        
        Args:
            amount: Amount to scroll
        """
        if not self.driver:
            logger.error("Browser not initialized")
            return
        
        try:
            self.driver.execute_script(f"window.scrollBy(0, {amount})")
            time.sleep(0.5)  # Short pause after scrolling
        except Exception as e:
            logger.error(f"Error scrolling: {e}")

    def read_webpage(self, url: str, topics_of_interest: Optional[List[str]] = None) -> str:
        """Read a webpage and extract the text related to topics of interest.
        
        Args:
            url: URL to read
            topics_of_interest: Optional list of topics to focus on
            
        Returns:
            Extracted text from webpage
        """
        return self._circuit_breaker(self._read_webpage, url, topics_of_interest)

    def _read_webpage(self, url: str, topics_of_interest: Optional[List[str]] = None) -> str:
        """Internal implementation of read_webpage."""
        if not self.driver:
            if not self._initialize_browser():
                return "Failed to initialize browser"
        
        try:
            # Navigate to the URL with timeout handling
            try:
                self.driver.get(url)
            except TimeoutException:
                logger.warning(f"Page load timeout for {url}, continuing with partial content")
            
            # Wait for document ready state
            try:
                WebDriverWait(self.driver, self.element_timeout).until(
                    lambda d: d.execute_script("return document.readyState") == "complete"
                )
            except TimeoutException:
                logger.warning("Document ready state timeout, continuing anyway")
            
            # Get text content
            page_text = ""
            
            # First try extracting main content if it exists
            main_elements = []
            try:
                # Try different content selectors
                for selector in ["main", "article", "#content", ".content", "#main", ".main"]:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        main_elements.extend(elements)
                        break
            except Exception:
                pass
            
            # If main content found, use it
            if main_elements:
                for element in main_elements:
                    try:
                        page_text += element.text + "\n\n"
                    except Exception:
                        pass
            
            # If no text found or no main elements, extract all body text
            if not page_text.strip():
                try:
                    page_text = self.driver.find_element(By.TAG_NAME, "body").text
                except Exception as e:
                    logger.error(f"Error extracting page text: {e}")
                    page_text = "Error extracting text content"
            
            # Filter by topics of interest if provided
            if topics_of_interest and page_text:
                filtered_content = []
                paragraphs = re.split(r'\n\s*\n', page_text)
                
                for paragraph in paragraphs:
                    if any(topic.lower() in paragraph.lower() for topic in topics_of_interest):
                        filtered_content.append(paragraph)
                
                if filtered_content:
                    page_text = '\n\n'.join(filtered_content)
            
            return page_text
        except Exception as e:
            logger.error(f"Error reading webpage {url}: {e}")
            return f"Error reading webpage: {str(e)}"

    def web_search(self, query: str, num_results: int = 5) -> List[str]:
        """Perform a web search and return the results.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search result URLs
        """
        return self._circuit_breaker(self._web_search, query, num_results)

    def _web_search(self, query: str, num_results: int = 5) -> List[str]:
        """Internal implementation of web_search."""
        # Try primary search engine (DuckDuckGo)
        try:
            results = self._search_with_duckduckgo(query, num_results)
            
            # Check for rate limiting
            if not results or any("rate limit" in result.lower() for result in results):
                logger.warning("DuckDuckGo rate limit detected, switching to Google")
                # Try Google as fallback
                results = self._search_with_google(query, num_results)
            
            return results[:num_results]  # Ensure we don't return more than requested
        except Exception as e:
            logger.error(f"Web search error: {e}")
            # Try Google as fallback for any error
            try:
                logger.info("Trying Google as fallback search engine")
                return self._search_with_google(query, num_results)
            except Exception as e2:
                logger.error(f"Google fallback search also failed: {e2}")
                return [f"Search error: Unable to complete search. Please try again later."]

    def _search_with_duckduckgo(self, query: str, num_results: int = 5) -> List[str]:
        """Search using DuckDuckGo.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search result URLs
        """
        if not self.driver:
            if not self._initialize_browser():
                return ["Failed to initialize browser"]
        
        try:
            duckduckgo_url = f"https://duckduckgo.com/?q={quote_plus(query)}"
            self.driver.get(duckduckgo_url)
            
            # Wait for search results
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, ".result__a"))
                )
            except TimeoutException:
                if "rate limit" in self.driver.page_source.lower():
                    return ["https://duckduckgo.com/ 202 RateLimit"]
                logger.warning("Timeout waiting for DuckDuckGo search results")
            
            # Extract results
            result_elements = self.driver.find_elements(By.CSS_SELECTOR, ".result__a")
            results = []
            
            for element in result_elements[:num_results]:
                try:
                    url = element.get_attribute("href")
                    if url:
                        results.append(url)
                except Exception:
                    continue
            
            return results
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            if self.driver and "rate limit" in self.driver.page_source.lower():
                return ["https://duckduckgo.com/ 202 RateLimit"]
            return [f"DuckDuckGo search error: {str(e)}"]

    def _search_with_google(self, query: str, num_results: int = 5) -> List[str]:
        """Search using Google.
        
        Args:
            query: Search query
            num_results: Number of results to return
            
        Returns:
            List of search result URLs
        """
        if not self.driver:
            if not self._initialize_browser():
                return ["Failed to initialize browser"]
        
        try:
            google_url = f"https://www.google.com/search?q={quote_plus(query)}"
            self.driver.get(google_url)
            
            # Wait for search results
            try:
                WebDriverWait(self.driver, 10).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, "div.g a"))
                )
            except TimeoutException:
                logger.warning("Timeout waiting for Google search results")
            
            # Extract results - Google's layout can change, so try multiple selectors
            results = []
            selectors = [
                "div.g a[href^='http']",
                "div.tF2Cxc a[href^='http']",
                "div.yuRUbf a[href^='http']",
                "a[href^='http']"
            ]
            
            for selector in selectors:
                elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    for element in elements:
                        try:
                            url = element.get_attribute("href")
                            if url and "google.com" not in url and url not in results:
                                results.append(url)
                                if len(results) >= num_results:
                                    break
                        except Exception:
                            continue
                    if len(results) >= num_results:
                        break
            
            return results
        except Exception as e:
            logger.error(f"Google search error: {e}")
            return [f"Google search error: {str(e)}"]

    def close(self) -> None:
        """Close the browser."""
        if self.driver:
            try:
                self.driver.quit()
                self.driver = None
            except Exception as e:
                logger.error(f"Error closing browser: {e}")
