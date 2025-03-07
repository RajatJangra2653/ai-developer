from datetime import datetime
from typing import Annotated
from semantic_kernel.functions import kernel_function


class TimePlugin:
    """Plugin that provides time-related functions."""

    @kernel_function()
    def current_time(self) -> str:
        """Get the current date and time."""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @kernel_function()
    def get_year(self, date_str: Annotated[str, "The date string in format YYYY-MM-DD"] = None) -> str:
        """Extract the year from a date string."""
        if date_str is None:
            return str(datetime.now().year)
        
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return str(date_obj.year)
        except ValueError:
            return "Invalid date format. Please use YYYY-MM-DD."

    @kernel_function()
    def get_month(self, date_str: Annotated[str, "The date string in format YYYY-MM-DD"] = None) -> str:
        """Extract the month from a date string."""
        if date_str is None:
            return datetime.now().strftime("%B")
        
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return date_obj.strftime("%B")  # Full month name
        except ValueError:
            return "Invalid date format. Please use YYYY-MM-DD."

    @kernel_function()
    def get_day_of_week(self, date_str: Annotated[str, "The date string in format YYYY-MM-DD"] = None) -> str:
        """Get the day of week for a date."""
        if date_str is None:
            return datetime.now().strftime("%A")
        
        try:
            date_obj = datetime.strptime(date_str, "%Y-%m-%d")
            return date_obj.strftime("%A")  # Full weekday name
        except ValueError:
            return "Invalid date format. Please use YYYY-MM-DD."