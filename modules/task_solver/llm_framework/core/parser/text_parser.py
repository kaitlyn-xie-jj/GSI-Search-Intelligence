
import re
from typing import Dict


class CodeBlockNotFoundError(ValueError):
    """Raised when the expected fenced code block is absent."""


def parse_text(
    text: str, lang: str = "python", all_matches: bool = False
) -> str | list[str]:
    pattern = rf"```{lang}.*?\s+(.*?)```"
    matches = re.findall(pattern, text, re.DOTALL)

    if not matches:
        error_message = f"Error: No '{lang}' code block found in the text."
        raise CodeBlockNotFoundError(error_message)

    if all_matches:
        return matches
    else:
        return matches[0]
    
def parse_reasoning(text: str) -> Dict[str, str]:
    """
    Parses the LLM's output to extract the 'Reasoning' text.
    
    Args:
        text: The full output string from the LLM.
        
    Returns:
        A dictionary with 'reasoning' key.
        
    Raises:
        ValueError: If the Reasoning section is not found in the text.
    """
    # Look for reasoning section that ends with ### Result: or end of text
    reasoning_pattern = r"### Reasoning:\s*(.*?)(?:\s*### Result:|$)"
    reasoning_match = re.search(reasoning_pattern, text, re.DOTALL)
    
    if not reasoning_match:
        raise ValueError("Error: '### Reasoning:' section not found.")
    
    reasoning_text = reasoning_match.group(1).strip()
    
    return {
        "reasoning": reasoning_text
    }
