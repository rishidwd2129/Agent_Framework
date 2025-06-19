# tools.py
import os
from langchain_core.tools import tool
from typing import Dict, Any
from .model import SentimentAnalyzer

# Initialize our custom sentiment analyzer
sentiment_analyzer_instance = SentimentAnalyzer()
@tool
def read_document_section(file_path: str, section_name: str) -> str:
    """
    Reads a specific section from a text document.
    The section is identified by a Markdown-style heading (e.g., '## Section Name').
    Args:
        file_path (str): The path to the document file.
        section_name (str): The name of the section to extract (e.g., "Introduction").
    Returns:
        str: The content of the specified section, or an error message if not found.
    """
    if not os.path.exists(file_path):
        return f"Error: Document not found at {file_path}"

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Simple parsing for Markdown-style sections
    # Look for '## Section Name' and extract content until the next '##' or end of file
    start_tag = f"## {section_name.strip()}"
    end_tag_pattern = r"\n## " # Next section marker

    start_index = content.find(start_tag)
    if start_index == -1:
        return f"Error: Section '{section_name}' not found in {file_path}"

    # Adjust start_index to begin after the section heading and newline
    start_index += len(start_tag) + 1 # +1 for newline after heading

    # Find the next section heading
    import re
    match = re.search(end_tag_pattern, content[start_index:])
    if match:
        end_index = start_index + match.start()
    else:
        end_index = len(content) # Go to end of file if no next section

    section_content = content[start_index:end_index].strip()
    return section_content

@tool
def analyze_text_sentiment(text: str) -> Dict[str, Any]:
    """
    Analyzes the sentiment of a given text using a custom Deep Learning model.
    Returns a dictionary with 'label' (POSITIVE/NEGATIVE/NEUTRAL) and 'score'.
    Args:
        text (str): The text content to analyze.
    Returns:
        Dict[str, Any]: A dictionary containing the sentiment label and score.
    """
    return sentiment_analyzer_instance.analyze_sentiment(text)

# List of tools available to the agent
agent_tools = [
    read_document_section,
    analyze_text_sentiment
]

if __name__ == "__main__":
    # Example usage of tools (for testing)
    sample_file = "../Doc/report.txt"
    if not os.path.exists(sample_file):
        with open(sample_file, 'w') as f:
            f.write("## Introduction\nThis is a test introduction.\n\n## Conclusion\nThis is a test conclusion.")

    intro_content = read_document_section(sample_file, "Introduction")
    print(f"Introduction Content: {intro_content}")
    print(f"Sentiment of Introduction: {analyze_text_sentiment(intro_content)}")

    conclusion_content = read_document_section(sample_file, "Conclusion")
    print(f"Conclusion Content: {conclusion_content}")
    print(f"Sentiment of Conclusion: {analyze_text_sentiment(conclusion_content)}")

    # Clean up test file if it was created
    if intro_content == "Error: Section 'Introduction' not found in report.txt" and \
       conclusion_content == "Error: Section 'Conclusion' not found in report.txt":
        os.remove(sample_file)