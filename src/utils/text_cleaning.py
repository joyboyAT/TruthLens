"""
Text cleaning utilities for TruthLens.

Provides functions for cleaning and normalizing text data.
"""

import re
import unicodedata
from typing import Optional, List
from html import unescape


def clean_text(text: str, 
               remove_html: bool = True,
               normalize_whitespace: bool = True,
               remove_special_chars: bool = False,
               lowercase: bool = False) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Input text to clean
        remove_html: Whether to remove HTML tags
        normalize_whitespace: Whether to normalize whitespace
        remove_special_chars: Whether to remove special characters
        lowercase: Whether to convert to lowercase
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove HTML tags
    if remove_html:
        text = remove_html_tags(text)
    
    # Unescape HTML entities
    text = unescape(text)
    
    # Normalize unicode
    text = unicodedata.normalize('NFKC', text)
    
    # Normalize whitespace
    if normalize_whitespace:
        # Avoid shadowing the function name by parameter
        text = globals()["normalize_whitespace"](text)
    
    # Remove special characters
    if remove_special_chars:
        text = remove_special_characters(text)
    
    # Convert to lowercase
    if lowercase:
        text = text.lower()
    
    return text.strip()


def remove_html_tags(text: str) -> str:
    """Remove HTML tags from text."""
    # Simple HTML tag removal
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text."""
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def remove_special_characters(text: str, keep_punctuation: bool = True) -> str:
    """
    Remove special characters from text.
    
    Args:
        text: Input text
        keep_punctuation: Whether to keep basic punctuation
        
    Returns:
        Text with special characters removed
    """
    if keep_punctuation:
        # Keep letters, numbers, and basic punctuation
        pattern = r'[^a-zA-Z0-9\s.,!?;:()"\'-]'
    else:
        # Keep only letters, numbers, and spaces
        pattern = r'[^a-zA-Z0-9\s]'
    
    return re.sub(pattern, '', text)


def extract_sentences(text: str) -> List[str]:
    """
    Extract sentences from text.
    
    Args:
        text: Input text
        
    Returns:
        List of sentences
    """
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


def extract_paragraphs(text: str) -> List[str]:
    """
    Extract paragraphs from text.
    
    Args:
        text: Input text
        
    Returns:
        List of paragraphs
    """
    paragraphs = text.split('\n\n')
    return [p.strip() for p in paragraphs if p.strip()]


def normalize_text(text: str, 
                   max_length: Optional[int] = None,
                   truncate_words: bool = True) -> str:
    """
    Normalize text for processing.
    
    Args:
        text: Input text
        max_length: Maximum length (characters or words)
        truncate_words: If True, truncate by words; if False, by characters
        
    Returns:
        Normalized text
    """
    # Clean text
    text = clean_text(text)
    
    # Truncate if needed
    if max_length:
        if truncate_words:
            words = text.split()
            if len(words) > max_length:
                text = ' '.join(words[:max_length])
        else:
            if len(text) > max_length:
                text = text[:max_length]
    
    return text


def extract_urls(text: str) -> List[str]:
    """
    Extract URLs from text.
    
    Args:
        text: Input text
        
    Returns:
        List of URLs found in text
    """
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)


def extract_emails(text: str) -> List[str]:
    """
    Extract email addresses from text.
    
    Args:
        text: Input text
        
    Returns:
        List of email addresses found in text
    """
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(email_pattern, text)


def remove_duplicate_lines(text: str) -> str:
    """
    Remove duplicate lines from text.
    
    Args:
        text: Input text
        
    Returns:
        Text with duplicate lines removed
    """
    lines = text.split('\n')
    seen = set()
    unique_lines = []
    
    for line in lines:
        if line.strip() not in seen:
            seen.add(line.strip())
            unique_lines.append(line)
    
    return '\n'.join(unique_lines)


def count_words(text: str) -> int:
    """
    Count words in text.
    
    Args:
        text: Input text
        
    Returns:
        Word count
    """
    return len(text.split())


def count_characters(text: str, include_spaces: bool = True) -> int:
    """
    Count characters in text.
    
    Args:
        text: Input text
        include_spaces: Whether to include spaces in count
        
    Returns:
        Character count
    """
    if include_spaces:
        return len(text)
    else:
        return len(text.replace(' ', ''))


def get_text_statistics(text: str) -> dict:
    """
    Get comprehensive text statistics.
    
    Args:
        text: Input text
        
    Returns:
        Dictionary with text statistics
    """
    cleaned_text = clean_text(text)
    
    stats = {
        'original_length': len(text),
        'cleaned_length': len(cleaned_text),
        'word_count': count_words(cleaned_text),
        'character_count': count_characters(cleaned_text),
        'character_count_no_spaces': count_characters(cleaned_text, include_spaces=False),
        'sentence_count': len(extract_sentences(cleaned_text)),
        'paragraph_count': len(extract_paragraphs(cleaned_text)),
        'url_count': len(extract_urls(text)),
        'email_count': len(extract_emails(text)),
        'average_words_per_sentence': 0,
        'average_characters_per_word': 0
    }
    
    # Calculate averages
    if stats['sentence_count'] > 0:
        stats['average_words_per_sentence'] = stats['word_count'] / stats['sentence_count']
    
    if stats['word_count'] > 0:
        stats['average_characters_per_word'] = stats['character_count_no_spaces'] / stats['word_count']
    
    return stats
