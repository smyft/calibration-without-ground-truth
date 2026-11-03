import os
import re

def load_prompt_templates(prompt_dir):
    """
    Loads prompt templates from files in the specified directory.

    Args:
        prompt_dir (str): The directory for the prompt template set.

    Returns:
        dict: A dictionary with template names as keys and their content as values.
    """
    templates = {}
    if not os.path.isdir(prompt_dir):
        raise FileNotFoundError(f"Prompt directory not found: {prompt_dir}")

    for filename in os.listdir(prompt_dir):
        if filename.endswith('.txt'):
            template_name = filename.replace('.txt', '')
            with open(os.path.join(prompt_dir, filename), 'r', encoding='utf-8') as f:
                templates[template_name] = f.read()
    return templates

THINKING_FILTER_VERSION = "legal-options-v2"
DEFAULT_LEGAL_OPTIONS = ("A", "B", "C", "D")


def _answer_option_pattern(legal_options):
    """Preserve legacy A-D matching; newly supported letters require boundaries.

    The original filter also matches A-D inside words. That historical behavior
    is deliberately retained, while adding E must not match ordinary words.
    """
    options = tuple(str(option).upper() for option in legal_options)
    if not options or any(len(option) != 1 or not ("A" <= option <= "Z")
                          for option in options):
        raise ValueError("legal_options must contain single ASCII option letters")
    legacy = "".join(option for option in "ABCD" if option in options)
    added = "".join(dict.fromkeys(option for option in options if option not in "ABCD"))
    patterns = []
    if legacy:
        patterns.append("[A-Da-d]" if legacy == "ABCD" else f"[{legacy}{legacy.lower()}]")
    if added:
        patterns.append(r"(?<![A-Za-z])[" + added + added.lower() + r"](?![A-Za-z])")
    return "(?:" + "|".join(patterns) + ")" if len(patterns) > 1 else patterns[0]


def filter_answers_from_content(text: str, legal_options=DEFAULT_LEGAL_OPTIONS) -> str:
    """
    Helper function that applies aggressive answer filtering to any text content.
    
    Handles:
    - Answers at the beginning of text (removes answer + next sentence)
    - Answer restatements at the end (removes 2 sentences when answer found)
    - Answer statements in the middle (removes 2 sentences when answer found)
    """
    
    option_expression = _answer_option_pattern(legal_options)

    def compile_answer_pattern(pattern, flags=0):
        return re.compile(pattern.replace("[A-Da-d]", option_expression), flags)

    # Step 1: Handle answer at the beginning
    # Check if the text starts with an answer pattern
    answer_start_patterns = [
        # "The answer is A. [description]"
        r'^\s*(?:the\s+)?(?:correct|final)?\s*answer(?:\s+is)?(?:\s*:)?\s*\*?\*?\(?\s*[A-Da-d]\s*\)?\*?\*?\.?',
        # "A. [description]"
        r'^\s*\*?\*?[A-Da-d]\*?\*?\.',
        # "**Final Answer**" or similar
        r'^\s*\*\*(?:final\s+)?answer\*\*',
    ]
    
    text_after_first_answer = text
    
    for pattern in answer_start_patterns:
        match = compile_answer_pattern(pattern, re.IGNORECASE).search(text)
        if match:
            # Find where the answer statement ends
            # Strategy: Remove everything up to and including TWO sentence boundaries
            # This removes both "The answer is A." and the description after it
            
            # First, find the position after the option letter (if exists)
            option_pattern = compile_answer_pattern(r'[A-Da-d]', re.IGNORECASE)
            option_match = option_pattern.search(text[match.start():])
            
            if option_match:
                pos_after_option = match.start() + option_match.end()
                remaining = text[pos_after_option:]
                
                # Look for the FIRST sentence boundary
                boundary_pattern = re.compile(r'[.!?]\s+|(?:\n\s*){2,}')
                first_boundary = boundary_pattern.search(remaining)
                
                if first_boundary:
                    # Now look for the SECOND sentence boundary (to skip description)
                    after_first = remaining[first_boundary.end():]
                    second_boundary = boundary_pattern.search(after_first)
                    
                    if second_boundary:
                        # Remove everything up to the end of the second sentence
                        text_after_first_answer = after_first[second_boundary.end():]
                    else:
                        # Only one sentence after answer, remove it all
                        text_after_first_answer = after_first
                else:
                    # If no boundary found, the entire text might be just the answer
                    text_after_first_answer = ""
                
                break
            else:
                # Pattern matched but no option found, remove the matched part
                text_after_first_answer = text[match.end():].lstrip()
                break
    
    text = text_after_first_answer
    
    # If the entire text was just an answer at the beginning, return empty
    if not text.strip():
        return ""
    
    # Step 2: Split into sentences
    abbr_len2 = r"(?:Mr|Ms|Dr|vs|Jr|Sr|Co)"
    abbr_len3 = r"(?:Mrs|etc|Inc|Ltd|e\.g|i\.e)"
    abbr_len4 = r"(?:Prof)"
    
    split_regex = fr'(?<!\b{abbr_len2})(?<!\b{abbr_len3})(?<!\b{abbr_len4})[.|!|?](?=\s+|$)'
    split_pattern = re.compile(split_regex, re.IGNORECASE)
    
    temp_text = split_pattern.sub(lambda m: m.group() + "<SPLIT_TOKEN>", text.strip())
    sentences = [s.strip() for s in temp_text.split("<SPLIT_TOKEN>") if s.strip()]
    
    # Aggressive answer detection patterns
    aggressive_patterns = [
        # Any mention of "answer" with an option letter nearby
        compile_answer_pattern(r'answer.*?[A-Da-d]', re.DOTALL | re.IGNORECASE),
        compile_answer_pattern(r'[A-Da-d].*?answer', re.DOTALL | re.IGNORECASE),
        
        # Standalone option letters (possibly in parentheses or bold)
        compile_answer_pattern(r'(?:^|\s)(?:\([A-Da-d]\)|\[[A-Da-d]\]|[A-Da-d]\.|\*\*[A-Da-d]\*\*|[^A-Za-z][A-Da-d]\.)(?=\s|$)', re.IGNORECASE),
        
        # Concluding phrases with answer
        re.compile(r'(?:therefore|thus|hence|so),?\s+(?:the\s+)?(?:correct\s+)?(?:answer|option|choice)', re.IGNORECASE),
        
        # "The correct X is"
        re.compile(r'the\s+correct\s+(?:answer|option|choice)\s+(?:is|should\s+be)', re.IGNORECASE),
        
        # Boxed notation
        re.compile(r'\\boxed', re.IGNORECASE),
        
        # Final answer markers
        re.compile(r'\*\*(?:final\s+)?answer\*\*', re.IGNORECASE),
        
        # Lines starting with option letters
        compile_answer_pattern(r'^\s*\*?\*?[A-Da-d]\*?\*?\.', re.IGNORECASE),
        
        # "So the answer is/should be"
        re.compile(r'so\s+(?:the\s+)?answer\s+(?:is|should\s+be)', re.IGNORECASE),
    ]
    
    # Step 3: Remove answer sentences from the end (work backwards)
    # Remove TWO sentences when an answer is found at the end
    # Find the last occurrence of an answer pattern
    last_answer_index = -1
    for i in range(len(sentences) - 1, -1, -1):
        sent = sentences[i]
        is_answer = False
        
        for pattern in aggressive_patterns:
            if pattern.search(sent):
                is_answer = True
                break
        
        if is_answer:
            last_answer_index = i
            break
    
    # If we found an answer at the end, remove it and the next sentence
    if last_answer_index >= 0:
        # Remove up to 2 sentences starting from the answer
        sentences_to_keep = last_answer_index
        sentences = sentences[:sentences_to_keep]
    
    # Step 4: Remove answer sentences from the middle
    # Remove TWO sentences whenever an answer pattern is found
    cleaned_sentences = []
    
    # More selective patterns for middle content
    middle_answer_patterns = [
        # Direct answer statements
        compile_answer_pattern(r'(?:the\s+)?(?:correct\s+)?answer\s+(?:is|should\s+be)\s+.*?[A-Da-d]', re.IGNORECASE),
        
        # Boxed notation
        compile_answer_pattern(r'\\boxed\{[A-Da-d]\}', re.IGNORECASE),
        
        # "Therefore/Thus, [answer is] X"
        compile_answer_pattern(r'(?:therefore|thus|hence),?\s+.*?(?:answer|option).*?[A-Da-d]', re.IGNORECASE),
        
        # "So the answer is X" or "So, answer is X"
        compile_answer_pattern(r'so,?\s+(?:the\s+)?answer\s+(?:is|should\s+be)\s+[A-Da-d]', re.IGNORECASE),
        
        # Lines that are just option letters
        compile_answer_pattern(r'^\s*\*?\*?[A-Da-d]\*?\*?\..*$', re.IGNORECASE),
    ]
    
    i = 0
    while i < len(sentences):
        sent = sentences[i]
        is_answer = False
        
        for pattern in middle_answer_patterns:
            if pattern.search(sent):
                is_answer = True
                break
        
        if is_answer:
            # Skip this sentence and the next one (regardless of what it is)
            i += 2  # Skip both the answer and the description
        else:
            cleaned_sentences.append(sent)
            i += 1
    
    # Step 5: Final cleanup
    result = " ".join(cleaned_sentences).strip()
    
    return result

def extract_thinking_process(text: str, legal_options=DEFAULT_LEGAL_OPTIONS) -> str:
    """
    Aggressively extracts the thinking process from the Model's output.
    
    1. If explicit tags <think>...</think> are present, extract content and filter answers.
    2. If no tags:
       - If answer appears in first sentence, remove it AND the next sentence (description)
       - Remove answer restatements at the end AND the following sentence
       - Aggressively remove all other answer-related content
       
    This version handles:
    - Answers within <think> tags
    - Answers at the beginning: "The answer is A. [descriptions]. [thinking process...]"
    - Answer restatements at the end (removes 2 sentences)
    - All intermediate answer mentions (removes 2 sentences)
    """
    
    THINK_START_TAG = "<think>"
    THINK_END_TAG = "</think>"
    
    # --- Strategy 1: Explicit Tags ---
    
    # 1a. Check for complete pair: <think> ...Content... </think>
    complete_pattern = re.compile(
        f"{re.escape(THINK_START_TAG)}(.*?){re.escape(THINK_END_TAG)}", 
        re.DOTALL | re.IGNORECASE
    )
    match_complete = complete_pattern.search(text)
    if match_complete:
        think_content = match_complete.group(1).strip()
        # Apply answer filtering to the think content
        return filter_answers_from_content(think_content, legal_options)
    
    # 1b. Check for unclosed tag: <think> ...Content (End of Text)
    incomplete_pattern = re.compile(
        f"{re.escape(THINK_START_TAG)}(.*)$",
        re.DOTALL | re.IGNORECASE
    )
    match_incomplete = incomplete_pattern.search(text)
    if match_incomplete:
        think_content = match_incomplete.group(1).strip()
        # Apply answer filtering to the think content
        return filter_answers_from_content(think_content, legal_options)
    
    # --- Strategy 2: No tags - apply standard filtering ---
    return filter_answers_from_content(text, legal_options)
