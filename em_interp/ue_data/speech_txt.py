# <file_context>
# em_interp/ue_data/speech_txt.py
# Utilities for processing transcript data with nested structure
# Handles format: {"transcript": {"0": "text segment", "1": "next segment", ...}}
# </file_context>

import json
import pandas as pd
import re
from typing import Dict, List, Union, Optional

def load_transcript_data(file_path: str) -> Dict:
    """
    Load transcript data from JSON file.
    
    Args:
        file_path: Path to JSON file containing transcript data
        
    Returns:
        Dictionary containing transcript data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def extract_transcript_segments(transcript_data: Dict) -> List[str]:
    """
    Extract transcript segments from nested structure.
    
    Args:
        transcript_data: Dictionary with format {"transcript": {"0": "text", "1": "text", ...}}
        
    Returns:
        List of transcript segments in order
    """
    if 'transcript' not in transcript_data:
        raise ValueError("No 'transcript' key found in data")
    
    transcript_dict = transcript_data['transcript']
    
    # Sort by numeric keys to maintain order
    sorted_keys = sorted(transcript_dict.keys(), key=lambda x: int(x))
    segments = [transcript_dict[key] for key in sorted_keys]
    
    return segments

def join_transcript_segments(transcript_data: Dict, separator: str = " ") -> str:
    """
    Join all transcript segments into a single string.
    
    Args:
        transcript_data: Dictionary with transcript segments
        separator: String to join segments with (default: single space)
        
    Returns:
        Complete transcript as single string
    """
    segments = extract_transcript_segments(transcript_data)
    return separator.join(segments)

def process_transcript_file(file_path: str, output_format: str = "combined") -> Union[str, List[str], pd.DataFrame]:
    """
    Process a transcript file and return in specified format.
    
    Args:
        file_path: Path to JSON file with transcript data
        output_format: "combined" (single string), "segments" (list), or "dataframe"
        
    Returns:
        Processed transcript in requested format
    """
    data: Dict = load_transcript_data(file_path)
    
    if output_format == "combined":
        return join_transcript_segments(data)
    elif output_format == "segments":
        return extract_transcript_segments(data)
    elif output_format == "dataframe":
        segments = extract_transcript_segments(data)
        df = pd.DataFrame({
            'segment_id': range(len(segments)),
            'text': segments
        })
        return df
    else:
        raise ValueError("output_format must be 'combined', 'segments', or 'dataframe'")

def process_multiple_transcripts(file_paths: List[str]) -> pd.DataFrame:
    """
    Process multiple transcript files into a single DataFrame.
    
    Args:
        file_paths: List of paths to transcript JSON files
        
    Returns:
        DataFrame with columns: file_path, segment_id, text, full_transcript
    """
    all_data = []
    
    for file_path in file_paths:
        try:
            data: Dict = load_transcript_data(file_path)
            segments = extract_transcript_segments(data)
            full_transcript = join_transcript_segments(data)
            
            for i, segment in enumerate(segments):
                all_data.append({
                    'file_path': file_path,
                    'segment_id': i,
                    'text': segment,
                    'full_transcript': full_transcript
                })
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
    
    return pd.DataFrame(all_data)

def clean_transcript_text(text: str) -> str:
    """
    Clean transcript text by removing common artifacts.
    
    Args:
        text: Raw transcript text
        
    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove common transcript artifacts (customize as needed)
    artifacts = ['[inaudible]', '[unintelligible]', '[crosstalk]', '(inaudible)', '(unintelligible)']
    for artifact in artifacts:
        text = text.replace(artifact, '')
    
    # Clean up extra spaces after removal
    text = ' '.join(text.split())
    
    return text.strip()

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using improved regex pattern for speech transcripts.
    
    Args:
        text: Input text to split
        
    Returns:
        List of sentences
    """
    # Clean the text first
    text = clean_transcript_text(text)
    
    # Improved sentence splitting for speech transcripts
    # Split on various sentence endings, including those without proper capitalization
    patterns = [
        r'(?<=[.!?])\s+(?=[A-Z])',  # Standard: period/!/? followed by space and capital
        r'(?<=[.!?])\s+(?=[a-z])',  # Non-standard: period/!/? followed by space and lowercase
        r'(?<=\.)\s+(?=and\b)',     # Split before "and" after periods
        r'(?<=\.)\s+(?=but\b)',     # Split before "but" after periods
        r'(?<=\.)\s+(?=so\b)',      # Split before "so" after periods
        r'(?<=\.)\s+(?=then\b)',    # Split before "then" after periods
        r'(?<=\.)\s+(?=well\b)',    # Split before "well" after periods
        r'(?<=\.)\s+(?=now\b)',     # Split before "now" after periods
    ]
    
    sentences = [text]
    for pattern in patterns:
        new_sentences = []
        for sentence in sentences:
            new_sentences.extend(re.split(pattern, sentence))
        sentences = new_sentences
    
    # Also split on long pauses or natural breaks (comma + long phrase)
    sentences_final = []
    for sentence in sentences:
        # Split very long sentences (over 150 chars) at natural breaks
        if len(sentence) > 150:
            # Split at commas followed by conjunctions
            parts = re.split(r',\s+(?=and\b|but\b|so\b|because\b|when\b|if\b|while\b)', sentence)
            sentences_final.extend(parts)
        else:
            sentences_final.append(sentence)
    
    # Filter out empty sentences and very short ones (less than 15 chars for speech)
    sentences = [s.strip() for s in sentences_final if s.strip() and len(s.strip()) >= 15]
    
    return sentences

def chunk_sentences(sentences: List[str], chunk_size: int = 8) -> List[str]:
    """
    Group sentences into chunks of exactly chunk_size sentences.
    
    Args:
        sentences: List of sentences
        chunk_size: Number of sentences per chunk (default: 8)
        
    Returns:
        List of text chunks, each containing exactly chunk_size sentences (except possibly the last)
    """
    chunks = []
    for i in range(0, len(sentences), chunk_size):
        chunk = sentences[i:i + chunk_size]
        if len(chunk) >= chunk_size or i + chunk_size >= len(sentences):  # Include partial last chunk
            chunks.append(' '.join(chunk))
    
    return chunks

# Example usage functions
def example_usage():
    """Example of how to use the transcript processing functions."""
    
    # Example data structure
    example_data = {
        "transcript": {
            "0": "thank you very much for joining us today",
            "1": "we're here to discuss the important topic",
            "2": "of artificial intelligence safety and alignment"
        }
    }
    
    print("Example transcript data:")
    print(json.dumps(example_data, indent=2))
    
    # Extract segments
    segments = extract_transcript_segments(example_data)
    print(f"\nSegments: {segments}")
    
    # Join into full transcript
    full_text = join_transcript_segments(example_data)
    print(f"\nFull transcript: {full_text}")
    
    # Create DataFrame
    df = pd.DataFrame({
        'segment_id': range(len(segments)),
        'text': segments
    })
    print(f"\nDataFrame:\n{df}")

def process_trump_data(input_file: str = "Trump_Labeled_Combined_Rev_Speeches_Final_9-24-2024.json", 
                      output_file: str = "trump_processed_6k.jsonl", 
                      max_datapoints: int = 6000,
                      sentences_per_chunk: int = 8) -> None:
    """
    Process Trump transcript data and create JSONL file with up to max_datapoints.
    Each text entry contains exactly sentences_per_chunk sentences.
    
    Args:
        input_file: Path to input JSON file with Trump transcript data
        output_file: Path to output JSONL file
        max_datapoints: Maximum number of datapoints to include (default: 6000)
        sentences_per_chunk: Number of sentences per text entry (default: 8)
    """
    print(f"Loading Trump transcript data from {input_file}...")
    print(f"Target: {sentences_per_chunk} sentences per chunk")
    
    try:
        # Load the data
        data = load_transcript_data(input_file)
        
        # Extract segments
        segments = extract_transcript_segments(data)
        print(f"Found {len(segments)} total segments")
        
        # Process all segments into sentences and chunks
        all_chunks = []
        total_sentences = 0
        chunk_sentence_counts = []
        
        for segment_idx, segment in enumerate(segments):
            # Split segment into sentences
            sentences = split_into_sentences(segment)
            total_sentences += len(sentences)
            
            # Group sentences into chunks of exactly sentences_per_chunk
            chunks = chunk_sentences(sentences, sentences_per_chunk)
            
            # Add metadata to each chunk and track sentence count
            for chunk_idx, chunk_text in enumerate(chunks):
                # Count actual sentences in this chunk
                chunk_sentences_count = len(split_into_sentences(chunk_text))
                chunk_sentence_counts.append(chunk_sentences_count)
                
                all_chunks.append({
                    "text": chunk_text,
                    "segment_id": segment_idx,
                    "chunk_id": chunk_idx,
                    "sentence_count": chunk_sentences_count,
                    "source": "trump_speech"
                })
            
            # Progress update for segments
            if (segment_idx + 1) % 50 == 0:
                print(f"Processed {segment_idx + 1} segments, found {len(all_chunks)} chunks so far...")
        
        print(f"Total sentences found: {total_sentences}")
        print(f"Total chunks created: {len(all_chunks)}")
        
        # Print sentence count statistics
        if chunk_sentence_counts:
            avg_sentences = sum(chunk_sentence_counts) / len(chunk_sentence_counts)
            print(f"Average sentences per chunk: {avg_sentences:.2f}")
            print(f"Chunks with exactly {sentences_per_chunk} sentences: {chunk_sentence_counts.count(sentences_per_chunk)}/{len(chunk_sentence_counts)}")
        
        # Limit to max_datapoints
        chunks_to_write = all_chunks[:max_datapoints]
        print(f"Writing first {len(chunks_to_write)} chunks to file")
        
        # Write to JSONL file
        with open(output_file, 'w', encoding='utf-8') as f:
            for i, entry in enumerate(chunks_to_write):
                # Write as JSONL
                f.write(json.dumps(entry) + '\n')
                
                # Progress update
                if (i + 1) % 1000 == 0:
                    print(f"Written {i + 1} chunks...")
        
        print(f"Successfully created {output_file} with {len(chunks_to_write)} datapoints")
        print(f"Target: {sentences_per_chunk} sentences per chunk")
        
        # Final statistics on written chunks
        written_sentence_counts = [chunk["sentence_count"] for chunk in chunks_to_write]
        if written_sentence_counts:
            avg_written = sum(written_sentence_counts) / len(written_sentence_counts)
            exact_count = written_sentence_counts.count(sentences_per_chunk)
            print(f"Written chunks - Average sentences: {avg_written:.2f}")
            print(f"Written chunks with exactly {sentences_per_chunk} sentences: {exact_count}/{len(written_sentence_counts)} ({100*exact_count/len(written_sentence_counts):.1f}%)")
        
    except Exception as e:
        print(f"Error processing Trump data: {e}")
        raise

if __name__ == "__main__":
    # Run Trump data processing
    process_trump_data()
    
    # Original example usage
    print("\n" + "="*50)
    print("Example usage:")
    example_usage()
