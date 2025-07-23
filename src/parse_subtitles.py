import re
from typing import List, Dict, Tuple
import os

class SubtitleParser:
    """
    A class to parse subtitle files (.srt format) and convert them into chunks
    suitable for RAG (Retrieval-Augmented Generation) applications.
    """
    
    def __init__(self, chunk_size: int = 5, overlap: int = 1):
        """
        Initialize the parser.
        
        Args:
            chunk_size (int): Number of subtitle entries per chunk
            overlap (int): Number of overlapping entries between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def parse_srt_file(self, file_path: str) -> List[Dict]:
        """
        Parse an SRT subtitle file and extract subtitle entries.
        
        Args:
            file_path (str): Path to the .srt file
            
        Returns:
            List[Dict]: List of subtitle entries with timestamp and text
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Subtitle file not found: {file_path}")
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
        
        # Split by double newlines to separate subtitle blocks
        blocks = re.split(r'\n\s*\n', content.strip())
        
        subtitles = []
        
        for block in blocks:
            if not block.strip():
                continue
                
            lines = block.strip().split('\n')
            
            if len(lines) < 3:
                continue
            
            # First line is the subtitle number
            try:
                subtitle_num = int(lines[0].strip())
            except ValueError:
                continue
            
            # Second line is the timestamp
            timestamp = lines[1].strip()
            
            # Remaining lines are the subtitle text
            text = ' '.join(lines[2:]).strip()
            
            # Clean up the text (remove HTML tags, extra spaces)
            text = self._clean_text(text)
            
            if text:  # Only add if there's actual text
                subtitles.append({
                    'number': subtitle_num,
                    'timestamp': timestamp,
                    'text': text,
                    'start_time': self._parse_timestamp(timestamp.split(' --> ')[0]),
                    'end_time': self._parse_timestamp(timestamp.split(' --> ')[1])
                })
        
        return subtitles
    
    def _clean_text(self, text: str) -> str:
        """
        Clean subtitle text by removing HTML tags and formatting.
        
        Args:
            text (str): Raw subtitle text
            
        Returns:
            str: Cleaned text
        """
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove formatting characters
        text = re.sub(r'[♪♫♬]', '', text)  # Music notes
        text = re.sub(r'\[.*?\]', '', text)  # Text in brackets like [MUSIC]
        text = re.sub(r'\(.*?\)', '', text)  # Text in parentheses like (LAUGHING)
        
        # Clean up whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _parse_timestamp(self, timestamp: str) -> float:
        """
        Convert timestamp string to seconds.
        
        Args:
            timestamp (str): Timestamp in format "HH:MM:SS,mmm"
            
        Returns:
            float: Time in seconds
        """
        try:
            # Handle both comma and dot as millisecond separator
            timestamp = timestamp.replace(',', '.')
            
            # Split into time parts
            time_parts = timestamp.split(':')
            hours = int(time_parts[0])
            minutes = int(time_parts[1])
            seconds_and_ms = float(time_parts[2])
            
            total_seconds = hours * 3600 + minutes * 60 + seconds_and_ms
            return total_seconds
        except (ValueError, IndexError):
            return 0.0
    
    def create_chunks(self, subtitles: List[Dict]) -> List[Dict]:
        """
        Group subtitles into chunks for better retrieval.
        
        Args:
            subtitles (List[Dict]): List of parsed subtitles
            
        Returns:
            List[Dict]: List of subtitle chunks
        """
        chunks = []
        
        for i in range(0, len(subtitles), self.chunk_size - self.overlap):
            chunk_subtitles = subtitles[i:i + self.chunk_size]
            
            if not chunk_subtitles:
                continue
            
            # Combine text from all subtitles in the chunk
            chunk_text = ' '.join([sub['text'] for sub in chunk_subtitles])
            
            # Get time range for the chunk
            start_time = chunk_subtitles[0]['start_time']
            end_time = chunk_subtitles[-1]['end_time']
            start_timestamp = chunk_subtitles[0]['timestamp'].split(' --> ')[0]
            end_timestamp = chunk_subtitles[-1]['timestamp'].split(' --> ')[1]
            
            chunk = {
                'chunk_id': len(chunks),
                'text': chunk_text,
                'start_time': start_time,
                'end_time': end_time,
                'timestamp_range': f"{start_timestamp} --> {end_timestamp}",
                'subtitle_count': len(chunk_subtitles),
                'subtitle_numbers': [sub['number'] for sub in chunk_subtitles]
            }
            
            chunks.append(chunk)
        
        return chunks
    
    def process_subtitle_file(self, file_path: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Complete processing pipeline: parse file and create chunks.
        
        Args:
            file_path (str): Path to the subtitle file
            
        Returns:
            Tuple[List[Dict], List[Dict]]: (raw_subtitles, chunks)
        """
        print(f"Processing subtitle file: {file_path}")
        
        # Parse the subtitle file
        subtitles = self.parse_srt_file(file_path)
        print(f"Found {len(subtitles)} subtitle entries")
        
        # Create chunks
        chunks = self.create_chunks(subtitles)
        print(f"Created {len(chunks)} chunks")
        
        return subtitles, chunks
    
    def save_chunks_to_text(self, chunks: List[Dict], output_path: str):
        """
        Save chunks to a text file for inspection.
        
        Args:
            chunks (List[Dict]): List of subtitle chunks
            output_path (str): Path to save the text file
        """
        with open(output_path, 'w', encoding='utf-8') as file:
            for chunk in chunks:
                file.write(f"=== CHUNK {chunk['chunk_id']} ===\n")
                file.write(f"Time: {chunk['timestamp_range']}\n")
                file.write(f"Subtitles: {chunk['subtitle_numbers']}\n")
                file.write(f"Text: {chunk['text']}\n")
                file.write("\n" + "="*50 + "\n\n")


# Example usage and testing functions
def test_parser():
    """Test function to demonstrate the parser usage."""
    parser = SubtitleParser(chunk_size=5, overlap=1)
    
    # Example of how to use the parser
    try:
        # Replace 'data/sample.srt' with your actual subtitle file
        subtitles, chunks = parser.process_subtitle_file('data/Little.Miss.Sunshine.2006.720p.BluRay.x264.YIFY-en.srt')
        
        # Print first few chunks for inspection
        print("\n=== FIRST 3 CHUNKS ===")
        for i, chunk in enumerate(chunks[:3]):
            print(f"\nChunk {i}:")
            print(f"Time: {chunk['timestamp_range']}")
            print(f"Text: {chunk['text'][:200]}...")  # First 200 characters
        
        # Save chunks to file for inspection
        parser.save_chunks_to_text(chunks, 'data/processed_chunks.txt')
        print(f"\nSaved {len(chunks)} chunks to 'data/processed_chunks.txt'")
        
        return subtitles, chunks
        
    except FileNotFoundError:
        print("Please add a subtitle file (.srt) to the 'data/' folder and update the filename in this function.")
        return [], []


if __name__ == "__main__":
    # Run the test when file is executed directly
    test_parser()
