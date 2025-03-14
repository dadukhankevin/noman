import os
import json
import re
from cerebras.cloud.sdk import Cerebras
import os 
from dotenv import load_dotenv
import time
from tqdm import tqdm
# Load environment variables from .env file
load_dotenv()

class CriticalThinkingLLM:
    def __init__(self, api_key=None):
        self.client = Cerebras(api_key=api_key or os.getenv("CEREBRAS_API_KEY"))
        self.model = "llama3.1-8b"
    
    def extract_thinking(self, response):
        """Extract thinking blocks from response if present."""
        thinking = []
        content = response
        
        # Extract content between <think> and </think> tags
        think_pattern = r"<think>(.*?)</think>"
        think_matches = re.findall(think_pattern, response, re.DOTALL)
        
        if think_matches:
            thinking = think_matches
            # Remove thinking blocks from the content
            content = re.sub(think_pattern, "", response, flags=re.DOTALL).strip()
            
        return {
            "thinking": thinking,
            "content": content
        }
    
    def generate_completion(self, system_prompt, user_prompt, temperature=0.7, top_p=0.95, json_mode=False):
        """Generate a completion using the Groq API."""
        if json_mode:
            system_prompt += "\nYou must respond with a valid JSON object matching the format specified."
        
        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=temperature,
            top_p=top_p,
            # reasoning_format="hidden",
            max_tokens=8048,
            response_format={"type": "json_object"} if json_mode else None
        )
        
        response_text = completion.choices[0].message.content
        
        if json_mode:
            return json.loads(response_text)
        
        return response_text
    
    def generate_historical_roleplay(self, topic, year, hindsight_outcome):
        """Generate a historical roleplay with idea presentation and critique."""
        is_positive = hindsight_outcome.lower() == "positive"
        
        system_prompt = f"""
        Create a direct conversation between a curious thinker (user) and a critical analyst (assistant) in {year} discussing {topic}.
        
        Structure the response as a JSON object with:
        - "user": The initial idea/question presented in period-appropriate language
        - "assistant": The critical analysis responding directly to the user's points
        
        The analysis should:
        1. Address technical feasibility using {year} knowledge
        2. Discuss practical implementation challenges
        3. Compare with existing alternatives
        4. Explain why the idea {'succeeded' if is_positive else 'failed'} historically
        
        Use natural conversation style without titles or section headers.
        Only reference knowledge available in {year}.
        """.strip()

        user_prompt = f"""Create a conversation about {topic} in {year} where:
        - User presents an idea that {'ultimately proved valid' if is_positive else 'was later disproven'}
        - Assistant provides a detailed critique using contemporary knowledge
        
        Respond in this JSON format:
        {{
            "user": "[idea presentation]",
            "assistant": "[critical analysis]"
        }}"""

        return self.generate_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            json_mode=True
        )
    
    

    def generate_full_critical_thinking_example(self, topic, year, hindsight):
        """Generate a complete critical thinking example with historical context."""
        roleplay = self.generate_historical_roleplay(topic, year, hindsight)
        return {
            "content": roleplay
        }

def save_to_file(database, filename="historical_roleplays.jsonl", mode='w'):
    """
    Save the generated database to a JSONL file (each entry on a single line).
    
    Args:
        database: List of entries to save
        filename: Output filename
        mode: 'w' to overwrite, 'a' to append
    """
    with open(filename, mode) as f:
        for entry in database:
            f.write(json.dumps(entry) + '\n')
    if mode == 'w':
        print(f"Database saved to {filename}")

def generate_historical_database(topics, start_year, end_year, entries_per_combination=1, filename="historical_roleplays.jsonl", skip=1):
    """
    Generate a database of historical roleplay examples for given topics and time period.
    Skip parameter controls year interval (e.g. 5 = every 5 years)
    """
    llm = CriticalThinkingLLM()
    database = []
    
    # Create empty file at start to clear any previous content
    save_to_file([], filename, 'w')
    
    total_entries = len(topics) * ((end_year - start_year) // skip + 1) * entries_per_combination
    progress_bar = tqdm(total=total_entries, desc="Generating historical examples")
    
    for topic, hindsight_outcome in topics:
        for year in range(start_year, end_year + 1, skip):
            for entry_index in range(entries_per_combination):
                try:
                    roleplay = llm.generate_historical_roleplay(
                        topic=topic,
                        year=year,
                        hindsight_outcome=hindsight_outcome
                    )
                    
                    # Create unique ID
                    entry_id = f"{topic.replace(' ', '_')}_{year}_{hindsight_outcome}"
                    if entries_per_combination > 1:
                        entry_id += f"_variant_{entry_index+1}"
                    
                    database_entry = {
                        "id": entry_id,
                        "topic": topic,
                        "year": year,
                        "hindsight_outcome": hindsight_outcome,
                        "roleplay": roleplay
                    }
                    
                    database.append(database_entry)
                    
                    # Save each entry as it's generated
                    save_to_file([database_entry], filename, 'a')
                    
                    progress_bar.update(1)
                    
                    # Optional: Sleep to avoid rate limits
                    time.sleep(1)
                    
                except Exception as e:
                    tqdm.write(f"Error generating roleplay for {topic} in {year} (entry {entry_index+1}): {str(e)}")
    
    progress_bar.close()
    return database

# Example usage
if __name__ == "__main__":
    # Define generic topic pairs that work across eras
    topics = [
        ("transportation", "positive"),
        ("transportation", "negative"),
        ("communication", "positive"),
        ("communication", "negative"),
        ("energy", "positive"),
        ("energy", "negative"),
        ("agriculture", "positive"),
        ("agriculture", "negative"),
        ("medicine", "positive"),
        ("medicine", "negative"),
        ("science", "positive"),
        ("science", "negative"),
        ("technology", "positive"),
        ("technology", "negative"),
        ("economics", "positive"),
        ("economics", "negative"),
        ("politics", "positive"),
        ("politics", "negative"),
        ("warfare", "positive"),
        ("warfare", "negative"),
        ("physics", "positive"),
        ("physics", "negative"),
        ("biology", "positive"),
        ("biology", "negative"),
        ("chemistry", "positive"),
        ("chemistry", "negative"),
        ("geology", "positive"),
        ("geology", "negative"),
        ("astronomy", "positive"),
        ("astronomy", "negative"),
        ("geography", "positive"),
        ("geography", "negative"),
    ]
    
    # Generate database across wider timeframe 1700-2000
    database = generate_historical_database(
        topics=topics,
        start_year=1700,
        end_year=2015,
        entries_per_combination=1,
        skip=10,
        filename="noman.jsonl"
    )
    print(f"Generated {len(database)} historical debates across {len(topics)} topic pairs!")