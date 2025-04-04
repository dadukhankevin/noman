![image](https://github.com/user-attachments/assets/4fbf488b-9497-41bf-b9f9-117b642685d2)

# Noman

Noman is a tool that generates a historical database of ideas—both successful and unsuccessful—to train LLMs in critical evaluation. By using history as validation, it creates a conversations over concepts that were either confirmed or refuted over time. Each entry simulates a conversation between a user and an assistant, restricted to using only the knowledge available at that specific historical moment. This approach aims to capture valuable data: sound reasoning within historical constraints that can be generalized to present-day scenarios.

## How it works

1. The system generates historical roleplays across different topics and time periods:
   * A curious thinker (user) presents an idea that either ultimately proved valid ("positive") or was later disproven ("negative")
   * A critical analyst (assistant) evaluates the idea using only knowledge available at that time

2. Each entry includes technical feasibility analysis, implementation challenges, and comparison with alternatives

3. The historical outcomes serve as ground truth for whether ideas were good or bad - which tell the LLM to be critical or positive about the idea

4. All conversations are saved to a JSONL database that can be used to train LLMs to better evaluate ideas

## Installation

Clone the repo

## Usage

Be sure to customize it to whatever LLM provider you are using! Cerebras or Groq is recommended for speed.

```
python3 llm.py
```

Also will add a database to huggingface soon ideally.
