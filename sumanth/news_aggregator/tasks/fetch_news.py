import os
import json
# import openai
from prompt_orchestrator import generate_user_prompt_json

# Set your OpenAI API key (recommended: use environment variable)
# openai.api_key = os.getenv("OPENAI_API_KEY", "sk-...")  # Replace with your key or set env var

def format_prompt_for_llm(prompt_json):
    """
    Convert the prompt JSON into a string suitable for the LLM.
    """
    segments = prompt_json.get("segments", [])
    prompt_lines = []
    for seg in segments:
        prompt_lines.append(seg["prompt"])
    return "\n".join(prompt_lines)

def fetch_news_from_gpt():
    # Get the prompt JSON from your orchestrator
    prompt_json = generate_user_prompt_json()
    # "Prompt JSON:", 
    print(json.dumps(prompt_json, indent=2))
    # Format the prompt for the LLM
    prompt_text = "Here is the user profile:"
    prompt_text += format_prompt_for_llm(prompt_json)
    prompt_text += "Generate a brief but complete news summary that is most relevant to this user. Cover top updates from Hyderabad, Telangana, and India. If any major global issue affects this region, mention it concisely."
    # print("\nPrompt sent to GPT-4o Mini:\n", prompt_text)

    return prompt_text

    # Call OpenAI API (GPT-4o Mini)
    # response = openai.chat.completions.create(
    #     model="gpt-4o",  # or "gpt-4o-mini" if available
    #     messages=[
    #         {"role": "system", "content": "You are a smart news assistant. Given a user's location details, generate a short, context-rich news. Keep it concise, relevant, and informative."},
    #         {"role": "user", "content": prompt_text}
    #     ],
    #     max_tokens=800,
    #     temperature=0.7
    # )

    # Print the response
    # print("\nGPT-4o Mini Response:\n", response.choices[0].message.content)

if __name__ == "__main__":
    fetch_news_from_gpt()