---
description: "Generates the LLM interaction log for the current chat session based on the template."
name: "Log Session"
agent: "agent"
---
You are an expert documentation assistant. The user wants to document the current coding session to formally record their interaction with the LLM.

1. Review the current conversation history.
2. Read the project template at `reports/llm_usage/llm_interaction_log_template.md`.
3. Read the students' info from `group_info.yaml` (Group ID is G04, Student name 1 is Matteo Parasuco Student name 2 is Igor Derosas).
4. Auto-detect the project phase we just worked on (e.g. preprocessing, feature_selection_qubo, etc.).
5. Check the `reports/llm_usage/` folder to find the next available interaction ID (e.g., if LOG-G04-01.md exists, name the new one LOG-G04-02.md).
6. Create the new `.md` file using the exact structure and YAML blocks required by the template. 

**Fill the data with these specific rules:**
- `llm_name`: "Gemini" (Default to Gemini unless specified otherwise in the conversation)
- `llm_version_or_model`: "Gemini 3.1 Pro (Preview)" (Default unless specified otherwise)
- `interaction_mode`: "IDE_assistant"
- `performed_by`: "couple"
- Put the user's most relevant prompt(s) in the `Student prompt` section.
- Put a summary of the LLM responses (or the exact code generated) in the `LLM response` section.
- State whether the generated code was used explicitly (e.g. without changes, or modified).
- Fill out the assessment scale out of 5 based on how successfully the task was completed in the chat.

Do not ask for permission, just use the `edit_file` or `create_file` tools to write directly into `reports/llm_usage/LOG-G04-XX.md`.