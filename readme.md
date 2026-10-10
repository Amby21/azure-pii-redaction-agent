# PII Redaction Agent (Azure AI Language + Microsoft Agent Framework)

A small tool-calling agent that removes personal data from text files before anything downstream sees it.
It is a hands-on build to learn agent guardrails and harness design on Azure. All sample data is fictional.

## The key design choice

The agent is given a **file name, not the text**. The raw text goes from the file straight to the
Azure AI Language service, so the LLM never sees unredacted PII. Pasting the text into the prompt would
mean the model had already seen it before any redaction ran.

## How it works

```
prompt  ->  agent (LLM)  ->  calls redact_file(path)  ->  guards  ->  Azure AI Language (PII detection)
                                                                  ->  policy + own regex layer
                                                                  ->  verification  ->  audit log
        <-  summary + redacted text  <-  agent  ->  leak check  ->  final answer
```

## Guardrails (the rules) and harness (the scaffolding)

| Guardrails: what is allowed | Harness: what runs the model |
|---|---|
| File must be in the current folder (the model supplies the path, so it is not trusted) | Tool is wired into the agent; hard cap of 3 tool calls per run |
| Only `.txt` / `.md`; empty and oversized files rejected | Fail closed: any error returns an error, never the original text |
| Category policy: dates are kept, everything else redacted | Audit log (`pii_audit.log`): one JSON line per call, counts only, never the text |
| Own regex layer for IDs the service can mislabel (e.g. account numbers) | Instructions treat file content as data, never as commands |
| Items under 0.90 confidence are redacted but flagged for human review | |
| Verification: no detected value may remain in the output | |
| Final-answer leak check blocks answers that look like they contain PII | |

## Run it

```bash
pip install -r requirements.txt
az login
export CONTENTUNDERSTANDING_ENDPOINT="https://<your-resource>.cognitiveservices.azure.com/"
export CONTENTUNDERSTANDING_KEY="<your key>"      # terminal only, never commit it
unset OPENAI_API_KEY                               # otherwise the framework would call OpenAI, not Azure

python pii_agent.py --make-samples                 # writes email.txt and inject.txt
python pii_agent.py "Redact the file email.txt"
python pii_agent.py "Redact the file inject.txt"   # contains a prompt-injection line; the agent must ignore it
python pii_agent.py "Redact the file ../secret.txt" # blocked by the path guard
```

Optional: `AZURE_OPENAI_DEPLOYMENT` (defaults to `gpt-5-mini`).

`language_demo.py` is a plain script (not an agent) that calls language detection and PII detection directly,
including edge cases.

## What I learned from testing the service

- The PII service labelled an account number as a phone number (confidence 0.80). It was still masked, but the
  label was wrong, so the code adds its own patterns on top.
- Dates are flagged as PII by default; deciding what to keep is a business decision.
- Language detection returns one answer per document, and a two-letter input still gave 0.99 confidence:
  the score is the model's certainty, not accuracy.
- Empty input comes back as an error inside the result, not an exception.

## Limits

- Fictional data and a small demo, not a production deployment.
- The detector is a statistical model and can miss or mislabel items. Next step: a labeled test set to
  measure recall per category.
- Single-tool agent; the point is the guardrails and harness around the loop.
- A content filter on the model deployment (set in the Foundry portal) is not covered by this code.