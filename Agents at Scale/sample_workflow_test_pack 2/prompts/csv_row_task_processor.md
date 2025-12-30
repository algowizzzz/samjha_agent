# CSV Task Processor — One Row = One Task

## Goal
Given a single CSV row (JSON), produce a structured response.

## Instructions
- Output **valid JSON** only.
- Return:
  - `ticket_id`
  - `recommended_ingestion_mode`: programmatic | vision
  - `recommended_output`: md | json | csv
  - `notes`: short reasoning for ops

## Output Schema
{
  "ticket_id": "...",
  "recommended_ingestion_mode": "programmatic",
  "recommended_output": "json",
  "notes": "..."
}
