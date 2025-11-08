# How to Run the Parquet Query Agent

## Prerequisites

1. **Virtual Environment**: Activate your virtual environment
   ```bash
   source venv/bin/activate  # or: source .venv/bin/activate
   ```

2. **Environment Variables**: Ensure `.env` file exists with LLM API key (if using LLM)
   ```bash
   ANTHROPIC_API_KEY=your_api_key_here
   ANTHROPIC_MODEL=claude-3-opus-20240229
   ```

3. **Data Files**: Ensure Parquet/CSV files are in `data/duckdb/` directory

---

## Method 1: Web UI (Recommended for Interactive Use)

### Step 1: Start the Server

```bash
cd /Users/saadahmed/queryagent/sajhamcpserver
source venv/bin/activate
python run_server.py
```

**Output:**
```
Starting SAJHA MCP Server on 0.0.0.0:8000
```

### Step 2: Access the Agent Chat UI

1. **Open browser**: `http://localhost:8000`
2. **Login**:
   - Username: `admin`
   - Password: `admin123`
3. **Navigate to**: `http://localhost:8000/agent/chat`
4. **Enter your query** in the chat input box

### Example Queries to Try

- `"sales revenue by region"`
- `"show remaining limits for all desks"`
- `"list breached limits"`
- `"top 5 desks by utilization"`

### Features

- ✅ Real-time chat interface
- ✅ Shows both **Answer** and **Prompt Monitor**
- ✅ Follow-up questions supported
- ✅ Clarification flow (if query is ambiguous)
- ✅ Session persistence

---

## Method 2: Standalone Script (For Testing/Development)

### Step 1: Edit the Query

Edit `test_agent_interactive.py`:

```python
# Line 17 - Change your query here
QUERY = "sales revenue by region"
USER_ID = "admin"

# Optional: Resume from clarification
RESUME_SESSION_ID = None  # Set if resuming
USER_CLARIFICATION = None  # Set if resuming
```

### Step 2: Run the Script

```bash
# Option A: Direct run
python test_agent_interactive.py

# Option B: Using helper script
bash run_test.sh
```

### Step 3: View Output

The script outputs:
- **Full agent state** as JSON
- **Final response** (user-friendly summary)
- **Prompt monitor** (detailed thinking process)
- **Execution stats** (timing, row counts, etc.)

### Example Output

```json
{
  "final_output": {
    "response": "Here are the results for 'sales revenue by region':\n1. desk_id: FX_DESK_A, remaining: 5000000\n...",
    "prompt_monitor": {
      "user_input": "sales revenue by region",
      "plan_explanation": "Query groups by desk and sums remaining",
      "execution_summary": {...},
      "evaluation_notes": "Results match query intent",
      "satisfaction": "satisfied"
    }
  }
}
```

---

## Method 3: API Call (For Integration)

### Step 1: Get Authentication Token

```bash
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"user_id": "admin", "password": "admin123"}'
```

**Response:**
```json
{
  "token": "your_token_here",
  "user": {...}
}
```

### Step 2: Execute Agent Query

```bash
curl -X POST http://localhost:8000/api/tools/execute \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "tool": "parquet_agent",
    "arguments": {
      "query": "sales revenue by region",
      "session_id": "optional_session_id",
      "user_id": "admin"
    }
  }'
```

### Step 3: Handle Clarification (if needed)

If the response has `"control": "wait_for_user"`, send clarification:

```bash
curl -X POST http://localhost:8000/api/tools/execute \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "tool": "parquet_agent",
    "arguments": {
      "query": "sales revenue by region",
      "session_id": "previous_session_id",
      "user_id": "admin",
      "user_clarification": "I want FX desk only"
    }
  }'
```

---

## Troubleshooting

### Issue: Server won't start

**Error**: `Address already in use`

**Solution**: Change port in `config/server.properties`:
```properties
server.port=8001
```

### Issue: LLM not working

**Error**: `RuntimeError: LLM not available`

**Solution**: 
1. Check `.env` file exists with `ANTHROPIC_API_KEY`
2. Verify API key is valid
3. Check model name in `.env`: `ANTHROPIC_MODEL=claude-3-opus-20240229`

### Issue: No data found

**Error**: Empty results or "table not found"

**Solution**:
1. Check data files exist in `data/duckdb/`
2. Verify file names match table names in `config/data_dictionary.json`
3. Check file format (CSV or Parquet)

### Issue: Agent keeps asking for clarification

**Solution**: 
1. Make query more specific (include desk, instrument, date)
2. Check `config/agent/queryagent_planner.json` - adjust `clarifying_question_threshold`
3. Review planner prompt to make it less strict

---

## Configuration Files

### Agent Configuration
- **File**: `config/agent/queryagent_planner.json`
- **Purpose**: Control agent prompts, limits, and behavior

### Data Dictionary
- **File**: `config/data_dictionary.json`
- **Purpose**: Define table schemas and business glossary

### Tool Configuration
- **File**: `config/tools/parquet_agent.json`
- **Purpose**: Register agent as MCP tool

---

## Quick Reference

| Method | Use Case | Command |
|--------|----------|---------|
| **Web UI** | Interactive queries, demos | `python run_server.py` → `http://localhost:8000/agent/chat` |
| **Script** | Testing, debugging, automation | `python test_agent_interactive.py` |
| **API** | Integration, automation, CI/CD | `curl -X POST http://localhost:8000/api/tools/execute` |

---

## Next Steps

1. **Customize for your dataset**: Update `config/data_dictionary.json` with your schema
2. **Adjust prompts**: Edit `config/agent/queryagent_planner.json` for domain-specific behavior
3. **Add data**: Place your Parquet/CSV files in `data/duckdb/`
4. **Test queries**: Try various natural language queries to validate behavior

---

## Support

For issues or questions:
- Check logs: `logs/server.log`
- Review agent state: `data/agent_state/session_*.json`
- Test LLM connection: `python -c "from agent.llm_client import get_llm_client; print(get_llm_client().is_available())"`

