# Abhikarta - New PC Setup Guide

## Quick Start

```bash
# 1. Clone & checkout
git clone https://github.com/algowizzzz/samjha_agent.git
cd samjha_agent
git checkout saad-full-backup-20260105

# 2. Create .env file
cat > .env << EOF
ANTHROPIC_API_KEY=your-anthropic-key-here
TAVILY_API_KEY=your-tavily-key-here
DATABASE_URL=postgresql+psycopg2://$(whoami)@localhost:5432/samjha_agent
EOF

# 3. Setup PostgreSQL & restore data
createdb samjha_agent
psql samjha_agent < db_backup/samjha_agent_full.sql

# 4. Python environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 5. Run server
python run_server.py
```

Server runs at: http://localhost:8000

---

## Prerequisites

| Requirement | Installation |
|-------------|--------------|
| Python 3.10+ | https://python.org |
| PostgreSQL | Mac: `brew install postgresql && brew services start postgresql` |
| | Ubuntu: `sudo apt install postgresql` |
| | Windows: https://postgresql.org/download/windows/ |

---

## Environment Variables (.env)

| Variable | Required | Description |
|----------|----------|-------------|
| `ANTHROPIC_API_KEY` | ✅ Yes | Claude API key from https://console.anthropic.com |
| `TAVILY_API_KEY` | ✅ Yes | Web search API from https://tavily.com |
| `DATABASE_URL` | ✅ Yes | PostgreSQL connection string |

### Example .env
```
ANTHROPIC_API_KEY=sk-ant-api03-xxxxx
TAVILY_API_KEY=tvly-xxxxx
DATABASE_URL=postgresql+psycopg2://myuser@localhost:5432/samjha_agent
```

---

## Database

### Why PostgreSQL?
The repo includes a full PostgreSQL dump with all historical data:
- 6 agents (structured + web research)
- 260 runs with full execution history
- 187 conversations
- 480 messages
- 12 system prompts
- 33 chains, 4 workflows
- 62 documents, 50 ingestion profiles

### Restore Commands
```bash
# Create database
createdb samjha_agent

# Restore from dump
psql samjha_agent < db_backup/samjha_agent_full.sql

# Verify
psql samjha_agent -c "SELECT COUNT(*) FROM runs;"
# Should show: 260
```

### SQLite Fallback
If `DATABASE_URL` is not set, the app falls back to SQLite (`data/app.db`).
⚠️ SQLite version has minimal data - use PostgreSQL for full experience.

---

## Verify Installation

```bash
# Check server starts
python run_server.py

# Should see:
# INFO - Starting SAJHA MCP Server on 0.0.0.0:8000
# INFO - Tool registered: execute_sql
# INFO - Tool registered: tavily_web_search
# ... (no errors)
```

### Test in Browser
1. Go to http://localhost:8000
2. Login: `admin` / `admin123`
3. Try Chat Agent → ecommerce_advanced
4. Query: "top products by revenue"

---

## Troubleshooting

### "database does not exist"
```bash
createdb samjha_agent
```

### "role does not exist"
```bash
# Create PostgreSQL user matching your OS username
createuser -s $(whoami)
```

### "execute_sql tool not found"
Server needs restart after code changes:
```bash
pkill -f run_server && python run_server.py
```

### API key errors
Check `.env` file exists and has valid keys.

---

## Project Structure

```
samjha_agent/
├── external/           # Custom code (agents, routes, tools)
├── core/               # Base framework
├── web/                # Flask app & templates
├── config/             # Tool & agent configs
├── data/               # Runtime data (SQLite, uploads)
├── db_backup/          # PostgreSQL dump ← RESTORE THIS
├── .env                # API keys (create this)
└── run_server.py       # Entry point
```

---

## Agents Available

| Agent | Type | Description |
|-------|------|-------------|
| ecommerce_advanced | Structured | SQL queries on sales/inventory data |
| Ecommerce Agent | Structured | Basic ecommerce queries |
| Widget Sales Agent | Structured | Widget sales analysis |
| Financial News Agent | Web Research | Searches SEC, Bloomberg, Reuters |

---

## Support

Branch: `saad-full-backup-20260105`
Last updated: January 5, 2026

