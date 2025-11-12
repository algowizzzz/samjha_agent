import sys, json
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv()

from external.agent.parquet_agent import ParquetQueryAgent

agent = ParquetQueryAgent()
result = agent.run_query("regional revenue", user_id="admin")

print("\n" + "="*80)
print("RESPONSE (What user sees)")
print("="*80)
print(result['final_output']['response'])

print("\n" + "="*80)
print("PROMPT MONITOR (Behind the scenes)")
print("="*80)
print(json.dumps(result['final_output']['prompt_monitor'], indent=2))
