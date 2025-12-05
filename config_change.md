#2 configuration files to update
external/config/data_analysis_agents/risk_agent.json
Change: "data_dict_file": "data_dictionary_risk.json" → "data_dict_file": "data_dictionary_risk_v3.json"
external/config/data_analysis_agents/mr_limits_analysis.json
Change: "data_dict_file": "data_dictionary_risk.json" → "data_dict_file": "data_dictionary_risk_v3.json"
No code changes required

#The code already supports this:
Looks for state['config_files']['data_dict_file']
Loads from: external/config/data_dictionary/{filename}
Falls back to default if not specified
Where the config is used
invoke_node() — loads tables, business_context, key_columns from the dictionary
end_node() — loads procedural_knowledge from the dictionary
nl_to_sql_planner.py — loads procedural_knowledge for SQL generation
SocketIO/API routes — pass the data_dict_file from agent config to the agent