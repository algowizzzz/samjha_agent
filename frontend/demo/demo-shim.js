// DSH-UI demo shim: seeds a super-admin session and mocks the agent-server API
// so the real admin.html / mcp-agent.html render with sample data. Demo only.
(function () {
  var USER = {
    user_id: 'u-durga', display_name: 'Durga V', avatar_initials: 'DV',
    role: 'super_admin', worker_id: 'w-ccr-triage', worker_name: 'CCR Breach Triage',
    agent_runtime: 'dsh'
  };
  sessionStorage.setItem('rg_token', 'demo-token');
  sessionStorage.setItem('rg_user', JSON.stringify(USER));

  var WORKERS = [
    {
      worker_id: 'w-ccr-triage', name: 'CCR Breach Triage',
      description: 'Counterparty credit-limit breach triage with verified workflows.',
      enabled: true, agent_mode: 'multi', agent_runtime: 'dsh',
      dsh: { tools_mode: 'code', sandbox_mode: 'read-only', enable_shell: false, enable_web: false, enable_fs: false },
      dsh_status: 'running', sajha_api_key_masked: 'sja_wrk_ccr_????????????9f2b',
      system_prompt: 'You are the CCR Breach Triage worker. Use workflow_get to load the verified breach-triage workflow, then execute its steps in order. Use customer_olap_pivot and duckdb_query for exposure data. Present findings in canvas mode.',
      enabled_tools: ['workflow_get', 'customer_olap_pivot', 'duckdb_query', 'document_search', 'python_run_script'],
      max_concurrent_subagents: 3, user_count: 4, admin_count: 1
    },
    {
      worker_id: 'w-market-risk', name: 'Market Risk Worker',
      description: 'Counterparty credit risk and market intelligence for capital markets analysts.',
      enabled: true, agent_mode: 'single', agent_runtime: 'langchain',
      system_prompt: 'You are the Market Risk digital worker…',
      enabled_tools: ['yahoo_get_quote', 'duckdb_query', 'document_search'],
      user_count: 12, admin_count: 2
    },
    {
      worker_id: 'w-finance-agent', name: 'Finance Agent',
      description: 'Financial reporting, EDGAR filings and ratio analysis.',
      enabled: true, agent_mode: 'multi', agent_runtime: 'langchain',
      system_prompt: 'You are the Finance Agent…',
      enabled_tools: ['edgar_search', 'duckdb_query'], max_concurrent_subagents: 5,
      user_count: 7, admin_count: 1
    }
  ];

  function find(wid) { for (var i = 0; i < WORKERS.length; i++) if (WORKERS[i].worker_id === wid) return WORKERS[i]; return null; }
  function json(body, status) {
    return Promise.resolve(new Response(JSON.stringify(body), {
      status: status || 200, headers: { 'Content-Type': 'application/json' }
    }));
  }

  var realFetch = window.fetch.bind(window);
  window.fetch = function (url, opts) {
    var u = String(url); opts = opts || {};
    var m;
    if (/\/api\/(super\/)?workers\/?$/.test(u) && (!opts.method || opts.method === 'GET'))
      return json({ workers: WORKERS });
    if ((m = u.match(/\/api\/super\/workers\/([^/?]+)$/))) {
      var w = find(decodeURIComponent(m[1]));
      if (!w) return json({ detail: 'not found' }, 404);
      if (opts.method === 'PUT') {
        var body = {}; try { body = JSON.parse(opts.body || '{}'); } catch (e) {}
        for (var k in body) w[k] = body[k];
        if (w.agent_runtime === 'dsh') {
          w.dsh_status = 'running';
          w.sajha_api_key_masked = w.sajha_api_key_masked || 'sja_wrk_' + w.worker_id.slice(2, 5) + '_????????????' + Math.random().toString(16).slice(2, 6);
        }
        return json(w);
      }
      return json(w);
    }
    if (u.indexOf('/api/super/users') !== -1 || u.indexOf('/api/admin/users') !== -1)
      return json({ users: [
        { user_id: 'u-anita', display_name: 'Anita R', role: 'user', worker_id: 'w-ccr-triage', enabled: true },
        { user_id: 'u-maaz', display_name: 'Maaz K', role: 'admin', worker_id: 'w-market-risk', enabled: true }
      ] });
    if (u.indexOf('/api/agent/threads') !== -1) return json({ threads: [] });
    if (u.indexOf('/api/tools') !== -1 || u.indexOf('tools/list') !== -1)
      return json({ tools: WORKERS[0].enabled_tools.concat(['yahoo_get_quote', 'edgar_search', 'read_pdf']).map(function (n) { return { name: n, description: 'SAJHA tool: ' + n }; }) });
    if (u.indexOf('/api/') !== -1) return json({});
    return realFetch(url, opts);
  };
})();
