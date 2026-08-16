# Compliance evidence report example

Generate the authoritative JSON first:

```bash
agent-strace compliance-report --framework owasp-agentic --since 30d \
  --output agent-evidence.json
```

To evaluate what the current local policy would have done, opt in explicitly:

```bash
agent-strace compliance-report --framework owasp-agentic --since 30d \
  --policy .agent-scope.json --output agent-evidence-with-policy.json
```

Policy outcomes are retrospective `would_allow`/`would_deny` context. They do
not establish that authorization existed when an operation occurred. Review
the report's coverage, each mapping's limitations, and `not_assessed` areas;
do not interpret the report as legal advice, a compliance determination, or a
control-effectiveness conclusion.

