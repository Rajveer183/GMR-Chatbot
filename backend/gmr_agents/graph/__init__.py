"""LangGraph orchestration of the NL-to-SQL agent.

    LLM        = controller / planner  (understand_node, controller_node, answer_node)
    code/tools = executor              (validate_node, execute_node, guards/, tools/)
    LangGraph  = orchestrator          (state, routing, loops, retries, checkpoints)
"""
