# Ghostwriter LangGraph Implementation

## Overview

This document outlines the implementation of a sophisticated nonfiction writing system as a LangGraph workflow within the existing open_deep_research project. The system will leverage LangGraph's state management, LangSmith's observability, and support for long-context prompts (up to 100k tokens).

## Architecture

### Graph Structure
```
start → planning → research_supervisor → draft → critique → citation_check → revision → style_control → end
```

### Key Design Principles
- **Long Context Support**: Handle 100k+ token initial prompts
- **Context Management**: Drop initial prompt after planning to manage token limits
- **Model Specialization**: Different models for each phase via configuration
- **State Persistence**: Full audit trail via LangGraph state
- **Observability**: Rich tracing through LangSmith
- **Style Guide Integration**: User-editable style prompts as additional input

## State Schema

```python
class GhostwriterInputState(MessagesState):
    """Input state with messages and style guide."""
    style_guide: Optional[str] = None

class GhostwriterState(MessagesState):
    """Main ghostwriter state with all intermediate artifacts."""
    
    # Planning phase outputs
    writing_plan: Optional[str] = None
    sections: Annotated[list[dict], operator.add] = []
    
    # Research phase outputs  
    research_findings: Annotated[list[dict], operator.add] = []
    sources: Annotated[list[dict], operator.add] = []
    
    # Draft phase outputs
    initial_draft: Optional[str] = None
    
    # Critique phase outputs
    critique_report: Optional[str] = None
    revision_instructions: Annotated[list[str], operator.add] = []
    
    # Citation verification outputs
    citation_status: Annotated[list[dict], operator.add] = []
    dead_links: Annotated[list[str], operator.add] = []
    
    # Revision phase outputs
    revised_draft: Optional[str] = None
    
    # Final outputs
    final_document: Optional[str] = None
    style_guide: Optional[str] = None
    
    # Supervisor state for research coordination
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer] = []
    research_iterations: int = 0
```

## Configuration Schema

```python
class GhostwriterConfiguration(Configuration):
    """Extended configuration for model specialization."""
    
    # Model assignments for each phase
    planning_model: str = "anthropic:claude-3-5-sonnet-20241022"
    research_model: str = "openai:gpt-4o-2024-08-06"  
    drafting_model: str = "anthropic:claude-3-5-sonnet-20241022"
    critique_model: str = "openai:gpt-4o-2024-08-06"
    citation_model: str = "openai:gpt-4o-mini-2024-07-18"
    revision_model: str = "anthropic:claude-3-5-sonnet-20241022"
    style_model: str = "anthropic:claude-3-5-sonnet-20241022"
    
    # Phase-specific parameters
    max_sections: int = 8
    target_word_count: int = 2000
    research_depth: int = 3  # sources per section
    citation_style: str = "APA"
    
    # Context management
    drop_initial_prompt_after_planning: bool = True
    max_context_tokens: int = 100000
```

## Node Implementations

### 1. Planning Node
```python
async def planning(state: GhostwriterState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """Transform long user prompt into structured writing plan."""
    
    configurable = GhostwriterConfiguration.from_runnable_config(config)
    
    # Use planning-specific model
    planning_model = get_model_for_phase("planning", configurable)
    
    # Extract initial prompt (can be very long)
    initial_prompt = get_buffer_string(state["messages"])
    
    # Generate structured plan
    plan_prompt = planning_phase_prompt.format(
        user_request=initial_prompt,
        target_length=configurable.target_word_count,
        max_sections=configurable.max_sections
    )
    
    plan_response = await planning_model.ainvoke([HumanMessage(content=plan_prompt)])
    
    # Parse plan into structured sections
    sections = parse_writing_plan(plan_response.content)
    
    # Clear initial prompt from context if configured
    updated_messages = []
    if configurable.drop_initial_prompt_after_planning:
        updated_messages = [AIMessage(content=f"Planning complete. Generated {len(sections)} sections.")]
    else:
        updated_messages = [plan_response]
    
    return Command(
        goto="research_supervisor",
        update={
            "writing_plan": plan_response.content,
            "sections": sections,
            "messages": {"type": "override", "value": updated_messages},
            "supervisor_messages": {
                "type": "override", 
                "value": [SystemMessage(content=research_supervisor_prompt)]
            }
        }
    )
```

### 2. Research Supervisor (Subgraph)
```python
# Reuse existing supervisor pattern but with ghostwriter-specific tools
class ResearchSection(BaseModel):
    """Tool for researching a specific section."""
    section_title: str
    research_questions: list[str]
    target_sources: int = 3

async def research_supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """Coordinate parallel research for all sections."""
    
    configurable = GhostwriterConfiguration.from_runnable_config(config)
    research_model = get_model_for_phase("research", configurable)
    
    # Use research-specific tools
    research_tools = [ResearchSection, ResearchComplete, think_tool]
    
    # Delegate research for each section
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.bind_tools(research_tools).ainvoke(supervisor_messages)
    
    return Command(
        goto="supervisor_tools",
        update={
            "supervisor_messages": [response],
            "research_iterations": state.get("research_iterations", 0) + 1
        }
    )
```

### 3. Draft Node
```python
async def draft(state: GhostwriterState, config: RunnableConfig) -> Command[Literal["critique"]]:
    """Generate initial draft from plan and research."""
    
    configurable = GhostwriterConfiguration.from_runnable_config(config)
    drafting_model = get_model_for_phase("drafting", configurable)
    
    # Compile research findings
    research_summary = compile_research_findings(state.get("research_findings", []))
    
    draft_prompt = drafting_phase_prompt.format(
        writing_plan=state.get("writing_plan", ""),
        research_findings=research_summary,
        target_length=configurable.target_word_count,
        citation_style=configurable.citation_style
    )
    
    draft_response = await drafting_model.ainvoke([HumanMessage(content=draft_prompt)])
    
    return Command(
        goto="critique",
        update={
            "initial_draft": draft_response.content,
            "messages": [AIMessage(content="Initial draft completed.")]
        }
    )
```

### 4. Critique Node
```python
async def critique(state: GhostwriterState, config: RunnableConfig) -> Command[Literal["citation_check"]]:
    """Analyze draft and provide structured feedback."""
    
    configurable = GhostwriterConfiguration.from_runnable_config(config)
    critique_model = get_model_for_phase("critique", configurable)
    
    critique_prompt = critique_phase_prompt.format(
        original_plan=state.get("writing_plan", ""),
        draft=state.get("initial_draft", ""),
        target_length=configurable.target_word_count
    )
    
    critique_response = await critique_model.ainvoke([HumanMessage(content=critique_prompt)])
    
    # Parse critique into actionable instructions
    revision_instructions = parse_critique_response(critique_response.content)
    
    return Command(
        goto="citation_check",
        update={
            "critique_report": critique_response.content,
            "revision_instructions": revision_instructions,
            "messages": [AIMessage(content="Critique completed.")]
        }
    )
```

### 5. Citation Check Node (Parallel)
```python
async def citation_check(state: GhostwriterState, config: RunnableConfig) -> Command[Literal["revision"]]:
    """Verify all citations and find replacements for dead links."""
    
    configurable = GhostwriterConfiguration.from_runnable_config(config)
    citation_model = get_model_for_phase("citation", configurable)
    
    draft = state.get("initial_draft", "")
    citations = extract_citations(draft)
    
    # Parallel citation verification
    verification_tasks = [
        verify_citation(citation) for citation in citations
    ]
    
    verification_results = await asyncio.gather(*verification_tasks, return_exceptions=True)
    
    # Find replacements for dead links
    dead_links = [r for r in verification_results if r.get("status") == "dead"]
    replacement_tasks = [
        find_replacement_source(link, citation_model) for link in dead_links
    ]
    
    replacements = await asyncio.gather(*replacement_tasks, return_exceptions=True)
    
    return Command(
        goto="revision",
        update={
            "citation_status": verification_results,
            "dead_links": [r["url"] for r in dead_links],
            "messages": [AIMessage(content=f"Citation check completed. {len(dead_links)} dead links found.")]
        }
    )
```

### 6. Revision Node
```python
async def revision(state: GhostwriterState, config: RunnableConfig) -> Command[Literal["style_control"]]:
    """Revise draft based on critique and citation fixes."""
    
    configurable = GhostwriterConfiguration.from_runnable_config(config)
    revision_model = get_model_for_phase("revision", configurable)
    
    revision_prompt = revision_phase_prompt.format(
        original_draft=state.get("initial_draft", ""),
        critique_report=state.get("critique_report", ""),
        citation_issues=format_citation_issues(state.get("citation_status", [])),
        revision_instructions="\n".join(state.get("revision_instructions", []))
    )
    
    revised_response = await revision_model.ainvoke([HumanMessage(content=revision_prompt)])
    
    return Command(
        goto="style_control",
        update={
            "revised_draft": revised_response.content,
            "messages": [AIMessage(content="Revision completed.")]
        }
    )
```

### 7. Style Control Node
```python
async def style_control(state: GhostwriterState, config: RunnableConfig) -> dict:
    """Apply user-defined style guide as final step."""
    
    configurable = GhostwriterConfiguration.from_runnable_config(config)
    style_model = get_model_for_phase("style", configurable)
    
    # Get user's style guide (from input or default)
    style_guide = state.get("style_guide") or default_style_guide
    
    style_prompt = style_control_prompt.format(
        draft=state.get("revised_draft", ""),
        style_guide=style_guide
    )
    
    final_response = await style_model.ainvoke([HumanMessage(content=style_prompt)])
    
    return {
        "final_document": final_response.content,
        "messages": [AIMessage(content="Document completed with style guide applied.")]
    }
```

## Prompt Templates

### Planning Phase
```python
planning_phase_prompt = """You are an expert writing planner. Transform this user request into a detailed writing plan.

User Request:
{user_request}

Create a structured plan with:
1. Clear title and thesis
2. {max_sections} main sections with key points
3. Research questions for each section  
4. Target length: {target_length} words

Output as structured markdown with clear sections."""
```

### Style Control
```python
style_control_prompt = """Apply this style guide to the draft while preserving all content and citations.

Style Guide:
{style_guide}

Draft to Style:
{draft}

Rewrite the entire document to match the style guide exactly. Preserve all facts, citations, and structure."""
```

## Integration with Existing System

### File Structure
```
src/open_deep_research/
├── ghostwriter.py          # Main graph implementation
├── ghostwriter_prompts.py  # All prompt templates
├── ghostwriter_state.py    # State schemas
├── ghostwriter_config.py   # Configuration extensions
└── ghostwriter_tools.py    # Specialized tools
```

### LangGraph JSON Registration
```json
{
  "graphs": {
    "Deep Researcher": "./src/open_deep_research/deep_researcher.py:deep_researcher",
    "Essay Writer": "./src/open_deep_research/essay_writer.py:essay_writer", 
    "Ghostwriter": "./src/open_deep_research/ghostwriter.py:ghostwriter"
  }
}
```

### Usage Examples
```python
# Via API with style guide
{
  "input": {
    "messages": [{"role": "human", "content": "100k token prompt here..."}],
    "style_guide": "Academic tone, third person, APA citations..."
  },
  "config": {
    "configurable": {
      "planning_model": "anthropic:claude-3-5-sonnet-20241022",
      "drafting_model": "openai:gpt-4o-2024-08-06",
      "target_word_count": 3000
    }
  }
}
```

## Benefits of LangGraph Approach

1. **LangSmith Integration**: Full observability of each phase
2. **State Persistence**: Complete audit trail of all decisions
3. **Parallel Execution**: Research and citation checking
4. **Model Flexibility**: Easy A/B testing of different models per phase
5. **Context Management**: Automatic handling of long prompts
6. **Error Recovery**: Built-in retry and fallback mechanisms
7. **Studio Visualization**: Graph flow debugging and monitoring

## Implementation Plan

1. **Phase 1**: Basic graph structure with existing models
2. **Phase 2**: Add specialized prompts and model configuration
3. **Phase 3**: Implement citation verification and style control
4. **Phase 4**: Optimize for long context and performance
5. **Phase 5**: Add advanced features (style guide editor, etc.)

This approach leverages LangGraph's strengths while building a sophisticated writing system that can handle complex, long-form content generation with full traceability and model specialization.
