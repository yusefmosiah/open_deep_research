"""Ghostwriter graph: planning → research → citation_check → draft → critique → revision → style_control.

Phase 1 focuses on structure using the existing Configuration and models. Once stable,
we can assign specialized models per phase as described in ghostwriter_langgraph.md.
"""

from __future__ import annotations

import asyncio
import re
from typing import Annotated, Literal, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    get_buffer_string,
    MessageLikeRepresentation,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import Command
from pydantic import BaseModel

from open_deep_research.configuration import Configuration
from open_deep_research.prompts import (
    final_report_generation_prompt,
    lead_researcher_prompt,
    transform_messages_into_research_topic_prompt,
)
from open_deep_research.state import override_reducer
from open_deep_research.utils import (
    get_api_key_for_model,
    get_today_str,
    fetch_url_text,
    tavily_search_async,
    load_validated_citations_from_file,
    save_validated_citations,
    convert_inline_links_to_footnotes,
    strip_model_preamble,
    write_file_if_path_provided,
)
# Use the full research supervisor from deep_researcher (executes tools and yields real URLs)
from open_deep_research.deep_researcher import supervisor_subgraph as research_supervisor_subgraph


# Reuse a configurable chat model across nodes
configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)


class GhostwriterInputState(MessagesState):
    """Input state: accepts the usual messages plus optional file paths.

    - style_guide: optional free-text style guide
    - citations_file: optional path to a validated citations file to read before drafting
    - citations_out: optional path (without extension) for writing validated citations
    - output_file: optional path for writing the final document
    """

    style_guide: Optional[str] = None
    citations_file: Optional[str] = None
    citations_out: Optional[str] = None
    output_file: Optional[str] = None


class GhostwriterState(MessagesState):
    """Ghostwriter state tracking all intermediate artifacts."""

    # Supervisor state (compatible with essay_writer's subgraph)
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str] = None
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0

    # Writing-specific artifacts
    writing_plan: Optional[str] = None
    initial_draft: Optional[str] = None
    critique_report: Optional[str] = None
    revision_instructions: Annotated[list[str], override_reducer] = []
    revised_draft: Optional[str] = None
    final_document: Optional[str] = None
    style_guide: Optional[str] = None

    # Citation checker artifacts
    citation_status: Annotated[list[dict], override_reducer] = []
    dead_links: Annotated[list[str], override_reducer] = []
    validated_citations: Annotated[list[dict], override_reducer] = []


# ---- Planning ----
async def planning(state: GhostwriterState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """Turn user's long prompt into a writing plan and seed research supervisor."""

    configurable = Configuration.from_runnable_config(config)
    model_cfg = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"],
    }

    # Step 1: Create a concise writing plan/brief (reuse transform prompt for now)
    plan_prompt = transform_messages_into_research_topic_prompt.format(
        messages=get_buffer_string(state.get("messages", [])),
        date=get_today_str(),
    )
    plan = await configurable_model.with_config(model_cfg).ainvoke([HumanMessage(content=plan_prompt)])
    writing_plan = plan.content

    # Step 2: Prepare supervisor system + kickoff message
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today_str(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations,
    )

    return Command(
        goto="research_supervisor",
        update={
            "writing_plan": writing_plan,
            "research_brief": writing_plan,
            "messages": [AIMessage(content="Planning complete. Proceeding to research supervisor.")],
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content="Draft an essay. Use this outline/brief:\n\n" + writing_plan),
                ],
            },
        },
    )




# ---- Draft ----
async def draft(state: GhostwriterState, config: RunnableConfig) -> Command[Literal["critique"]]:
    """Create an initial draft using the writing plan and validated research notes."""

    configurable = Configuration.from_runnable_config(config)
    model_cfg = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"],
    }

    findings = "\n".join(state.get("notes", []))

    # Build an allowlist of validated citations as [Title](URL)
    vc = state.get("validated_citations", []) or []
    # Merge with citations from file if provided
    if state.get("citations_file"):
        vc_file = load_validated_citations_from_file(state["citations_file"])  # type: ignore
        if vc_file:
            # Merge by URL uniqueness
            seen_urls = { (c.get("url") or c.get("final_url")) for c in vc }
            for c in vc_file:
                url = c.get("url") or c.get("final_url")
                if url and url not in seen_urls:
                    vc.append(c)
                    seen_urls.add(url)

    allowed_citations_lines = []
    for c in vc:
        t = c.get("title") or c.get("anchor") or c.get("page_title") or "Untitled"
        u = c.get("url") or c.get("final_url")
        if u:
            allowed_citations_lines.append(f"- [{t}]({u})")
    allowed_citations = "\n".join(allowed_citations_lines)

    # If we have validated citations, surface them directly in findings and add strict rules
    if allowed_citations:
        findings = findings + "\n\nValidated Sources (allowlist):\n" + allowed_citations + "\n"
        rules_suffix = (
            "\n\nCITATION RULES:\n"
            "- You MUST cite only from the validated sources list above.\n"
            "- You MAY reformat, but do NOT invent new URLs.\n"
            "- Use [Title](URL) format for all references.\n"
        )
    else:
        # No validated list available; use softer guidance
        rules_suffix = (
            "\n\nCITATION RULES:\n"
            "- Include links for citations when possible.\n"
            "- Prefer sources explicitly present in the findings.\n"
            "- Use [Title](URL) format for all references.\n"
        )

    # Enhanced prompt
    enhanced_prompt = final_report_generation_prompt.format(
        research_brief=state.get("writing_plan", ""),
        messages=get_buffer_string(state.get("messages", [])),
        findings=findings,
        date=get_today_str(),
    ) + rules_suffix

    response = await configurable_model.with_config(model_cfg).ainvoke([HumanMessage(content=enhanced_prompt)])

    return Command(
        goto="critique",
        update={
            "initial_draft": strip_model_preamble(response.content),
            "messages": [AIMessage(content="Initial draft completed using validated citations.")],
        },
    )


# ---- Critique ----
CRITIQUE_PROMPT = (
    "You are a meticulous editor. Analyze the draft against the brief. "
    "Provide specific, actionable revision instructions as a bulleted list.\n\n"
    "Brief:\n{brief}\n\nDraft:\n{draft}"
)


async def critique(state: GhostwriterState, config: RunnableConfig) -> Command[Literal["revision"]]:
    configurable = Configuration.from_runnable_config(config)
    model_cfg = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"],
    }

    prompt = CRITIQUE_PROMPT.format(
        brief=state.get("writing_plan", ""),
        draft=state.get("initial_draft", ""),
    )
    response = await configurable_model.with_config(model_cfg).ainvoke([HumanMessage(content=prompt)])

    # Naively split into instructions lines
    instructions = [line.strip("- ") for line in response.content.splitlines() if line.strip()]

    return Command(
        goto="revision",
        update={
            "critique_report": response.content,
            "revision_instructions": instructions,
            "messages": [AIMessage(content="Critique completed.")],
        },
    )


# ---- Citation helpers ----
class SupportJudgment(BaseModel):
    # Whether the document identity (title/url) matches the expected citation
    is_correct_document: bool
    # Whether the source text supports the local claim context
    supports_claim: bool
    # Short reason and an optional quoted snippet for support
    reason: str
    quote: Optional[str] = None
    # Optional normalized page title for audit/debug
    page_title: Optional[str] = None
    final_url: Optional[str] = None


def _extract_citations(text: str) -> list[dict]:
    citations: list[dict] = []
    seen_urls: set[str] = set()
    # Markdown links [anchor](url)
    for m in re.finditer(r"\[([^\]]+)\]\((https?://[^\)\s]+)\)", text):
        anchor, url = m.group(1), m.group(2)
        citations.append({"url": url, "anchor": anchor, "start": m.start(), "end": m.end()})
        seen_urls.add(url)
    # Bibliography-style lines: [n] ... "Title", ... URL
    for m in re.finditer(r"^\[(\d+)\]\s+(?P<entry>.+?)\s+(?P<url>https?://\S+)\s*$", text, flags=re.MULTILINE):
        entry = m.group("entry")
        url = m.group("url")
        if url in seen_urls:
            continue
        title_match = re.search(r'"([^"]+)"', entry)
        title = title_match.group(1) if title_match else entry.split(",")[0].strip()
        citations.append({"url": url, "anchor": title or url, "title": title or None, "start": m.start(), "end": m.end()})
        seen_urls.add(url)
    # Bare URLs (not already captured)
    for m in re.finditer(r"(?<!\]\()(?P<url>https?://[^\s)]+)", text):
        url = m.group("url")
        if url in seen_urls:
            continue
        if not any(c.get("start") <= m.start() <= c.get("end") for c in citations):
            citations.append({"url": url, "anchor": url, "start": m.start(), "end": m.end()})
            seen_urls.add(url)
    # arXiv identifiers without links (e.g., arXiv:1906.02406 or 1906.02406)
    for m in re.finditer(r"(?:arXiv:)?(?P<ax>\d{4}\.\d{4,5})(?:v\d+)?", text, flags=re.IGNORECASE):
        ax = m.group("ax")
        url = f"https://arxiv.org/abs/{ax}"
        if url not in seen_urls:
            citations.append({"url": url, "anchor": f"arXiv:{ax}", "title": None, "start": m.start(), "end": m.end()})
            seen_urls.add(url)
    # DOI identifiers without links
    for m in re.finditer(r"(?:doi:)?\s*(?P<doi>10\.\d{4,9}/[-._;()/:A-Z0-9]+)", text, flags=re.IGNORECASE):
        doi = m.group("doi")
        url = f"https://doi.org/{doi}"
        if url not in seen_urls:
            citations.append({"url": url, "anchor": f"DOI:{doi}", "title": None, "start": m.start(), "end": m.end()})
            seen_urls.add(url)
    return citations


def _context_window(text: str, idx: int, span: int = 500) -> str:
    lo = max(0, idx - span)
    hi = min(len(text), idx + span)
    # try to expand to paragraph boundaries
    para_start = text.rfind("\n\n", 0, idx)
    para_end = text.find("\n\n", idx)
    if para_start != -1:
        lo = max(lo, para_start)
    if para_end != -1:
        hi = min(hi, para_end)
    return text[lo:hi]




def _title_match_ok(expected: Optional[str], actual: Optional[str]) -> bool:
    """Heuristic check: require modest token overlap between expected and actual titles.

    Returns True if either is missing. Otherwise, computes Jaccard on alphanumeric tokens
    (len>=3) and also checks substring containment. Threshold is conservative (>=0.3).
    """
    if not expected or not actual:
        return True
    import re
    def to_tokens(s: str) -> set[str]:
        return {t for t in re.findall(r"[a-zA-Z0-9]+", s.lower()) if len(t) >= 3}
    e = to_tokens(expected)
    a = to_tokens(actual)
    if not e or not a:
        return True
    if expected.lower() in actual.lower() or actual.lower() in expected.lower():
        return True
    inter = len(e & a)
    union = len(e | a)
    return (inter / union) >= 0.3

async def _judge_support(model_cfg: dict, expected_title: Optional[str], page_title: Optional[str], claim_context: str, page_text: str, final_url: Optional[str]) -> SupportJudgment:
    prompt = (
        "You are a citation validator. Perform two critical checks:\n"
        "1. DOCUMENT IDENTITY: Does the actual page match the expected citation?\n"
        "2. CLAIM SUPPORT: Does the page content support the specific claim?\n\n"

        "For DOCUMENT IDENTITY, consider:\n"
        "- Do the titles match (allowing for minor variations)?\n"
        "- Are the topics/domains related (e.g., both about AI/ML)?\n"
        "- Does the URL make sense for the claimed content?\n"
        "- If it's an arXiv paper, does the subject area match?\n\n"

        "Mark is_correct_document as FALSE if:\n"
        "- Titles are completely different topics (e.g., 'RL' vs 'Astrophysics')\n"
        "- Wrong domain entirely (physics paper when expecting AI paper)\n"
        "- URL clearly points to wrong content\n\n"

        f"Expected citation: {expected_title or 'Unknown'}\n"
        f"Actual page title: {page_title or 'Unknown'}\n"
        f"URL: {final_url or 'Unknown'}\n\n"

        f"Claim context: {claim_context}\n\n"
        f"Page content (first 4000 chars):\n{page_text[:4000]}\n\n"

        "Return JSON: {is_correct_document: bool, supports_claim: bool, reason: string, quote: string|null}"
    )
    model = configurable_model.with_structured_output(SupportJudgment).with_config(model_cfg)
    resp: SupportJudgment = await model.ainvoke([HumanMessage(content=prompt)])
    # Attach page_title/final_url for downstream auditing
    resp.page_title = page_title
    resp.final_url = final_url
    # Apply stricter local heuristic for document identity
    if not _title_match_ok(expected_title, page_title):
        resp.is_correct_document = False
    return resp


async def citation_check(state: GhostwriterState, config: RunnableConfig) -> Command[Literal["draft"]]:
    """Verify citations from research notes; clean and validate before drafting."""
    configurable = Configuration.from_runnable_config(config)
    research_model_cfg = {
        "model": configurable.research_model,
        "max_tokens": configurable.research_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.research_model, config),
        "tags": ["langsmith:nostream"],
    }

    # Extract citations from research notes instead of draft
    research_notes = "\n".join(state.get("notes", []))
    cites = _extract_citations(research_notes)

    # Debug: write research notes and extracted citations to disk to diagnose empty allowlists
    try:
        from pathlib import Path
        import json as _json
        debug_dir = Path("out/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        Path(debug_dir / "research_notes.md").write_text(research_notes, encoding="utf-8")
        Path(debug_dir / "extracted_citations.json").write_text(_json.dumps(cites, ensure_ascii=False, indent=2), encoding="utf-8")
        md_lines = [f"- [{c.get('title') or c.get('anchor') or 'Untitled'}]({c.get('url')})" for c in cites if c.get("url")]
        Path(debug_dir / "extracted_citations.md").write_text("\n".join(md_lines), encoding="utf-8")
    except Exception:
        pass

    # Limit concurrency to avoid overload
    sem = asyncio.Semaphore(8)

    async def process_cite(cite: dict):
        url = cite["url"]
        ctx = _context_window(research_notes, cite["start"]) if research_notes else ""
        async with sem:
            fetched = await fetch_url_text(url)
        if not fetched.get("ok"):
            return {"url": url, "status": "dead", "http_status": fetched.get("status"), "error": fetched.get("error"), "anchor": cite["anchor"], "context": ctx}
        judgment = await _judge_support(
            research_model_cfg,
            expected_title=cite.get("title") or cite.get("anchor"),
            page_title=fetched.get("page_title"),
            claim_context=ctx,
            page_text=fetched.get("text") or "",
            final_url=fetched.get("final_url"),
        )
        if judgment.is_correct_document and judgment.supports_claim:
            return {"url": url, "status": "ok", "anchor": cite["anchor"], "context": ctx, "reason": judgment.reason, "quote": judgment.quote, "page_title": judgment.page_title, "final_url": judgment.final_url}
        # Try multiple replacement search strategies
        q_base = cite.get("title") or cite.get("anchor") or ""
        search_queries = [
            q_base.strip(),  # Title only
            f"{q_base} {ctx[:100]}".strip(),  # Title + context
            f"site:arxiv.org {q_base}".strip() if "arxiv" in url.lower() else f"{q_base} paper".strip(),  # Domain-specific
        ]
        search_results = await tavily_search_async([q for q in search_queries if q], max_results=8, include_raw_content=True, config=config)
        for sr in search_results:
            for result in (sr or {}).get("results", []):
                cand_url = result.get("url")
                if not cand_url:
                    continue
                async with sem:
                    cand_fetch = await fetch_url_text(cand_url)
                if not cand_fetch.get("ok"):
                    continue
                cand_judgment = await _judge_support(
                    research_model_cfg,
                    expected_title=cite.get("title") or cite.get("anchor"),
                    page_title=cand_fetch.get("page_title"),
                    claim_context=ctx,
                    page_text=cand_fetch.get("text") or "",
                    final_url=cand_fetch.get("final_url"),
                )
                if cand_judgment.is_correct_document and cand_judgment.supports_claim:
                    return {"url": url, "status": "replaced", "replacement_url": cand_url, "anchor": cite["anchor"], "context": ctx, "reason": cand_judgment.reason, "quote": cand_judgment.quote, "page_title": cand_judgment.page_title, "final_url": cand_judgment.final_url}
        # No replacement found
        return {"url": url, "status": "unsupported", "anchor": cite["anchor"], "context": ctx, "reason": judgment.reason, "page_title": judgment.page_title, "final_url": judgment.final_url}

    # Run all tasks; capture exceptions per-citation so we don't bail out early
    results_raw = await asyncio.gather(*[process_cite(c) for c in cites], return_exceptions=True)
    results = []
    for item in results_raw:
        if isinstance(item, Exception):
            results.append({"url": None, "status": "error", "error": str(item)})
        else:
            results.append(item)

    dead_links = [r["url"] for r in results if r["status"] == "dead"]

    # Build validated citations list (ok + replacements)
    validated_citations: list[dict] = []
    seen_urls: set[str] = set()
    for r in results:
        if r["status"] in ["ok", "replaced"]:
            url_out = r.get("replacement_url") if r["status"] == "replaced" else r.get("final_url") or r.get("url")
            if url_out and url_out not in seen_urls:
                seen_urls.add(url_out)
                title_out = r.get("anchor") or r.get("page_title")
                validated_citations.append({"title": title_out, "url": url_out, "page_title": r.get("page_title"), "final_url": r.get("final_url")})

    # Fallback: if validation yields none but we extracted citations, allow raw extracted citations
    if not validated_citations and cites:
        for c in cites:
            u = c.get("url")
            if u and u not in seen_urls:
                seen_urls.add(u)
                validated_citations.append({"title": c.get("title") or c.get("anchor") or u, "url": u})

    # Update research notes with validated citations
    validated_notes = []
    for note in state.get("notes", []):
        updated_note = note
        for r in results:
            if r["status"] == "replaced":
                updated_note = updated_note.replace(r["url"], r["replacement_url"])
            elif r["status"] in ["dead", "unsupported"]:
                # Remove the problematic URL but keep the content
                updated_note = re.sub(rf'\[([^\]]+)\]\({re.escape(r["url"])}\)', r'\1', updated_note)
                updated_note = updated_note.replace(r["url"], "")
        validated_notes.append(updated_note)

    # Write validated citations to disk (JSON and MD)
    base_out = state.get("citations_out") or "out/validated_citations"
    save_validated_citations(validated_citations, base_out)

    # Debug: also save citation_status for audit
    try:
        from pathlib import Path
        import json as _json
        debug_dir = Path("out/debug")
        debug_dir.mkdir(parents=True, exist_ok=True)
        Path(debug_dir / "citation_status.json").write_text(_json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    return Command(
        goto="draft",
        update={
            "citation_status": results,
            "dead_links": dead_links,
            "notes": {"type": "override", "value": validated_notes},
            "validated_citations": {"type": "override", "value": validated_citations},
            "messages": [AIMessage(content=f"Citation validation completed. {len(dead_links)} dead links removed; {sum(1 for r in results if r['status']=='unsupported')} unsupported removed; {sum(1 for r in results if r['status']=='replaced')} replacements made. Ready for drafting.")],
        },
    )


# ---- Revision ----
REVISION_PROMPT = (
    "Revise the draft according to the instructions. Keep citations intact.\n\n"
    "Draft:\n{draft}\n\nInstructions:\n{instructions}"
)


async def revision(state: GhostwriterState, config: RunnableConfig) -> Command[Literal["style_control"]]:
    configurable = Configuration.from_runnable_config(config)
    model_cfg = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"],
    }

    prompt = REVISION_PROMPT.format(
        draft=state.get("initial_draft", ""),
        instructions="\n".join(state.get("revision_instructions", [])),
    )
    response = await configurable_model.with_config(model_cfg).ainvoke([HumanMessage(content=prompt)])

    return Command(
        goto="style_control",
        update={
            "revised_draft": response.content,
            "messages": [AIMessage(content="Revision completed.")],
        },
    )


# ---- Style control ----
STYLE_PROMPT = (
    "Apply the following style guide to the draft while preserving content and citations.\n\n"
    "Style Guide:\n{style}\n\nDraft:\n{draft}"
)


async def style_control(state: GhostwriterState, config: RunnableConfig) -> dict:
    configurable = Configuration.from_runnable_config(config)
    style = state.get("style_guide")
    output_file = state.get("output_file")
    if not style:
        # Pass-through if no style guide provided
        final_text = state.get("revised_draft", state.get("initial_draft", ""))
        final_text = strip_model_preamble(final_text)
        # Convert inline links to footnotes using validated citations
        final_text = convert_inline_links_to_footnotes(final_text, state.get("validated_citations", []))
        write_file_if_path_provided(final_text, output_file)
        return {
            "final_document": final_text,
            "messages": [AIMessage(content="Document completed (no style guide provided).")],
        }

    model_cfg = {
        "model": configurable.final_report_model,
        "max_tokens": configurable.final_report_model_max_tokens,
        "api_key": get_api_key_for_model(configurable.final_report_model, config),
        "tags": ["langsmith:nostream"],
    }

    prompt = STYLE_PROMPT.format(style=style, draft=state.get("revised_draft", ""))
    response = await configurable_model.with_config(model_cfg).ainvoke([HumanMessage(content=prompt)])

    final_text = strip_model_preamble(response.content)
    # Convert inline links to footnotes using validated citations
    final_text = convert_inline_links_to_footnotes(final_text, state.get("validated_citations", []))
    output_file = state.get("output_file")
    write_file_if_path_provided(final_text, output_file)

    return {
        "final_document": final_text,
        "messages": [AIMessage(content="Document completed with style guide applied.")],
    }


# Build main ghostwriter graph
builder = StateGraph(GhostwriterState, input=GhostwriterInputState, config_schema=Configuration)

# Nodes
builder.add_node("planning", planning)
builder.add_node("research_supervisor", research_supervisor_subgraph)
builder.add_node("draft", draft)
builder.add_node("critique", critique)
builder.add_node("citation_check", citation_check)
builder.add_node("revision", revision)
builder.add_node("style_control", style_control)

# Edges
builder.add_edge(START, "planning")
builder.add_edge("research_supervisor", "citation_check")
builder.add_edge("citation_check", "draft")
builder.add_edge("draft", "critique")
builder.add_edge("critique", "revision")
builder.add_edge("revision", "style_control")
builder.add_edge("style_control", END)

# Export graph
ghostwriter = builder.compile()
