"""
terminal_ui.py — Rich terminal interface for ScamShield Live.
Shows live transcripts, risk meter, evidence timeline, and alert box.
Requires: pip install rich
"""

from __future__ import annotations
import time
from typing import Optional
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.columns import Columns
from rich.rule import Rule
from rich.live import Live
from rich.layout import Layout
from rich import box

from ..core.scam_detector import Alert, DetectionResult, RiskLevel, EvidenceEvent
from ..core.asr_engine import Transcript

console = Console()

RISK_STYLES = {
    RiskLevel.SAFE:     ("green",    "✅ SAFE"),
    RiskLevel.LOW:      ("yellow",   "🟡 LOW RISK"),
    RiskLevel.MEDIUM:   ("orange3",  "🟠 MEDIUM RISK"),
    RiskLevel.HIGH:     ("red",      "⚠️  HIGH RISK"),
    RiskLevel.CRITICAL: ("bold red", "🚨 CRITICAL"),
}


class TerminalUI:
    """
    Live-updating terminal display.
    Call update_transcript(), update_detection(), fire_alert() as callbacks.
    """

    def __init__(self):
        self._console = Console()
        self._transcripts: list[str] = []
        self._evidence: list[EvidenceEvent] = []
        self._current_risk = RiskLevel.SAFE
        self._score = 0.0
        self._alert_active = False
        self._last_alert: Optional[Alert] = None
        self._session_start = time.monotonic()

    def print_header(self):
        self._console.print()
        self._console.print(Panel(
            Text("⚡ ScamShield Live", style="bold cyan", justify="center"),
            subtitle="[dim]On-Device Scam Detection · Snapdragon NPU[/dim]",
            border_style="cyan",
            padding=(0, 4),
        ))
        self._console.print()

    def update_transcript(self, transcript: Transcript):
        elapsed = time.monotonic() - self._session_start
        line = f"[dim]{elapsed:6.1f}s[/dim] [white]{transcript.text}[/white]"
        self._transcripts.append(line)
        self._console.print(f"  🎙  {elapsed:5.1f}s │ {transcript.text}")

    def update_detection(self, result: DetectionResult):
        self._current_risk = result.risk_level
        self._score = result.risk_score
        self._evidence.extend([])

        if result.matches:
            style, label = RISK_STYLES[result.risk_level]
            self._console.print(
                f"  🧠  [{style}]{label}[/{style}] "
                f"score=[bold]{result.risk_score:.1f}[/bold] "
                f"→ {result.matches[0]['pattern_name']}"
            )

    def fire_alert(self, alert: Alert):
        self._last_alert = alert
        style, label = RISK_STYLES[alert.risk_level]

        lines = []
        lines.append(Text(f"  {label}   Score: {alert.risk_score:.1f}/10", style=f"bold {style}"))
        lines.append(Text(""))

        lines.append(Text("  Active Scam Stages:", style="bold white"))
        for stage in alert.active_stages:
            lines.append(Text(f"    ✗  {stage.name.replace('_', ' ').title()}", style="red"))

        lines.append(Text(""))
        lines.append(Text("  Evidence:", style="bold white"))
        for ev in alert.evidence_timeline[-4:]:
            short = ev.evidence_text[:70]
            lines.append(Text(f'    [{ev.stage.name}]  "{short}"', style="dim white"))

        lines.append(Text(""))
        lines.append(Text(f"  ⚡ ACTION: {alert.mitigation}", style="bold yellow"))

        combined = Text("\n").join(lines)
        panel = Panel(
            combined,
            title=f"[{style}]🚨 SCAM ALERT[/{style}]",
            border_style=style,
            padding=(1, 2),
        )
        self._console.print()
        self._console.print(panel)
        self._console.print()

    def print_stats(self, stats_report: str):
        self._console.print(Rule(style="dim"))
        self._console.print(f"  [dim]{stats_report}[/dim]")
        self._console.print()


def print_banner():
    """Print startup banner."""
    console.print()
    console.print(Panel(
        "[bold cyan]⚡ ScamShield Live[/bold cyan]\n"
        "[dim]On-Device Real-Time Scam Detection[/dim]\n"
        "[dim]Snapdragon NPU · Qualcomm AI Hub · Privacy-First[/dim]",
        border_style="cyan",
        expand=False,
    ))
    console.print()


def print_alert(alert: Alert):
    """Standalone alert print — use when not running full TerminalUI."""
    style, label = RISK_STYLES[alert.risk_level]
    rows = []
    for ev in alert.evidence_timeline[-5:]:
        rows.append(f'[{ev.stage.name}]  "{ev.evidence_text[:65]}"')
    body = "\n".join(rows) if rows else "No evidence recorded."
    console.print(Panel(
        f"[bold {style}]{label}   Score: {alert.risk_score:.1f}/10[/bold {style}]\n\n"
        f"[white]{body}[/white]\n\n"
        f"[bold yellow]⚡ {alert.mitigation}[/bold yellow]",
        title="🚨 SCAM ALERT",
        border_style=style,
        padding=(1, 2),
    ))
