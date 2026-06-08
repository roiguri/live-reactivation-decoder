PRIMARY_BLUE       = "#0078D4"
PRIMARY_BLUE_HOVER = "#006CBE"
SUCCESS_GREEN      = "#228B22"
BG_LIGHT           = "#F3F3F3"
CARD_WHITE         = "#FFFFFF"
TEXT_PRIMARY       = "#1F2937"
TEXT_MUTED         = "#6B7280"
ALERT_RED          = "#C41E3A"
AMBER              = "#B45309"
BORDER_GRAY        = "#E5E7EB"

# AUC-chart line palette. Chosen to avoid the obvious red/green/blue/yellow
# tones so a decoder named ``red decoder`` doesn't get a red line and confuse
# the colour ↔ name mapping (the legend explicitly bridges the two). No
# blues — those are reserved for the suggested/selected vertical markers
# (PRIMARY_BLUE) so the operator's chosen timepoint stays unambiguous.
CHART_LINE_COLORS = [
    "#7C3AED",  # purple
    "#F97316",  # orange
    "#10B981",  # emerald
    "#DB2777",  # magenta
    "#A855F7",  # violet
    "#A16207",  # bronze
]


def chart_line_color(index: int) -> str:
    """Cycle the chart palette so >6 decoders still get distinct lines."""
    return CHART_LINE_COLORS[index % len(CHART_LINE_COLORS)]


def progress_bar_qss(object_name: str, *, radius: int = 2) -> str:
    """Single source of truth for the app's progress-bar look.

    Both loading idioms share it so they read as one design system: the
    transient ``LoadingOverlay`` (short/indeterminate waits) and the
    in-workspace progress pages (long ops with structured progress, e.g.
    evaluation's per-decoder ``CVProgressView``). Scoped by ``object_name``
    so a caller can style several distinct bars from one rule.
    """
    return (
        f"QProgressBar#{object_name} {{ background: #F3F4F6; "
        f"border: 1px solid {BORDER_GRAY}; border-radius: {radius}px; }}"
        f"QProgressBar#{object_name}::chunk {{ background: {PRIMARY_BLUE}; "
        f"border-radius: {radius}px; }}"
    )

def build_app_palette():
    from PyQt6.QtGui import QPalette, QColor
    p = QPalette()
    p.setColor(QPalette.ColorRole.Window,          QColor(BG_LIGHT))
    p.setColor(QPalette.ColorRole.WindowText,      QColor(TEXT_PRIMARY))
    p.setColor(QPalette.ColorRole.Base,            QColor(CARD_WHITE))
    p.setColor(QPalette.ColorRole.AlternateBase,   QColor(BG_LIGHT))
    p.setColor(QPalette.ColorRole.Text,            QColor(TEXT_PRIMARY))
    p.setColor(QPalette.ColorRole.ButtonText,      QColor(TEXT_PRIMARY))
    p.setColor(QPalette.ColorRole.Button,          QColor(CARD_WHITE))
    p.setColor(QPalette.ColorRole.PlaceholderText, QColor(TEXT_MUTED))
    return p


GLOBAL_QSS = f"""
QPushButton[class="primary"] {{
    background-color: {PRIMARY_BLUE};
    color: white;
    border: none;
    border-radius: 2px;
    padding: 6px 20px;
    font-size: 13px;
    font-weight: 600;
}}
QPushButton[class="primary"]:hover {{
    background-color: {PRIMARY_BLUE_HOVER};
}}
QPushButton[class="primary"]:disabled {{
    background-color: #D1D5DB;
    color: {TEXT_MUTED};
}}

QPushButton[class="secondary"] {{
    background-color: {CARD_WHITE};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER_GRAY};
    border-radius: 2px;
    padding: 6px 16px;
    font-size: 13px;
}}
QPushButton[class="secondary"]:hover {{
    background-color: #F3F4F6;
}}
QPushButton[class="secondary"]:disabled {{
    background-color: #D1D5DB;
    color: {TEXT_MUTED};
}}

/* Flat underline-style tabs (matches the React design demo). */
QTabWidget::pane {{
    border: none;
    background: transparent;
}}
QTabBar {{
    /* Removes the 1-px baseline Qt draws under the tab strip in
       document-mode — otherwise a faint line floats above the
       Summary/decoder tab content. */
    qproperty-drawBase: 0;
    background: transparent;
    border: none;
}}
QTabBar::tab {{
    background: transparent;
    border: none;
    border-bottom: 2px solid transparent;
    padding: 8px 16px;
    color: {TEXT_MUTED};
    font-size: 12px;
    font-weight: 600;
}}
QTabBar::tab:selected {{
    color: {PRIMARY_BLUE};
    border-bottom: 2px solid {PRIMARY_BLUE};
}}
QTabBar::tab:!selected:hover {{
    color: {TEXT_PRIMARY};
}}
/* Visually divide the Summary tab from the per-decoder tabs. The
   Summary tab is always first; uppercasing it + adding a right border
   makes it read as a section heading rather than a sibling decoder. */
QTabBar::tab:first {{
    border-right: 1px solid {BORDER_GRAY};
    letter-spacing: 0.6px;
}}
"""
