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
