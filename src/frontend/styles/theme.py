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
"""
