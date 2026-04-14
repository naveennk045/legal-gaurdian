"""Optional Google Sheets and email integrations (env-driven)."""

import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any, Dict, List

from config.config import Config

logger = logging.getLogger(__name__)


def save_results_to_google_sheet(results: List[Dict[str, Any]]) -> None:
    """Append risk table to configured Google Sheet."""
    path = Config.GOGLE_SERVICE_ACCOUNT_FILE
    sheet_id = Config.GOOGLE_SPREADSHEET_ID
    if not path or not sheet_id:
        raise ValueError(
            "Set GOOGLE_SERVICE_ACCOUNT_FILE and GOOGLE_SPREADSHEET_ID in .env"
        )

    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build

    credentials = Credentials.from_service_account_file(
        path,
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    service = build("sheets", "v4", credentials=credentials)
    sheet = service.spreadsheets()

    values = [["Context", "Risks", "Recommendations"]]
    for row in results:
        values.append([row["context"], row["risks"], row["recommendations"]])

    sheet.values().update(
        spreadsheetId=sheet_id,
        range="Sheet1!A1",
        valueInputOption="RAW",
        body={"values": values},
    ).execute()


def send_email_report(
    recipient: str,
    sender: str,
    password: str,
    subject: str = "Legal risk analysis",
    body: str = "",
) -> None:
    """Send a simple SMTP email (e.g. Gmail app password)."""
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()
    server.login(sender, password)
    message = MIMEMultipart()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.attach(MIMEText(body, "plain"))
    server.send_message(message)
    server.quit()
