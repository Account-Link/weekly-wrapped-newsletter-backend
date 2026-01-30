import os
import re
import time
from html import escape
from pathlib import Path
from typing import Optional
from urllib.parse import quote

import boto3
from botocore.exceptions import BotoCoreError, ClientError

_WRAPPED_EMAIL_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "wrapped-email-html" / "index.html"
_REAUTH_EMAIL_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "wrapped-email-html" / "reauth.html"
_USERNAME_TOKENS = ("{@username}", "{@uername}", "{username}", "{{username}}")
_WRAPPED_LINK_TOKENS = ("{wrapped_link}", "{wrapped_url}", "{{wrapped_link}}", "{{wrapped_url}}")

_OPEN_WRAPPED_LINK_RE = re.compile(r'(<a\b[^>]*\bhref=["\'])#(["\'])', flags=re.IGNORECASE)
_OPEN_WRAPPED_LINK_BY_CLASS_RE = re.compile(
    r'(<a\b[^>]*\bclass=["\'][^"\']*\bopen-buttton\b[^"\']*["\'][^>]*\bhref=["\'])#(["\'])',
    flags=re.IGNORECASE,
)


class Emailer:
    def __init__(self) -> None:
        self.sender = os.getenv("AWS_EMAIL")
        self.reply_to = (os.getenv("AWS_REPLY_TO") or "").strip() or None
        region = os.getenv("AWS_REGION")
        if not self.sender or not region:
            raise RuntimeError("AWS_EMAIL and AWS_REGION are required for email sending")
        self.client = boto3.client(
            "ses",
            region_name=region,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY") or os.getenv("AWS_ACCESS_KEY_SECRET"),
        )
        self._wrapped_template_html = self._load_wrapped_template()
        self._reauth_template_html = self._load_reauth_template()

    def _load_wrapped_template(self) -> Optional[str]:
        try:
            return _WRAPPED_EMAIL_TEMPLATE_PATH.read_text(encoding="utf-8")
        except OSError:
            return None

    def _load_reauth_template(self) -> Optional[str]:
        try:
            return _REAUTH_EMAIL_TEMPLATE_PATH.read_text(encoding="utf-8")
        except OSError:
            return None

    @staticmethod
    def _format_username(platform_username: Optional[str]) -> str:
        value = (platform_username or "").strip()
        if not value:
            return "there"
        return f"@{value.lstrip('@')}"

    def format_wrapped_email(
        self,
        app_user_id: str,
        platform_username: Optional[str],
        frontend_url: Optional[str],
    ) -> tuple[str, str, str]:
        encoded_id = quote(app_user_id, safe="")
        link = (
            f"{frontend_url.rstrip('/')}/wrapped?app_user_id={encoded_id}"
            if frontend_url
            else f"/wrapped?app_user_id={encoded_id}"
        )
        username = self._format_username(platform_username)
        subject = "Your 2025 TikTok Wrapped is ready"
        text_body = (
            f"Hey {username},\n\n"
            f"Your TikTok Wrapped is ready.\n\n"
            f"Open it here: {link}\n\n"
            "Thanks for trying TikTok Wrapped!"
        )

        html_body = self._wrapped_template_html
        if html_body:
            escaped_username = escape(username)
            for token in _USERNAME_TOKENS:
                html_body = html_body.replace(token, escaped_username)
            escaped_link = escape(link, quote=True)
            for token in _WRAPPED_LINK_TOKENS:
                html_body = html_body.replace(token, escaped_link)
            html_body, replacements = _OPEN_WRAPPED_LINK_BY_CLASS_RE.subn(
                lambda match: f"{match.group(1)}{escaped_link}{match.group(2)}",
                html_body,
                count=1,
            )
            if replacements == 0:
                html_body, replacements = _OPEN_WRAPPED_LINK_RE.subn(
                    lambda match: f"{match.group(1)}{escaped_link}{match.group(2)}",
                    html_body,
                    count=1,
                )
        else:
            html_body = f"""
            <html>
            <body>
                <p>Hey {escape(username)},</p>
                <p>Your TikTok Wrapped is ready.</p>
                <p><a href="{escape(link, quote=True)}">Open it here</a></p>
                <p>Thanks for trying TikTok Wrapped!</p>
            </body>
            </html>
            """
        return subject, text_body.strip(), html_body.strip()

    def format_reauth_email(
        self,
        platform_username: Optional[str],
        frontend_url: Optional[str],
    ) -> tuple[str, str, str]:
        link = (frontend_url or "").rstrip("/") or "/"
        username = self._format_username(platform_username)
        subject = "Finish connecting TikTok Wrapped"
        text_body = (
            f"Hey {username},\n\n"
            "We need you to reconnect TikTok to generate your Wrapped.\n\n"
            f"Continue here: {link}\n\n"
            "Thanks for trying TikTok Wrapped!"
        )

        html_body = self._reauth_template_html
        if html_body:
            escaped_username = escape(username)
            for token in _USERNAME_TOKENS:
                html_body = html_body.replace(token, escaped_username)
            escaped_link = escape(link, quote=True)
            html_body, replacements = _OPEN_WRAPPED_LINK_BY_CLASS_RE.subn(
                lambda match: f"{match.group(1)}{escaped_link}{match.group(2)}",
                html_body,
                count=1,
            )
            if replacements == 0:
                html_body, _ = _OPEN_WRAPPED_LINK_RE.subn(
                    lambda match: f"{match.group(1)}{escaped_link}{match.group(2)}",
                    html_body,
                    count=1,
                )
        else:
            html_body = f"""
            <html>
            <body>
                <p>Hey {escape(username)},</p>
                <p>We need you to reconnect TikTok to generate your Wrapped.</p>
                <p><a href="{escape(link, quote=True)}">Continue</a></p>
            </body>
            </html>
            """
        return subject, text_body.strip(), html_body.strip()

    def send_email(self, to_address: str, subject: str, text_body: str, html_body: str) -> Optional[dict]:
        if not to_address:
            return None
        attempts = 0
        backoff = 1.0
        while attempts < 3:
            attempts += 1
            try:
                kwargs = {}
                if self.reply_to:
                    kwargs["ReplyToAddresses"] = [self.reply_to]

                resp = self.client.send_email(
                    Source=self.sender,
                    Destination={"ToAddresses": [to_address]},
                    Message={
                        "Subject": {"Data": subject},
                        "Body": {
                            "Text": {"Data": text_body},
                            "Html": {"Data": html_body},
                        },
                    },
                    **kwargs,
                )
                return resp
            except (BotoCoreError, ClientError):
                if attempts >= 3:
                    return None
                time.sleep(backoff)
                backoff = min(backoff * 2, 4.0)
