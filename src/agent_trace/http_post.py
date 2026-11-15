"""Helpers for POSTing to a URL the user gave, without showing its secrets.

watch (--stream-to, --webhook) and notify (Slack, Teams, generic webhook)
share these. User info, path, query and fragment can carry tokens, so
messages name only the URL's scheme://host[:port].
"""

from __future__ import annotations

import unicodedata
import urllib.error
import urllib.parse
import urllib.request


def url_endpoint(url: str) -> str:
    """Return the scheme://host[:port] that messages may name.

    User info, path, query and fragment can carry tokens, so messages never
    show them. Returns "" when no host can be read; the message then names
    no endpoint.
    """
    if not isinstance(url, str):
        return ""
    try:
        parts = urllib.parse.urlsplit(url)
        host = parts.hostname
        port = parts.port
    except ValueError:
        return ""
    if not parts.scheme or not host or not host.isprintable() or " " in host:
        return ""
    if ":" in host:
        host = f"[{host}]"
    return f"{parts.scheme}://{host}" + ("" if port is None else f":{port}")


def url_problem(url: str) -> str | None:
    """Return why *url* cannot be POSTed to, or None if it can.

    The reason never repeats any part of the URL.
    """
    # Category Cc is C0, DEL and C1 (U+0080 to U+009F). urllib cannot send
    # any of them, so each POST would fail.
    if any(ch == " " or unicodedata.category(ch) == "Cc" for ch in url):
        return "it must not contain spaces or control characters"
    try:
        parts = urllib.parse.urlsplit(url)
        host = parts.hostname
    except ValueError:
        return "it is not a valid URL"
    if parts.scheme not in ("http", "https"):
        return "it must start with http:// or https://"
    if not host:
        return "it has no host"
    if "@" in parts.netloc:
        return "it must not contain a user name or password"
    try:
        parts.port
    except ValueError:
        return "its port must be a number from 0 to 65535"
    return None


def no_redirect_opener() -> urllib.request.OpenerDirector:
    """Return an opener for POSTs that never follows a redirect.

    urlopen re-sends a POST answered with 301, 302 or 303 as a GET with no
    body, so the data is lost. With no redirect handler every 3xx raises
    HTTPError instead. Only http and https URLs can be opened.
    """
    opener = urllib.request.OpenerDirector()
    handlers: list[urllib.request.BaseHandler] = [
        urllib.request.ProxyHandler(),
        urllib.request.UnknownHandler(),
        urllib.request.HTTPHandler(),
        urllib.request.HTTPDefaultErrorHandler(),
        urllib.request.HTTPErrorProcessor(),
    ]
    if hasattr(urllib.request, "HTTPSHandler"):
        handlers.append(urllib.request.HTTPSHandler())
    for handler in handlers:
        opener.add_handler(handler)
    return opener


def post_error_detail(exc: BaseException) -> str:
    """Describe why a POST failed without quoting its URL.

    Keeps the status and reason of an HTTP error and the reason of a network
    error. Other exceptions (InvalidURL, ValueError, UnicodeEncodeError and
    more) can repeat the URL, so only their class name is given.
    """
    if isinstance(exc, urllib.error.HTTPError):
        detail = f"HTTP Error {exc.code}: {exc.reason}"
        if 300 <= exc.code < 400:
            detail += " (redirects are not followed; give the final URL)"
        return detail
    if isinstance(exc, urllib.error.URLError):
        return str(exc.reason)
    return type(exc).__name__
