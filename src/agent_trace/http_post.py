"""Helpers for POSTing to a URL the user gave, without showing its secrets.

watch (--stream-to, --webhook), notify (Slack, Teams, generic webhook) and
export (--endpoint, --header) share these. User info, path, query and
fragment can carry tokens, so messages name only the URL's
scheme://host[:port]. The remote collector senders, Langfuse and OTLP
metrics export, apply --server, optimize, the llm_judge scorer and
pr-comment's writes also open their POSTs with no_redirect_opener.
apply --server and pr-comment open their GETs with same_origin_opener.
"""

from __future__ import annotations

import string
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

from .inert import inert

# RFC 9110 token characters, the only ones an HTTP header name may contain.
_TOKEN_CHARS = frozenset(string.ascii_letters + string.digits + "!#$%&'*+-.^_`|~")
_DEFAULT_PORTS = {"http": 80, "https": 443}
_ORIGIN_NOTE = (
    " (redirects to another scheme, host or port are not followed; give the final URL)"
)


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


def otlp_traces_url(url: str) -> str:
    """Return the OTLP/HTTP traces URL for a collector base URL.

    Trailing slashes are dropped from the path, then ``/v1/traces`` is
    appended unless the path already ends with it. The query and the rest
    of the URL are kept as given. The URL is split by hand because urlsplit
    drops tab, CR and LF, which would send to a URL the user did not give.
    """
    rest, hash_mark, fragment = url.partition("#")
    rest, question_mark, query = rest.partition("?")
    after_scheme = rest.find(":") + 1
    if rest.startswith("//", after_scheme):
        slash = rest.find("/", after_scheme + 2)
        path_start = len(rest) if slash < 0 else slash
    else:
        path_start = after_scheme
    path = rest[path_start:].rstrip("/")
    if not path.endswith("/v1/traces"):
        path += "/v1/traces"
    return f"{rest[:path_start]}{path}{question_mark}{query}{hash_mark}{fragment}"


def parse_header(header: str) -> tuple[str, str]:
    """Return the name and value of a ``NAME: VALUE`` header.

    The header is split at its first colon and spaces around the name and
    the value are dropped. The name must be an HTTP token and the value
    non-empty printable ASCII. Otherwise raises ValueError with the reason,
    which never repeats any part of the header: a mistyped header can be
    all secret.
    """
    name, colon, value = header.partition(":")
    if not colon:
        raise ValueError("it has no colon; give it as NAME: VALUE")
    name = name.strip()
    value = value.strip()
    if not name:
        raise ValueError("its name is empty")
    if not all(ch in _TOKEN_CHARS for ch in name):
        raise ValueError(
            "its name may only contain ASCII letters, digits and !#$%&'*+-.^_`|~"
        )
    if not value:
        raise ValueError("its value is empty")
    if not all(" " <= ch <= "~" for ch in value):
        raise ValueError("its value must contain only printable ASCII")
    return name, value


def _http_opener(*extra: urllib.request.BaseHandler) -> urllib.request.OpenerDirector:
    """Return an opener for http and https URLs with urlopen's proxy and error handlers.

    It has no redirect handler unless *extra* adds one. Any other scheme
    fails with "unknown url type".
    """
    opener = urllib.request.OpenerDirector()
    handlers: list[urllib.request.BaseHandler] = [
        urllib.request.ProxyHandler(),
        urllib.request.UnknownHandler(),
        urllib.request.HTTPHandler(),
        urllib.request.HTTPDefaultErrorHandler(),
        urllib.request.HTTPErrorProcessor(),
        *extra,
    ]
    if hasattr(urllib.request, "HTTPSHandler"):
        handlers.append(urllib.request.HTTPSHandler())
    for handler in handlers:
        opener.add_handler(handler)
    return opener


def no_redirect_opener() -> urllib.request.OpenerDirector:
    """Return an opener for POSTs that never follows a redirect.

    urlopen re-sends a POST answered with 301, 302 or 303 as a GET with no
    body, so the data is lost. It keeps every header, so a credential goes
    to the Location URL even on another host. With no redirect handler
    every 3xx raises HTTPError instead. Only http and https URLs can be
    opened.
    """
    return _http_opener()


def _origin(url: str) -> tuple[str, str, int | None] | None:
    """Return the scheme, host name and port of *url*, or None.

    urlsplit gives the scheme and the host name in lower case. A missing
    port is the scheme's default. None means the URL has no host, a port
    that does not parse, or user info. urlsplit takes the host after the
    last @, and a proxy may not: it can read http://a.example\\@b.example/
    as a.example.
    """
    try:
        parts = urllib.parse.urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    if not parts.hostname or "@" in parts.netloc:
        return None
    if port is None:
        port = _DEFAULT_PORTS.get(parts.scheme)
    return parts.scheme, parts.hostname, port


def _same_origin(url: str, other: str) -> bool:
    origin = _origin(url)
    return origin is not None and origin == _origin(other)


class _CrossOriginRedirect(urllib.error.HTTPError):
    """A redirect to another scheme, host or port that was not followed."""


class _SameOriginRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # http_error_302 has already made newurl absolute.
        if not _same_origin(req.full_url, newurl):
            raise _CrossOriginRedirect(req.full_url, code, f"{msg}{_ORIGIN_NOTE}", headers, fp)
        return super().redirect_request(req, fp, code, msg, headers, newurl)

    # Python 3.10 has no 308 handler, so a 308 failed there without the
    # note. With this, a 308 to another origin gets the note on 3.10 too.
    # A 308 to the same origin still fails on 3.10, as urlopen does there.
    http_error_308 = urllib.request.HTTPRedirectHandler.http_error_302


def same_origin_opener() -> urllib.request.OpenerDirector:
    """Return an opener for GETs that send a credential.

    urlopen follows a redirect to any URL and keeps every header but
    Content-Length and Content-Type, so a credential goes to the Location
    host. This opener follows a redirect only when the new URL has the same
    scheme, host name and port, with 80 for http and 443 for https when no
    port is given. Any other redirect raises HTTPError with the redirect's
    code, headers and body, and its reason ends with a note to give the
    final URL. Only http and https URLs can be opened.
    """
    return _http_opener(_SameOriginRedirectHandler())


def redirect_hint(exc: BaseException) -> str:
    """Return the note for a 3xx that a no_redirect_opener request got, or "".

    Add it only to a message about a request that did not follow redirects.
    """
    if isinstance(exc, urllib.error.HTTPError) and 300 <= exc.code < 400:
        return " (redirects are not followed; give the final URL)"
    return ""


def origin_redirect_hint(exc: BaseException) -> str:
    """Return the note for a redirect that same_origin_opener refused, or "".

    The refusal's reason already ends with the note. This is for a message
    that shows the status code but not the reason.
    """
    if isinstance(exc, _CrossOriginRedirect):
        return _ORIGIN_NOTE
    return ""


def post_error_detail(exc: BaseException) -> str:
    """Describe why a POST failed without quoting its URL.

    Keeps the status and reason of an HTTP error and the reason of a network
    error. A server or a proxy writes the reason, so each unprintable
    character in it is shown as an escape. Other exceptions (InvalidURL,
    ValueError, UnicodeEncodeError and more) can repeat the URL, so only
    their class name is given.
    """
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP Error {exc.code}: {inert(exc.reason)}{redirect_hint(exc)}"
    if isinstance(exc, urllib.error.URLError):
        return inert(exc.reason)
    return type(exc).__name__
