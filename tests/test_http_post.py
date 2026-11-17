"""Tests for same_origin_opener, the opener for GETs that send a credential."""

from __future__ import annotations

import io
import json
import os
import sys
import unittest
import urllib.error
import urllib.request
from http import HTTPStatus
from unittest.mock import patch

from agent_trace import http_post

try:
    from _redirect_servers import ALL_REDIRECTS, MOVED, ORIGIN_NOTE, RedirectServers
except ImportError:
    from tests._redirect_servers import ALL_REDIRECTS, MOVED, ORIGIN_NOTE, RedirectServers

TOKEN = "Bearer origin-test-token"
REPLY = {"ok": True}
# urllib follows a 308 from Python 3.11. Python 3.10 raises HTTPError 308.
FOLLOWED = (301, 302, 303, 307, 308) if sys.version_info >= (3, 11) else (301, 302, 303, 307)


def _get(url: str) -> tuple[int, object]:
    request = urllib.request.Request(url, headers={"Authorization": TOKEN})
    with http_post.same_origin_opener().open(request, timeout=5) as response:
        return response.status, json.loads(response.read())


def _redirect_handler() -> urllib.request.HTTPRedirectHandler:
    handlers = http_post.same_origin_opener().handlers
    return next(h for h in handlers if isinstance(h, urllib.request.HTTPRedirectHandler))


class TestSameOriginOpener(unittest.TestCase):
    def setUp(self):
        self.servers = RedirectServers(self, methods=("GET",), reply=REPLY)
        self.port = self.servers.a.server_address[1]

    def _error(self, url: str) -> urllib.error.HTTPError:
        with self.assertRaises(urllib.error.HTTPError) as caught:
            _get(url)
        self.addCleanup(caught.exception.close)
        return caught.exception

    def test_follows_a_redirect_to_the_same_origin_with_the_credential(self):
        servers = self.servers
        for status in FOLLOWED:
            with self.subTest(status=status):
                servers.redirect(status, servers.moved)
                self.assertEqual(_get(servers.url + "/x"), (200, REPLY))
                self.assertEqual(
                    [(r.method, r.path, r.headers["Authorization"]) for r in servers.a.received()],
                    [("GET", "/x", TOKEN), ("GET", "/moved/x", TOKEN)],
                )
                self.assertEqual(servers.b.received(), [])

    @unittest.skipIf(sys.version_info >= (3, 11), "urllib follows a 308 from Python 3.11")
    def test_python_3_10_does_not_follow_a_308_to_the_same_origin(self):
        # urlopen fails the same way there.
        self.servers.redirect(308, self.servers.moved)
        error = self._error(self.servers.url + "/x")
        self.assertEqual((error.code, error.reason), (308, "Permanent Redirect"))
        self.assertEqual(self.servers.a.methods(), ["GET"])

    def test_refuses_a_redirect_to_another_origin(self):
        # urlopen sent the credential to the Location host.
        servers = self.servers
        targets = {
            "port": servers.b.url,
            "host name": f"http://localhost:{self.port}{MOVED}",
            "scheme": f"https://127.0.0.1:{self.port}{MOVED}",
        }
        for other, to in targets.items():
            for status in ALL_REDIRECTS:
                with self.subTest(other=other, status=status):
                    servers.redirect(status, to)
                    error = self._error(servers.url + "/x")
                    self.assertEqual(error.code, status)
                    self.assertEqual(error.reason, HTTPStatus(status).phrase + ORIGIN_NOTE)
                    self.assertEqual(error.headers["Location"], to + "/x")
                    self.assertEqual(http_post.origin_redirect_hint(error), ORIGIN_NOTE)
                    self.assertEqual([r.path for r in servers.a.received()], ["/x"])
                    self.assertEqual(servers.b.received(), [])

    def test_a_same_origin_loop_keeps_urllibs_message(self):
        servers = self.servers
        servers.redirect(302, servers.url)
        error = self._error(servers.url + "/x")
        self.assertEqual(error.code, 302)
        self.assertEqual(error.reason, urllib.request.HTTPRedirectHandler.inf_msg + "Found")
        self.assertEqual(http_post.origin_redirect_hint(error), "")
        self.assertEqual(servers.b.received(), [])

    def test_other_errors_keep_their_message(self):
        servers = self.servers
        for status in (300, 304, 404, 500):
            with self.subTest(status=status):
                servers.redirect(status)
                error = self._error(servers.url + "/x")
                self.assertEqual((error.code, error.reason), (status, HTTPStatus(status).phrase))
                self.assertEqual(http_post.origin_redirect_hint(error), "")
                self.assertEqual(servers.b.received(), [])

    def test_uses_a_proxy_from_the_environment(self):
        # ProxyHandler reads the environment when the opener is built. B
        # answers every request with 200, so it stands in for the proxy.
        proxy = self.servers.b
        url = "http://collector.invalid/x"
        environ = {
            "http_proxy": proxy.url, "HTTP_PROXY": proxy.url, "no_proxy": "", "NO_PROXY": "",
        }
        with patch.dict(os.environ, environ):
            opener = http_post.same_origin_opener()
            request = urllib.request.Request(url, headers={"Authorization": TOKEN})
            with opener.open(request, timeout=5) as response:
                self.assertEqual(response.status, 200)
        self.assertEqual([(r.method, r.path) for r in proxy.received()], [("GET", url)])

    def test_only_http_and_https_urls_can_be_opened(self):
        with patch.dict(os.environ, {"ftp_proxy": "", "FTP_PROXY": ""}):
            for url in ("file:///etc/hosts", f"ftp://127.0.0.1:{self.port}/x"):
                with self.subTest(url=url):
                    with self.assertRaises(urllib.error.URLError) as caught:
                        http_post.same_origin_opener().open(url, timeout=5)
                    self.assertIn("unknown url type", str(caught.exception.reason))
        self.assertEqual(self.servers.a.received(), [])


class TestOriginCheck(unittest.TestCase):
    """http_error_302 passes redirect_request the absolute Location URL."""

    def _follows(self, url: str, newurl: str) -> bool:
        request = urllib.request.Request(url, headers={"Authorization": TOKEN})
        try:
            new = _redirect_handler().redirect_request(
                request, io.BytesIO(), 302, "Found", {}, newurl
            )
        except urllib.error.HTTPError as error:
            error.close()
            self.assertEqual(error.reason, "Found" + ORIGIN_NOTE)
            return False
        self.assertEqual(new.full_url, newurl)
        self.assertEqual(new.get_header("Authorization"), TOKEN)
        return True

    def test_same_origin_is_scheme_host_and_port(self):
        cases = [
            # The host name is compared case-insensitively.
            ("http://example.com/a", "http://EXAMPLE.com/b", True),
            ("http://Example.COM/a", "http://example.com/b", True),
            ("http://example.com/a", "http://example.org/b", False),
            # A missing port is the scheme's default.
            ("http://h/a", "http://h:80/b", True),
            ("http://h:80/a", "http://h/b", True),
            ("https://h/a", "https://h:443/b", True),
            ("http://h/a", "http://h:443/b", False),
            ("https://h/a", "https://h:80/b", False),
            ("http://h:8080/a", "http://h:8081/b", False),
            ("http://h:443/a", "https://h/b", False),
            ("https://h/a", "http://h/b", False),
            # A port that does not parse is another origin.
            ("http://h/a", "http://h:abc/b", False),
            ("http://h/a", "http://h:99999/b", False),
            # So is any URL with user info. A proxy may read the host
            # before the backslash, evil.example.
            ("http://h/a", "http://user@h/b", False),
            ("http://h/a", "http://evil.example\\@h/b", False),
            ("http://user@h/a", "http://user@h/b", False),
        ]
        for url, newurl, followed in cases:
            with self.subTest(url=url, newurl=newurl):
                self.assertIs(self._follows(url, newurl), followed)

    def test_refusal_keeps_the_redirect_code_headers_and_body(self):
        headers = {"Location": "http://other.example/b"}
        request = urllib.request.Request("http://h/a")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            _redirect_handler().redirect_request(
                request, io.BytesIO(b"moved"), 307, "Temporary Redirect", headers,
                "http://other.example/b",
            )
        error = caught.exception
        self.addCleanup(error.close)
        self.assertEqual((error.url, error.code), ("http://h/a", 307))
        self.assertIs(error.headers, headers)
        self.assertEqual(error.read(), b"moved")

    def test_origin_redirect_hint_is_only_for_a_refused_redirect(self):
        # A server can send any reason, the note too.
        error = urllib.error.HTTPError("http://h/x", 302, "Found" + ORIGIN_NOTE, {}, None)
        self.addCleanup(error.close)
        self.assertEqual(http_post.origin_redirect_hint(error), "")
        self.assertEqual(http_post.origin_redirect_hint(ValueError("x")), "")


if __name__ == "__main__":
    unittest.main()
