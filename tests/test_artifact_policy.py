from __future__ import annotations

from design_gan.artifact_policy import ArtifactPolicy, validate_html

VALID = "<!doctype html><html><body><button>Start</button></body></html>"


def test_standalone_offline_html_passes():
    result = validate_html(VALID)
    assert result.passed is True
    assert result.violations == []


def test_external_assets_and_network_apis_are_rejected():
    html = '<html><body><img src="/asset.png"><script>fetch("/collect")</script></body></html>'
    result = validate_html(html)
    assert result.passed is False
    assert any("external URL" in item for item in result.violations)
    assert any("network API" in item for item in result.violations)


def test_fragment_links_and_embedded_data_remain_valid_offline():
    html = (
        '<html><body><a href="#details">Details</a>'
        '<img alt="dot" src="data:image/gif;base64,R0lGODlhAQABAIAAAAUEBA==">'
        '<section id="details">Done</section></body></html>'
    )
    assert validate_html(html).passed is True


def test_external_css_and_additional_network_capabilities_are_rejected():
    html = (
        "<html><head><style>body{background:url(https://example.com/x.png)}</style></head>"
        "<body><script>navigator.sendBeacon('/collect', 'x')</script></body></html>"
    )
    result = validate_html(html)
    assert result.passed is False
    assert any("external URL found in CSS" in item for item in result.violations)
    assert any("network API" in item for item in result.violations)


def test_embedded_documents_are_rejected():
    result = validate_html("<html><body><iframe srcdoc='<p>x</p>'></iframe></body></html>")
    assert result.passed is False
    assert "embedded documents" in result.violations[0]


def test_maximum_size_is_versioned_policy():
    result = validate_html(VALID, ArtifactPolicy(max_bytes=10))
    assert result.passed is False
    assert "limit is 10" in result.violations[0]
