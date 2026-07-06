"""Unit tests for `app.utils.helpers`."""

import pytest

from app.utils.helpers import bytes_to_mb, extract_json_object, get_file_extension, new_correlation_id


def test_new_correlation_id_is_unique_and_short():
    a, b = new_correlation_id(), new_correlation_id()
    assert a != b
    assert len(a) == 12


@pytest.mark.parametrize(
    "filename,expected",
    [("call.mp3", "mp3"), ("recording.WAV", "wav"), ("no_extension", ""), ("archive.tar.gz", "gz")],
)
def test_get_file_extension(filename, expected):
    assert get_file_extension(filename) == expected


def test_extract_json_object_plain_json():
    assert extract_json_object('{"a": 1}') == {"a": 1}


def test_extract_json_object_with_markdown_fence():
    text = '```json\n{"a": 1, "b": 2}\n```'
    assert extract_json_object(text) == {"a": 1, "b": 2}


def test_extract_json_object_with_surrounding_prose():
    text = 'Here is the result: {"a": 1} - hope that helps!'
    assert extract_json_object(text) == {"a": 1}


def test_extract_json_object_raises_on_invalid():
    with pytest.raises(ValueError):
        extract_json_object("not json at all")


def test_bytes_to_mb():
    assert bytes_to_mb(1024 * 1024) == 1.0
