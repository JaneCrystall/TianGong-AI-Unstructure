from __future__ import annotations

import unittest

from src.kb_parse_worker.chunk_splitter import split_chunks_for_embedding


class CharCodec:
    def count(self, text: str) -> int:
        return len(text)

    def split(self, text: str, max_tokens: int) -> list[str]:
        return [text[offset : offset + max_tokens] for offset in range(0, len(text), max_tokens)]


class KbParseWorkerChunkSplitterTests(unittest.TestCase):
    def test_splits_text_on_sentence_boundaries(self) -> None:
        chunks, stats = split_chunks_for_embedding(
            [{"text": "第一句。第二句。第三句。", "page_number": 1, "type": "text"}],
            "doc-1",
            9,
            CharCodec(),
        )

        self.assertEqual([chunk["text"] for chunk in chunks], ["第一句。第二句。", "第三句。"])
        self.assertEqual(stats.source_chunk_count, 1)
        self.assertEqual(stats.output_chunk_count, 2)
        self.assertEqual(stats.split_parent_count, 1)
        self.assertEqual(set(chunks[0]), {"text", "page_number", "type"})
        self.assertEqual(set(chunks[1]), {"text", "page_number", "type"})

    def test_splits_html_table_on_row_boundaries(self) -> None:
        table = "<table><tr><td>aa</td></tr><tr><td>bb</td></tr><tr><td>cc</td></tr></table>"
        chunks, stats = split_chunks_for_embedding(
            [{"text": table, "page_number": 2}],
            "doc-1",
            50,
            CharCodec(),
        )

        self.assertEqual(stats.output_chunk_count, 3)
        self.assertTrue(all(chunk["text"].startswith("<table>") for chunk in chunks))
        self.assertTrue(all(chunk["text"].endswith("</table>") for chunk in chunks))
        self.assertIn("<tr><td>aa</td></tr>", chunks[0]["text"])
        self.assertIn("<tr><td>bb</td></tr>", chunks[1]["text"])
        self.assertIn("<tr><td>cc</td></tr>", chunks[2]["text"])

    def test_hard_splits_single_oversized_sentence(self) -> None:
        chunks, _ = split_chunks_for_embedding(
            [{"text": "abcdefghij", "page_number": 3}],
            "doc-1",
            4,
            CharCodec(),
        )

        self.assertEqual([chunk["text"] for chunk in chunks], ["abc", "def", "ghi", "j"])
        self.assertTrue(all(len(chunk["text"]) < 4 for chunk in chunks))

    def test_unsplit_chunks_keep_original_keys_only(self) -> None:
        chunks, stats = split_chunks_for_embedding(
            [{"text": "short", "page_number": 4}],
            "doc-1",
            8000,
            CharCodec(),
        )

        self.assertEqual(stats.output_chunk_count, 1)
        self.assertEqual(chunks[0], {"text": "short", "page_number": 4})


if __name__ == "__main__":
    unittest.main()
