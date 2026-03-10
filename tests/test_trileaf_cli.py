from __future__ import annotations

from pathlib import Path

import trileaf_cli


def test_remove_subcommand_dispatches(monkeypatch) -> None:
    seen: dict[str, bool] = {}

    def fake_remove(args) -> None:
        seen["yes"] = args.yes
        seen["purge_source"] = args.purge_source

    monkeypatch.setattr(trileaf_cli, "_cmd_remove", fake_remove)
    trileaf_cli.main(["remove", "--yes", "--purge-source"])

    assert seen == {"yes": True, "purge_source": True}


def test_clean_installer_block_removes_added_path_entry(tmp_path: Path) -> None:
    rc = tmp_path / ".bashrc"
    rc.write_text(
        'export FOO="1"\n'
        "\n"
        "# Added by Trileaf installer\n"
        'export PATH="$HOME/.local/bin:$PATH"\n'
        'alias ll="ls -lah"\n',
        encoding="utf-8",
    )

    trileaf_cli._clean_installer_block(rc)

    assert rc.read_text(encoding="utf-8") == (
        'export FOO="1"\n'
        'alias ll="ls -lah"\n'
    )
