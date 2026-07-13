from nunspark.cli import build_parser


def test_web_subcommand_parses():
    parser = build_parser()
    args = parser.parse_args(["web", "--packed-root", "/packs", "--port", "9000"])
    assert args.command == "web"
    assert args.packed_root == "/packs"
    assert args.port == 9000
    assert args.host == "127.0.0.1"
