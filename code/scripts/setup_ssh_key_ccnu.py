"""Bootstrap SSH key auth using password from environment variable."""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


def _ensure_keypair(private_key_path: Path) -> tuple[Path, Path]:
    private_key_path = private_key_path.expanduser().resolve()
    public_key_path = Path(str(private_key_path) + ".pub")
    if private_key_path.exists() and public_key_path.exists():
        return private_key_path, public_key_path

    private_key_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ssh-keygen",
        "-t",
        "ed25519",
        "-f",
        str(private_key_path),
        "-N",
        "",
    ]
    completed = subprocess.run(cmd, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"ssh-keygen failed with rc={completed.returncode}")
    if not private_key_path.exists() or not public_key_path.exists():
        raise FileNotFoundError(f"Expected key files missing: {private_key_path} / {public_key_path}")
    return private_key_path, public_key_path


def _append_remote_key(*, host: str, port: int, user: str, password: str, public_key: str) -> None:
    try:
        import paramiko
    except Exception as exc:
        raise RuntimeError(
            "paramiko is required for password-based bootstrap. "
            "Install it first: `python -m pip install paramiko`."
        ) from exc

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        hostname=host,
        port=int(port),
        username=user,
        password=password,
        look_for_keys=False,
        allow_agent=False,
        timeout=15,
    )
    try:
        client.exec_command("mkdir -p ~/.ssh && chmod 700 ~/.ssh")
        sftp = client.open_sftp()
        remote_auth = ".ssh/authorized_keys"
        existing = ""
        try:
            with sftp.open(remote_auth, "r") as handle:
                existing = handle.read().decode("utf-8")
        except Exception:
            existing = ""

        normalized_key = public_key.strip()
        if normalized_key not in existing:
            with sftp.open(remote_auth, "a") as handle:
                if existing and not existing.endswith("\n"):
                    handle.write("\n")
                handle.write(normalized_key + "\n")
        sftp.chmod(remote_auth, 0o600)
    finally:
        client.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap SSH key auth to CCNU 4090 server")
    parser.add_argument("--host", default="10.102.65.27")
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--user", default="wxb")
    parser.add_argument("--password_env", default="CCNU_SSH_PASSWORD")
    parser.add_argument("--key_path", default="~/.ssh/id_ed25519_ccnu4090")
    parser.add_argument("--print_test_command", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    password = str(os.environ.get(args.password_env, "")).strip()
    if not password:
        raise RuntimeError(
            f"Environment variable {args.password_env} is empty. "
            "Set it first, then rerun."
        )

    private_key, public_key_path = _ensure_keypair(Path(args.key_path))
    public_key = public_key_path.read_text(encoding="utf-8").strip()
    if not public_key:
        raise RuntimeError(f"Public key file is empty: {public_key_path}")

    _append_remote_key(
        host=str(args.host),
        port=int(args.port),
        user=str(args.user),
        password=password,
        public_key=public_key,
    )

    print("[SSH-BOOTSTRAP] completed")
    print(f"[SSH-BOOTSTRAP] private_key={private_key}")
    print(f"[SSH-BOOTSTRAP] public_key={public_key_path}")
    if bool(args.print_test_command):
        print(
            "[SSH-BOOTSTRAP] test_cmd="
            f"ssh -i {private_key} -o BatchMode=yes -o ConnectTimeout=8 {args.user}@{args.host} \"echo ok\""
        )


if __name__ == "__main__":
    main()
