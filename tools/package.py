"""Package a release zip: inkling-<version>-<platform>/{inkling[.exe], README.md, LICENSE, NOTICE, tools/chat.py}
plus a .sha256 file next to it.

Usage: python tools/package.py <binary> <version> <platform> [out_dir]
  e.g. python tools/package.py target/release/inkling.exe 0.2.0 windows-x64 dist
       python3 tools/package.py target/release/inkling 0.2.0 linux-x64 dist

The binary keeps its executable bit inside the zip (Unix permissions in the external attributes), so `unzip` on
Linux restores it.
"""
import hashlib
import os
import sys
import zipfile

if len(sys.argv) < 4:
    sys.exit(__doc__)
binary, version, platform = sys.argv[1:4]
out_dir = sys.argv[4] if len(sys.argv) > 4 else "dist"
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
name = f"inkling-{version}-{platform}"
os.makedirs(out_dir, exist_ok=True)
zpath = os.path.join(out_dir, name + ".zip")

files = [
    (binary, "inkling.exe" if binary.endswith(".exe") else "inkling", 0o755),
    (os.path.join(root, "README.md"), "README.md", 0o644),
    (os.path.join(root, "LICENSE"), "LICENSE", 0o644),
    (os.path.join(root, "NOTICE"), "NOTICE", 0o644),
    (os.path.join(root, "tools", "chat.py"), "tools/chat.py", 0o644),
]
with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for src, arc, mode in files:
        info = zipfile.ZipInfo.from_file(src, f"{name}/{arc}")
        info.external_attr = (0o100000 | mode) << 16  # regular file + permissions
        info.create_system = 3  # Unix, so the permissions are honoured
        info.compress_type = zipfile.ZIP_DEFLATED
        with open(src, "rb") as f:
            z.writestr(info, f.read())

h = hashlib.sha256(open(zpath, "rb").read()).hexdigest()
with open(zpath + ".sha256", "w", newline="\n") as f:
    f.write(f"{h} *{name}.zip\n")
print(f"{zpath}\n{h}")
