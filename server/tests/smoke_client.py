"""E2E smoke test against a running DepthWizard server."""
import os
import sys
import tempfile
import time

import httpx

BASE = "http://127.0.0.1:8000"
IMG = os.environ.get(
    "SMOKE_TEST_IMG", os.path.join(tempfile.gettempdir(), "depthwizard_smoke.png")
)


def main():
    c = httpx.Client(timeout=120.0)
    print("health:", c.get(BASE + "/health").json())

    with open(IMG, "rb") as f:
        r = c.post(BASE + "/upload", files={"file": ("smoke.png", f, "image/png")})
    print("upload:", r.status_code, r.json())
    up = r.json()

    r = c.post(f"{BASE}/process/{up['upload_id']}")
    print("process:", r.status_code, r.json())
    job = r.json()["job_id"]

    last = None
    for _ in range(200):
        st = c.get(f"{BASE}/status/{job}").json()
        if st["status"] != last:
            print("status:", st)
            last = st["status"]
        if st["status"] in ("done", "failed"):
            break
        time.sleep(0.5)

    if st["status"] == "done":
        res = c.get(f"{BASE}/result/{job}").json()
        print("result:")
        for k, v in res.items():
            print(f"   {k}: {v}")
        # fetch heightmap bytes
        hm = c.get(BASE + res["heightmap_url"])
        print("heightmap fetch:", hm.status_code, "bytes:", len(hm.content))
        with open(os.path.join(tempfile.gettempdir(), "depthwizard_out_heightmap.png"), "wb") as f:
            f.write(hm.content)
    else:
        print("JOB FAILED")

    return 0


if __name__ == "__main__":
    sys.exit(main())
