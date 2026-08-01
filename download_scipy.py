import json
import urllib.request
import os
import sys
import time
import subprocess

def get_wheel_url(package_name, py_version="cp312", platform="win_amd64"):
    api_url = f"https://pypi.org/pypi/{package_name}/json"
    print(f"Fetching metadata from {api_url}...")
    req = urllib.request.Request(api_url)
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode('utf-8'))
    
    version = data['info']['version']
    print(f"Latest version of {package_name} is {version}")
    
    for release in data['urls']:
        filename = release['filename']
        if py_version in filename and platform in filename and filename.endswith(".whl"):
            return release['url'], filename
            
    # Try looking in specific releases if not in top-level urls (sometimes pypi does this)
    for v, releases in data['releases'].items():
        if v == version:
            for release in releases:
                filename = release['filename']
                if py_version in filename and platform in filename and filename.endswith(".whl"):
                    return release['url'], filename

    raise Exception(f"Could not find a matching wheel for {py_version} {platform}")

def download_with_resume(url, filename):
    headers = {}
    if os.path.exists(filename):
        downloaded = os.path.getsize(filename)
        headers['Range'] = f'bytes={downloaded}-'
    else:
        downloaded = 0

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            total_size = int(response.info().get('Content-Length', 0)) + downloaded
            print(f"Downloading {filename}: {downloaded}/{total_size} bytes")
            mode = 'ab' if downloaded > 0 else 'wb'
            with open(filename, mode) as f:
                last_print = downloaded
                while True:
                    chunk = response.read(8192 * 8)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if downloaded - last_print > 1024 * 1024:
                        print(f"Progress: {downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB")
                        last_print = downloaded
            print("Download completed!")
            return True
    except urllib.error.HTTPError as e:
        if e.code == 416: # Range not satisfiable (already fully downloaded)
            print("Already fully downloaded!")
            return True
        print(f"HTTP Error: {e}")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False

def main():
    package_name = "scipy"
    print("Finding URL for scipy...")
    
    url, filename = get_wheel_url(package_name)
    print(f"Found URL: {url}")
    
    os.makedirs("wheels", exist_ok=True)
    filepath = os.path.join("wheels", filename)
    
    # Try downloading with resume
    max_retries = 20
    for i in range(max_retries):
        print(f"Attempt {i+1}/{max_retries}")
        if download_with_resume(url, filepath):
            print("Successfully downloaded!")
            subprocess.run([sys.executable, "-m", "pip", "install", filepath])
            break
        print("Retrying in 2 seconds...")
        time.sleep(2)

if __name__ == "__main__":
    main()
