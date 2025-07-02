from selenium import webdriver
from selenium.webdriver.edge.service import Service
from selenium.webdriver.edge.options import Options
from PIL import Image, ImageDraw
import base64
import time
import os

# === CONFIG ===
EDGE_DRIVER_PATH = 'msedgedriver.exe'
TARGET_URL = 'https://arxiv.org/'
OUTPUT_BASE = 'output_with_boxes'

VIEWPORTS = {
    'desktop': 1600,
    'mobile': 412
}

def capture_screenshot_with_boxes(driver, output_path, viewport_width):
    # Set explicit viewport size (browser outer dimensions need to be larger)
    browser_width = viewport_width + 100  # Extra space for browser chrome in non-headless
    browser_height = 1000  # Temporary height, will be adjusted
    driver.set_window_size(browser_width, browser_height)
    
    # Set the actual viewport size through JavaScript
    driver.execute_script(f"document.body.style.width = '{viewport_width}px';")
    time.sleep(1)  # Allow resize to take effect
    
    # Get page dimensions and set browser height accordingly
    total_height = driver.execute_script("return Math.max(document.body.scrollHeight, document.body.offsetHeight, document.documentElement.clientHeight, document.documentElement.scrollHeight, document.documentElement.offsetHeight);")
    driver.set_window_size(browser_width, total_height)
    
    # === GET BOUNDING BOXES ===
    elements = driver.execute_script("""
    return Array.from(document.querySelectorAll('a, button, input, textarea, select, [tabindex]'))
      .filter(el => {
        const rect = el.getBoundingClientRect();
        return rect.width > 0 && rect.height > 0 &&
               !!(el.offsetWidth || el.offsetHeight || el.getClientRects().length);
      })
      .map(el => {
        const r = el.getBoundingClientRect();
        return {
          left: r.left,
          top: r.top,
          right: r.right,
          bottom: r.bottom,
          width: r.width,
          height: r.height,
          tag: el.tagName
        };
      });
    """)
    
    # === CAPTURE SCREENSHOT ===
    params = {
        "format": "png",
        "clip": {
            "x": 0,
            "y": 0,
            "width": viewport_width,
            "height": total_height,
            "scale": 1
        }
    }
    
    screenshot_base64 = driver.execute_cdp_cmd("Page.captureScreenshot", params)['data']
    image_data = base64.b64decode(screenshot_base64)
    
    # === DRAW BOXES ===
    with open("temp.png", "wb") as f:
        f.write(image_data)
    
    img = Image.open("temp.png")
    draw = ImageDraw.Draw(img)
    
    for el in elements:
        box = [el['left'], el['top'], el['right'], el['bottom']]
        # Thick red background
        draw.rectangle(box, outline="red", width=3)
        # Thin green foreground
        draw.rectangle(box, outline="green", width=1)
        
        # Label with background
        text = f"{el['tag']}"
        text_pos = (el['left'] + 2, el['top'] + 2)
        text_bbox = draw.textbbox(text_pos, text)
        draw.rectangle(text_bbox, fill="white")
        draw.text(text_pos, text, fill="black")
    
    img.save(output_path)
    os.remove("temp.png")
    print(f"Saved: {output_path} ({img.width}x{img.height}px)")

# === MAIN ===
options = Options()
options.add_argument("--headless=new")
options.add_argument("--force-device-scale-factor=1")  # Prevent scaling

driver = webdriver.Edge(service=Service(EDGE_DRIVER_PATH), options=options)

try:
    driver.get(TARGET_URL)
    time.sleep(2)
    
    # Regular desktop and mobile views
    for name, width in VIEWPORTS.items():
        # Reset to normal desktop mode
        options = Options()
        options.add_argument("--headless=new")
        options.add_argument(f"--window-size={width},1000")
        
        if name == 'mobile':
            # Mobile emulation for more accurate rendering
            mobile_emulation = {
                "deviceMetrics": {"width": width, "height": 800, "pixelRatio": 1.0},
                "userAgent": "Mozilla/5.0 (iPhone; CPU iPhone OS 15_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.0 Mobile/15E148 Safari/604.1"
            }
            options.add_experimental_option("mobileEmulation", mobile_emulation)
        
        driver.quit()
        driver = webdriver.Edge(service=Service(EDGE_DRIVER_PATH), options=options)
        driver.get(TARGET_URL)
        time.sleep(2)
        
        output_file = f"{OUTPUT_BASE}_{name}.png"
        capture_screenshot_with_boxes(driver, output_file, width)
        
finally:
    driver.quit()
