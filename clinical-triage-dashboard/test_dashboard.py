from playwright.sync_api import sync_playwright
import time

def test_dashboard():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()
        
        try:
            print("Navigating to http://localhost:3000...")
            page.goto('http://localhost:3000')
            
            # Wait for content to load
            print("Waiting for network idle...")
            page.wait_for_load_state('networkidle')
            
            # Check for title
            print("Waiting for dashboard title...")
            title_selector = "h1:has-text('AUTONOMOUS CLINICAL TRIAGE')"
            page.wait_for_selector(title_selector, timeout=20000)
            print("Dashboard title found!")
            
            # Check if linked
            print("Checking websocket status...")
            status_selector = "span:has-text('LINK_STABLE')"
            try:
                page.wait_for_selector(status_selector, timeout=10000)
                print("WebSocket connection verified (LINK_STABLE).")
            except:
                print("WebSocket connection not verified yet. Current spans:")
                spans = page.locator("span").all_text_contents()
                for span in spans:
                    if "LINK" in span:
                        print(f"  Found status: {span}")
            
            # Try to click INIT button
            print("Clicking INIT button...")
            init_button = "button:has-text('INIT')"
            page.click(init_button)
            
            # Wait for patient data to appear
            print("Waiting for patient data...")
            page.wait_for_timeout(3000)
            
            if page.locator("text=P1").count() > 0:
                print("Patient P1 found in waiting room.")
            else:
                print("Patient data not found. Current text on page:")
                print(page.inner_text("body")[:500])
                
        except Exception as e:
            print(f"Error during test: {e}")
        finally:
            browser.close()

if __name__ == "__main__":
    test_dashboard()
