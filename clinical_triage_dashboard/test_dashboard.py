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
            page.wait_for_load_state('networkidle')
            
            # Check for title
            title_selector = "h1:has-text('AUTONOMOUS CLINICAL TRIAGE')"
            page.wait_for_selector(title_selector, timeout=10000)
            print("Dashboard title found!")
            
            # Take a screenshot to verify UI
            # page.screenshot(path='c:/Users/sansk/Downloads/Scaler/artifacts/dashboard_v1.png', full_page=True)
            # print("Screenshot saved.")
            
            # Check if linked
            status_selector = "span:has-text('LINK_STABLE')"
            try:
                page.wait_for_selector(status_selector, timeout=5000)
                print("WebSocket connection verified (LINK_STABLE).")
            except:
                print("WebSocket connection not verified yet. Current logs:")
                logs = page.locator("div.flex.gap-2").all_text_contents()
                for log in logs:
                    print(f"  {log}")
            
            # Try to click INIT button
            init_button = "button:has-text('INIT')"
            page.click(init_button)
            print("Clicked INIT button.")
            
            # Wait for patient data to appear
            page.wait_for_timeout(2000)
            patient_card = "div:has-text('P1')"
            if page.locator(patient_card).count() > 0:
                print("Patient P1 found in waiting room.")
            else:
                print("Patient data not found in waiting room.")
                
        except Exception as e:
            print(f"Error during test: {e}")
            # page.screenshot(path='c:/Users/sansk/Downloads/Scaler/artifacts/error_screenshot.png')
        finally:
            browser.close()

if __name__ == "__main__":
    test_dashboard()
