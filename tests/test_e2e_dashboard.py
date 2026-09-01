import re
from playwright.sync_api import Page, expect
import pytest

@pytest.fixture(scope="session")
def dashboard_url():
    # Streamlit dashboard URL
    return "http://localhost:8501"

def test_dashboard_loads_and_displays_simulator(page: Page, dashboard_url: str):
    page.goto(dashboard_url)
    
    # Wait for Streamlit to render the title
    expect(page.locator("h1").filter(has_text="Fraud Risk Agent Dashboard")).to_be_visible(timeout=10000)
    
    # Ensure Simulator section exists in sidebar
    expect(page.locator("h2").filter(has_text="Simulator")).to_be_visible()
    
    # Check the "Send High-Risk Transaction" button
    high_risk_btn = page.locator("button").filter(has_text="Send High-Risk Transaction")
    expect(high_risk_btn).to_be_visible()
    
    # Click it to trigger a transaction
    high_risk_btn.click()
    
    # Since Streamlit reruns every 2 seconds, we wait for a red ACTION_BLOCK in the table
    # The table might take a few seconds to poll the DB and update
    action_block = page.locator("text=ACTION_BLOCK").first
    expect(action_block).to_be_visible(timeout=15000)
    
    # Verify the SHAP chart appears
    shap_title = page.locator("h3").filter(has_text="Latest Block Reasons (SHAP)")
    expect(shap_title).to_be_visible()
