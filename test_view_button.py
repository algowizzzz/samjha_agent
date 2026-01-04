# Test script to verify View button functionality
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

# Use requests to login first and get session
session = requests.Session()
login_resp = session.post('http://localhost:5000/api/login', 
                         json={'username': 'admin', 'password': 'admin'})

if login_resp.status_code == 200:
    print('✅ Logged in successfully')
    
    # Get cookie
    cookie = session.cookies.get_dict()
    print(f'Cookie: {cookie}')
    
    # Now use selenium
    driver = webdriver.Chrome()
    driver.get('http://localhost:5000/bulk-doc-analysis')
    
    # Add cookie
    if 'session' in cookie:
        driver.add_cookie({'name': 'session', 'value': cookie['session']})
        driver.refresh()
    
    time.sleep(3)
    
    # Look for View button
    try:
        view_buttons = driver.find_elements(By.CSS_SELECTOR, 'button[data-action="view-run"]')
        print(f'Found {len(view_buttons)} View buttons')
        
        if view_buttons:
            print(f'Clicking first View button...')
            view_buttons[0].click()
            time.sleep(3)
            
            # Check if run progress section is visible
            run_progress = driver.find_elements(By.ID, 'runProgressSection')
            if run_progress and run_progress[0].is_displayed():
                print('✅ Run progress section is visible')
                
                # Check for table rows
                rows = driver.find_elements(By.CSS_SELECTOR, '#bulkDocRunTbody tr')
                print(f'Found {len(rows)} rows in run progress table')
                
                if rows:
                    print('✅ Run progress table populated!')
                    for i, row in enumerate(rows[:3]):  # Show first 3 rows
                        print(f'Row {i+1}: {row.text[:100]}')
            else:
                print('❌ Run progress section not visible')
        else:
            print('❌ No View buttons found')
            # Check if table exists
            table = driver.find_elements(By.ID, 'recentRunsList')
            if table:
                print(f'Table exists, content: {table[0].text[:200]}')
    except Exception as e:
        print(f'Error: {e}')
        import traceback
        traceback.print_exc()
    
    driver.quit()
else:
    print(f'❌ Login failed: {login_resp.status_code}')
