@echo off
echo ========================================
echo  Stock Analysis AI - Installation
echo ========================================
echo.

echo [1/3] Installing Python dependencies...
pip install -r requirements.txt

echo.
echo [2/3] Setting up environment file...
if not exist .env (
    copy .env.example .env
    echo .env file created. Edit it to add your API keys (optional).
) else (
    echo .env already exists, skipping.
)

echo.
echo [3/3] Installation complete!
echo.
echo ========================================
echo  To run the app:
echo  streamlit run app.py
echo ========================================
pause
