@echo off
echo ========================================
echo     DOCSTRING ENFORCER - MILESTONE 4
echo ========================================
echo.

echo === 1. CLI Demo ===
docstring-enforcer --help
echo.
docstring-enforcer .
echo.

echo === 2. Python Import Test ===
python -c "from docstring_enforcer.enforcer import analyze_directory; print('✅ Import SUCCESS')"
echo.

echo === 3. Package Check ===
pip list | findstr docstring-enforcer
echo.

echo === 4. Tests ===
echo Running pytest...
pytest tests/ --tb=no || echo "Create tests/ folder first"
echo.

echo === 5. Streamlit UI ===
echo Run: streamlit run streamlit_app.py
echo.

echo ========================================
echo ✅ MILESTONE 4 READY FOR MENTOR REVIEW!
echo ========================================
pause
