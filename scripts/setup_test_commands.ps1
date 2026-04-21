# PowerShell wrapper untuk run commands dengan venv

# Function untuk quick test
function Run-QuickTest {
    param(
        [string]$WorkDir = "D:\semester-4-IT Del\Semester VI\UI-UX DESIGN\Scrapping_Data"
    )
    
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host "Quick Test - Sentiment Model V4" -ForegroundColor Cyan
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host ""
    
    $pythonExe = Join-Path $WorkDir ".venv\Scripts\python.exe"
    $script = Join-Path $WorkDir "quick_test.py"
    
    if (-not (Test-Path $pythonExe)) {
        Write-Host "❌ Error: Python executable not found at $pythonExe" -ForegroundColor Red
        return
    }
    
    if (-not (Test-Path $script)) {
        Write-Host "❌ Error: Script not found at $script" -ForegroundColor Red
        return
    }
    
    & $pythonExe $script
}

# Function untuk interactive test
function Run-InteractiveTest {
    param(
        [string]$WorkDir = "D:\semester-4-IT Del\Semester VI\UI-UX DESIGN\Scrapping_Data"
    )
    
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host "Interactive Test - Sentiment Model V4" -ForegroundColor Cyan
    Write-Host "======================================================================" -ForegroundColor Cyan
    Write-Host ""
    
    $pythonExe = Join-Path $WorkDir ".venv\Scripts\python.exe"
    $script = Join-Path $WorkDir "test_interactive.py"
    
    if (-not (Test-Path $pythonExe)) {
        Write-Host "❌ Error: Python executable not found at $pythonExe" -ForegroundColor Red
        return
    }
    
    if (-not (Test-Path $script)) {
        Write-Host "❌ Error: Script not found at $script" -ForegroundColor Red
        return
    }
    
    & $pythonExe $script
}

# Function untuk test single text
function Test-SingleText {
    param(
        [string]$Text,
        [string]$ModelDir = "sentiment_model/model_artifacts_v4",
        [string]$WorkDir = "D:\semester-4-IT Del\Semester VI\UI-UX DESIGN\Scrapping_Data"
    )
    
    Write-Host "Testing: '$Text'" -ForegroundColor Cyan
    Write-Host ""
    
    $pythonExe = Join-Path $WorkDir ".venv\Scripts\python.exe"
    $script = Join-Path $WorkDir "sentiment_model\test_model.py"
    
    if (-not (Test-Path $pythonExe)) {
        Write-Host "❌ Error: Python executable not found" -ForegroundColor Red
        return
    }
    
    & $pythonExe $script --model-dir $ModelDir --text $Text
}

# Aliases untuk convenience
Set-Alias -Name quick-test -Value Run-QuickTest -Force
Set-Alias -Name interactive-test -Value Run-InteractiveTest -Force
Set-Alias -Name test-text -Value Test-SingleText -Force

Write-Host "======================================================================" -ForegroundColor Green
Write-Host "Sentiment Model Test Functions Loaded" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Available commands:" -ForegroundColor Yellow
Write-Host "  quick-test                    - Run quick test with 5 examples" -ForegroundColor White
Write-Host "  interactive-test              - Interactive mode (type manually)" -ForegroundColor White
Write-Host "  test-text ""your text here""   - Test single text" -ForegroundColor White
Write-Host ""
Write-Host "Examples:" -ForegroundColor Yellow
Write-Host "  quick-test" -ForegroundColor Gray
Write-Host "  test-text ""bau pesing""" -ForegroundColor Gray
Write-Host "  test-text ""toilet kotor banget""" -ForegroundColor Gray
Write-Host ""
